import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.agent.service import SYSTEM_PROMPT
from app.agent.tools import tool_schemas
from app.config import Settings, get_settings
from app.rag.loader import load_markdown
from app.rag.splitter import split_pages

VALID_TOOLS = {"search_documents", "search_articles"}
MULTIPLE_TOOLS = "__multiple_tools__"


@dataclass(frozen=True)
class RoutingCase:
    question: str
    expected_tool: str | None


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_filename: str


def load_cases(path: Path) -> tuple[list[RoutingCase], list[RetrievalCase]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    routing = [
        RoutingCase(
            question=str(item["question"]),
            expected_tool=item.get("expected_tool"),
        )
        for item in payload["routing"]
    ]
    retrieval = [
        RetrievalCase(
            query=str(item["query"]),
            expected_filename=str(item["expected_filename"]),
        )
        for item in payload["retrieval"]
    ]
    if not routing or not retrieval:
        raise ValueError("评测集不能为空")
    invalid = {
        case.expected_tool
        for case in routing
        if case.expected_tool is not None and case.expected_tool not in VALID_TOOLS
    }
    if invalid:
        raise ValueError(f"未知工具标签: {sorted(invalid)}")
    return routing, retrieval


def routing_metrics(
    cases: list[RoutingCase],
    predictions: list[str | None],
) -> dict[str, Any]:
    if len(cases) != len(predictions):
        raise ValueError("路由预测数量与评测问题数量不一致")
    correct = sum(
        prediction == case.expected_tool
        for case, prediction in zip(cases, predictions, strict=True)
    )
    total = len(cases)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "multiple_tool_rate": (
            round(predictions.count(MULTIPLE_TOOLS) / total, 4) if total else 0.0
        ),
    }


def retrieval_metrics(ranks: list[int | None], top_k: int) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    total = len(ranks)
    hit_at_1 = sum(rank == 1 for rank in ranks)
    hit_at_k = sum(rank is not None and rank <= top_k for rank in ranks)
    reciprocal_rank_sum = sum(1 / rank for rank in ranks if rank is not None)
    return {
        "total": total,
        "hit_at_1": round(hit_at_1 / total, 4) if total else 0.0,
        f"hit_at_{top_k}": round(hit_at_k / total, 4) if total else 0.0,
        "mrr": round(reciprocal_rank_sum / total, 4) if total else 0.0,
    }


def _prediction_from_message(message: Any) -> str | None:
    calls = list(getattr(message, "tool_calls", None) or [])
    if not calls:
        return None
    if len(calls) > 1:
        return MULTIPLE_TOOLS
    return str(calls[0].get("name", ""))


async def evaluate_routing(
    model: Any,
    cases: list[RoutingCase],
) -> dict[str, Any]:
    bound_model = model.bind_tools(tool_schemas())
    predictions: list[str | None] = []
    details: list[dict[str, Any]] = []
    for case in cases:
        response = await bound_model.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=case.question),
            ]
        )
        prediction = _prediction_from_message(response)
        predictions.append(prediction)
        details.append(
            {
                "question": case.question,
                "expected_tool": case.expected_tool,
                "predicted_tool": prediction,
                "correct": prediction == case.expected_tool,
            }
        )
    return {**routing_metrics(cases, predictions), "details": details}


def _load_corpus(
    store: Chroma,
    corpus_dir: Path,
    settings: Settings,
) -> int:
    chunks = []
    for path in sorted(corpus_dir.glob("*.md")):
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()
        chunks.extend(
            split_pages(
                load_markdown(path),
                document_id=document_id,
                filename=path.name,
                source_type="markdown",
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                page_count=1,
            )
        )
    if not chunks:
        raise ValueError("评测语料目录中没有 Markdown 文件")
    store.add_texts(
        texts=[chunk.content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.id for chunk in chunks],
    )
    return len(chunks)


def evaluate_retrieval(
    embeddings: Any,
    cases: list[RetrievalCase],
    corpus_dir: Path,
    settings: Settings,
    top_k: int,
) -> dict[str, Any]:
    store = Chroma(
        collection_name=f"evaluation_{uuid4().hex}",
        embedding_function=embeddings,
    )
    try:
        chunk_count = _load_corpus(store, corpus_dir, settings)
        ranks: list[int | None] = []
        details: list[dict[str, Any]] = []
        for case in cases:
            matches = store.similarity_search_with_score(case.query, k=top_k)
            filenames = [str(document.metadata.get("filename", "")) for document, _ in matches]
            rank = (
                filenames.index(case.expected_filename) + 1
                if case.expected_filename in filenames
                else None
            )
            ranks.append(rank)
            details.append(
                {
                    "query": case.query,
                    "expected_filename": case.expected_filename,
                    "retrieved_filenames": filenames,
                    "rank": rank,
                }
            )
        return {
            **retrieval_metrics(ranks, top_k),
            "corpus_documents": len(list(corpus_dir.glob("*.md"))),
            "corpus_chunks": chunk_count,
            "details": details,
        }
    finally:
        store.delete_collection()


def _chat_model(settings: Settings) -> ChatOpenAI:
    if not settings.chat_configured:
        raise RuntimeError("未配置聊天模型密钥和 CHAT_MODEL，无法运行路由评测")
    model_options: dict[str, Any] = {}
    if settings.model_reasoning_effort:
        model_options["reasoning_effort"] = settings.model_reasoning_effort
    if settings.disable_response_storage:
        model_options["store"] = False
    return ChatOpenAI(
        api_key=settings.resolved_chat_api_key,
        base_url=settings.resolved_chat_base_url,
        model=settings.chat_model,
        timeout=settings.model_timeout_seconds,
        max_retries=0,
        use_responses_api=settings.use_responses_api,
        **model_options,
    )


def _embeddings(settings: Settings) -> OpenAIEmbeddings:
    if not settings.embedding_configured:
        raise RuntimeError("未配置 Embedding 密钥和 EMBEDDING_MODEL，无法运行检索评测")
    return OpenAIEmbeddings(
        api_key=settings.resolved_embedding_api_key,
        base_url=settings.resolved_embedding_base_url,
        model=settings.embedding_model,
        request_timeout=settings.model_timeout_seconds,
        max_retries=0,
        check_embedding_ctx_length=False,
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    routing_cases, retrieval_cases = load_cases(args.cases)
    result: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "routing_case_count": len(routing_cases),
        "retrieval_case_count": len(retrieval_cases),
    }
    if args.mode in {"routing", "all"}:
        result["chat_model"] = settings.chat_model
        result["routing"] = await evaluate_routing(_chat_model(settings), routing_cases)
    if args.mode in {"retrieval", "all"}:
        result["embedding_model"] = settings.embedding_model
        result["retrieval"] = await asyncio.to_thread(
            evaluate_retrieval,
            _embeddings(settings),
            retrieval_cases,
            args.corpus,
            settings,
            args.top_k,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行真实模型 Tool Calling 与 RAG 检索评测")
    parser.add_argument("--mode", choices=("routing", "retrieval", "all"), default="all")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.json"))
    parser.add_argument("--corpus", type=Path, default=Path("evals/corpus"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = asyncio.run(run(args))
    except RuntimeError as exc:
        raise SystemExit(f"EVALUATION_NOT_CONFIGURED: {exc}") from exc
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
