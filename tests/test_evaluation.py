from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

import evals.run_evaluation as evaluation
from app.config import Settings
from evals.run_evaluation import (
    MULTIPLE_TOOLS,
    RoutingCase,
    evaluate_routing,
    load_cases,
    retrieval_metrics,
    routing_metrics,
)


def test_retrieval_evaluation_sends_raw_text_to_compatible_api(
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    class CapturingEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(evaluation, "OpenAIEmbeddings", CapturingEmbeddings)
    evaluation._embeddings(
        Settings(
            embedding_api_key="embedding-key",
            embedding_base_url="https://embedding.example/v1/",
            embedding_model="test-embedding",
        )
    )

    assert captured["api_key"] == "embedding-key"
    assert captured["base_url"] == "https://embedding.example/v1"
    assert captured["check_embedding_ctx_length"] is False


class EvaluationModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.bound_tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]]) -> "EvaluationModel":
        self.bound_tools = tools
        return self

    async def ainvoke(self, _messages: list[Any]) -> AIMessage:
        return self.responses.pop(0)


def test_evaluation_dataset_has_balanced_routing_and_retrieval_cases() -> None:
    routing, retrieval = load_cases(Path("evals/cases.json"))

    assert len(routing) == 30
    assert len(retrieval) == 15
    assert sum(case.expected_tool == "search_documents" for case in routing) == 10
    assert sum(case.expected_tool == "search_articles" for case in routing) == 10
    assert sum(case.expected_tool is None for case in routing) == 10


def test_metric_calculations_are_deterministic() -> None:
    cases = [
        RoutingCase("document", "search_documents"),
        RoutingCase("direct", None),
        RoutingCase("multiple", "search_articles"),
    ]
    routing = routing_metrics(cases, ["search_documents", None, MULTIPLE_TOOLS])
    retrieval = retrieval_metrics([1, 2, None, 3], top_k=3)

    assert routing == {
        "total": 3,
        "correct": 2,
        "accuracy": 0.6667,
        "multiple_tool_rate": 0.3333,
    }
    assert retrieval == {
        "total": 4,
        "hit_at_1": 0.25,
        "hit_at_3": 0.75,
        "mrr": 0.4583,
    }


async def test_routing_evaluation_uses_native_tools_and_counts_multiple_calls() -> None:
    model = EvaluationModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "Redis"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "Redis"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_articles",
                        "args": {"keyword": "Go"},
                        "id": "call-3",
                        "type": "tool_call",
                    },
                ],
            ),
        ]
    )
    cases = [
        RoutingCase("查文档", "search_documents"),
        RoutingCase("查博客", "search_articles"),
    ]

    result = await evaluate_routing(model, cases)

    assert len(model.bound_tools) == 2
    assert result["correct"] == 1
    assert result["accuracy"] == 0.5
    assert result["multiple_tool_rate"] == 0.5
