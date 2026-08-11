import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from app.agent.service import SYSTEM_PROMPT, AgentService
from app.exceptions import AppError
from app.models import DocumentSource, RetrievedChunk
from tests.fakes import FakeBlogClient, FakeChatModel, FakeRAGService


def make_agent(
    mode: str,
    *,
    rag: FakeRAGService | None = None,
    blog: FakeBlogClient | None = None,
) -> tuple[AgentService, FakeChatModel, FakeRAGService, FakeBlogClient]:
    model = FakeChatModel(mode)
    rag_service = rag or FakeRAGService()
    blog_client = blog or FakeBlogClient()
    return (
        AgentService(model, rag_service, blog_client),
        model,
        rag_service,
        blog_client,
    )


async def test_document_question_executes_one_tool_and_two_model_calls() -> None:
    agent, model, rag, blog = make_agent("documents")
    response = await agent.chat("Redis 连接持续超时，应按什么顺序排查？", 4)
    assert response.mode == "tool"
    assert response.tool_call is not None
    assert response.tool_call.name == "search_documents"
    assert response.tool_call.arguments["top_k"] == 4
    assert response.sources[0].source_id == "S1"
    assert response.sources[0].filename == "redis_connection_timeout.md"
    assert rag.calls == 1
    assert blog.calls == 0
    assert model.calls == 2
    assert model.bound_invocations == 1
    assert model.unbound_invocations == 1


async def test_both_native_tools_are_registered_with_openai_schemas() -> None:
    agent, model, _rag, _blog = make_agent("direct")
    await agent.chat("你好", 4)
    assert all(isinstance(item, BaseTool) for item in agent.tools.values())
    functions = {item["function"]["name"]: item["function"] for item in model.bound_tools}
    assert set(functions) == {"search_documents", "search_articles"}
    assert functions["search_documents"]["parameters"]["properties"]["query"]
    assert functions["search_articles"]["parameters"]["properties"]["keyword"]
    assert functions["search_documents"]["parameters"]["additionalProperties"] is False
    assert functions["search_articles"]["parameters"]["additionalProperties"] is False


def test_incident_prompt_requires_evidence_and_uncertainty() -> None:
    assert "症状概括、可能原因和排查步骤" in SYSTEM_PROMPT
    assert "不要把推测写成已确认根因" in SYSTEM_PROMPT
    assert "回答正文只能使用来源编号" in SYSTEM_PROMPT


async def test_request_top_k_overrides_model_tool_argument() -> None:
    agent, _model, rag, _blog = make_agent("documents")
    _result, arguments = await agent._execute_tool(
        "search_documents",
        {"query": "连接池超时", "top_k": 1},
        request_top_k=7,
    )
    assert arguments["top_k"] == 7
    assert rag.last_top_k == 7


async def test_blog_question_uses_blog_sources() -> None:
    agent, model, rag, blog = make_agent("articles")
    response = await agent.chat("我的博客里有 Go 连接池超时排查文章吗？", 4)
    assert response.tool_call is not None
    assert response.tool_call.name == "search_articles"
    assert response.sources[0].source_id == "B1"
    assert response.sources[0].article_id == 5
    assert blog.last_keyword == "连接池超时"
    assert rag.calls == 0
    assert model.calls == 2
    assert model.bound_invocations == 1
    assert model.unbound_invocations == 1


async def test_direct_answer_uses_one_model_call_and_no_tool() -> None:
    agent, model, rag, blog = make_agent("direct")
    response = await agent.chat("你好", 4)
    assert response.mode == "direct"
    assert response.sources == []
    assert response.tool_call is None
    assert model.calls == 1
    assert model.bound_invocations == 1
    assert model.unbound_invocations == 0
    assert rag.calls == blog.calls == 0


async def test_multiple_tool_calls_are_rejected_without_execution() -> None:
    agent, model, rag, blog = make_agent("multiple")
    with pytest.raises(AppError, match="AGENT_TOOL_LIMIT"):
        await agent.chat("同时搜索", 4)
    assert model.calls == 1
    assert rag.calls == blog.calls == 0


async def test_second_model_tool_attempt_is_rejected_without_execution() -> None:
    agent, model, rag, blog = make_agent("second_tool_attempt")
    with pytest.raises(AppError, match="AGENT_TOOL_LIMIT"):
        await agent.chat("继续调用另一个工具", 4)
    assert model.calls == 2
    assert model.bound_invocations == 1
    assert model.unbound_invocations == 1
    assert rag.calls == 1
    assert blog.calls == 0


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("invalid_name", "INVALID_TOOL_NAME"),
        ("invalid_args", "INVALID_TOOL_ARGUMENTS"),
        ("extra_args", "INVALID_TOOL_ARGUMENTS"),
    ],
)
async def test_invalid_tool_request_is_controlled(mode: str, code: str) -> None:
    agent, _model, rag, blog = make_agent(mode)
    with pytest.raises(AppError, match=code):
        await agent.chat("问题", 4)
    assert rag.calls == blog.calls == 0


async def test_empty_tool_result_has_no_fabricated_sources() -> None:
    agent, _model, _rag, _blog = make_agent("empty_results", rag=FakeRAGService([]))
    response = await agent.chat("查资料", 4)
    assert response.sources == []
    assert response.mode == "tool"


async def test_empty_blog_result_has_no_fabricated_sources() -> None:
    agent, _model, _rag, _blog = make_agent("empty_articles", blog=FakeBlogClient([]))
    response = await agent.chat("搜索博客", 4)
    assert response.sources == []
    assert response.mode == "tool"


@pytest.mark.parametrize("mode", ["timeout", "httpx_timeout", "openai_timeout"])
async def test_model_timeout_is_mapped(mode: str) -> None:
    agent, _model, _rag, _blog = make_agent(mode)
    with pytest.raises(AppError, match="MODEL_TIMEOUT"):
        await agent.chat("问题", 4)


async def test_model_failure_is_mapped() -> None:
    agent, _model, _rag, _blog = make_agent("failure")
    with pytest.raises(AppError, match="MODEL_CALL_FAILED"):
        await agent.chat("问题", 4)


async def test_sources_are_built_from_tool_data_not_model_text() -> None:
    rag = FakeRAGService(
        [
            RetrievedChunk(
                source_id="S1",
                content="真实工具内容",
                filename="real.md",
                page=7,
                document_id="real-id",
            )
        ]
    )
    agent, _model, _rag, _blog = make_agent("documents", rag=rag)
    response = await agent.chat("文档问题", 4)
    assert response.sources[0].excerpt == "真实工具内容"
    assert response.sources[0].page == 7
    assert "real.md" not in response.answer


async def test_fabricated_source_reference_is_rejected() -> None:
    agent, model, rag, blog = make_agent("fabricated_citation")
    with pytest.raises(AppError, match="MODEL_CALL_FAILED"):
        await agent.chat("文档问题", 4)
    assert model.calls == 2
    assert rag.calls == 1
    assert blog.calls == 0


@pytest.mark.parametrize(
    "answer",
    [
        "根据 fake.pdf 可以确认根因。[S1]",
        "根据第 99 页可以确认根因。[S1]",
        "根据文章《不存在的排障记录》可以确认根因。[S1]",
    ],
)
def test_source_metadata_in_model_answer_is_rejected(answer: str) -> None:
    source = DocumentSource(
        source_id="S1",
        source_type="document",
        excerpt="真实内容",
        filename="real.md",
        page=1,
        document_id="real-id",
    )
    with pytest.raises(AppError, match="MODEL_CALL_FAILED"):
        AgentService._validated_answer(AIMessage(content=answer), [source])


def test_answer_with_sources_requires_a_citation() -> None:
    source = DocumentSource(
        source_id="S1",
        source_type="document",
        excerpt="真实内容",
        filename="real.md",
        page=1,
        document_id="real-id",
    )
    with pytest.raises(AppError, match="MODEL_CALL_FAILED"):
        AgentService._validated_answer(AIMessage(content="没有引用。"), [source])


def test_model_tool_result_hides_authoritative_source_metadata() -> None:
    result = AgentService._model_tool_result(
        "search_documents",
        [
            {
                "source_id": "S1",
                "content": "真实内容",
                "filename": "real.md",
                "page": 7,
                "document_id": "real-id",
            }
        ],
    )
    assert result == [{"source_id": "S1", "content": "真实内容"}]
