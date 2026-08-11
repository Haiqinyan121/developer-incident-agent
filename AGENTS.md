# AGENTS.md

- 使用 Python 3.11，目标是实现易读、可测试的研发故障诊断 Agent。
- 仅有 `search_documents`（上传文档向量检索）和 `search_articles`（Go 博客标题搜索）两个工具。
- 每个请求最多执行一个工具、最多调用模型两次，不得加入循环规划。
- 不增加 LangGraph、多 Agent、长期记忆、BM25、Reranker、消息队列等重型功能。
- 禁止硬编码 API Key；新增或修改功能必须补充离线测试。
- 启动：`uvicorn app.main:app --reload`
- 测试：`pytest -q`
- 静态检查：`ruff check .`
- 镜像构建：`docker build -t developer-incident-agent .`
- README 必须始终与真实代码和限制一致。
