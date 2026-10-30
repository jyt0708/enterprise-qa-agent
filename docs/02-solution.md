# 调用流程、动态决策、技术取舍

Graph有 `model` 和 `tools` 两个节点。模型调用原生工具选择接口，每轮自行产生一个动作和参数；工具观察作为 `ToolMessage` 追加到完整会话。路由代码判断是否已有合法最终结果。

动作包括 `search_documents(query, top_k)`、`read_document(doc_id, chunk_id?)`、`finish(...)`。每个动作还带一句公开的 `rationale` 简短说明目的。追问通过 `finish(status="needs_clarification", follow_up=...)` 完成。

检索按文档标题、片段标题、原文构建字符 2–4 gram TF-IDF，以每份文档最匹配片段的分数排序。

成功读取的文本进入本轮证据集合。`finish` 的引用若未成功读取、版本不一致或不属于原文片段，会被拒绝，并作为观察交回模型纠正。声明资料不足也要求至少一次成功检索。

每轮检查模型和工具预算。另设 LangGraph recursion_limit 作为图运行保护。每轮需恰好一个动作（由图验证）

记录中的 `next_action` 来自下一步实际执行事件，结束后生成，不是预设日志。每个用例只生成一份 JSON，保存观察原文、步骤、参数、摘要和依据供复核。`run_kind` 明确区分真实模型与工程测试替身。

模型通过 `langchain-openrouter` 的 `ChatOpenRouter` 直接接入 OpenRouter。`openrouter/free` 在服务端选择支持工具调用的免费模型。

`finish` 的工具 schema 将答案、引用、冲突、缺口和追问字段全部声明为必填，减少型漏填字段。OpenRouter 请求默认间隔至少 4 秒，可通过 `LLM_MIN_INTERVAL_SECONDS` 调整。
