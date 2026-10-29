# 企业知识问答 Agent

使用 Python 和 LangGraph 构建的企业制度问答 Agent。

| 部分 | 方法/框架 | 选择理由 |
|---|---|---|
| 语言 | Python | 个人喜好; 便于接入模型、检索和测试 |
| Agent 编排 | LangGraph `StateGraph` | 明确展示决策循环和状态变化 |
| 交互入口 | CLI | 开发和评审成本低 |
| 模型 | 一个支持工具调用的模型 | 先验证工具调用，再开发其他功能 |
| 数据 | 本地 JSON，约 6–8 份制度 | 可控、可替换，方便测试 |
| 检索 | 中文字符 n-gram TF-IDF | 无需额外 embedding 服务，容易复现 |
| 输出校验 | Pydantic + 引用检查 | 保证结构和引用来源可检查 |
| 测试 | pytest + 真实模型评估脚本 | 分别验证工程逻辑和模型表现 |

## 环境

- Python 3.14.4
- 以下命令使用 Windows PowerShell

## 安装

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## src下运行

```powershell
.\.venv\Scripts\python.exe -m kb_agent --question "年假需要提前多久申请？"
```
