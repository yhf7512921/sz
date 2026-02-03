# LangGraph Agent API

面向“道路病害”场景的 LangGraph + FastAPI 服务端，提供对话与流式对话接口，并支持：
- Postgres 记忆与会话历史落库
- MCP 工具接入（可选）
- SQL / RAG / 报告生成等工具协作

本 README 基于 `api.py` 与 `agent.py` 的实现说明接口和运行方式。

## 功能概览
- 单轮对话：`POST /chat`
- 流式对话（SSE）：`POST /chat_stream_tokens`
- 会话列表：`GET /api/conversations`
- 会话消息：`GET /api/conversations/{conversation_key}/messages`
- 健康检查：`GET /health`
- 重置智能体与记忆：`POST /reset`

## 核心文件
- `api.py`：FastAPI 入口，数据库会话存储、SSE 流式输出、工具加载
- `agent.py`：LangGraph 图与系统提示词、工具路由
- `local_llm.py`：本地 LLM 绑定入口
- `ragflow_tool.py` / `sql_tool_demo.py` / `report_tool.py`：工具实现

## 运行依赖
- Python（建议 3.10+）
- Postgres（必需）
- MCP Server（可选，默认 `http://127.0.0.1:8080/`）

安装依赖：
```bash
pip install -r requirements.txt
```

## 环境变量
`.env` 由 `python-dotenv` 自动加载（见 `api.py` 中的 `load_dotenv()`）。

数据库连接当前在 `api.py` 中硬编码：
- `PG_URL_ASYNCPG = postgresql+asyncpg://postgres:lcjsdtc@127.0.0.1:8976/postgres`
- `PG_URL_PSYCOPG = postgresql://postgres:lcjsdtc@127.0.0.1:8976/postgres`

建议改造：
1. 在 `.env` 中设置 `PG_URL_ASYNCPG` 与 `PG_URL_PSYCOPG`
2. 在 `api.py` 中读取环境变量替换硬编码

## 启动服务
若 `run.py` 已配置，可直接使用：
```bash
python run.py
```

或直接启动 FastAPI：
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## 接口说明

**1) 健康检查**
- `GET /health`
```json
{
  "status": "ok",
  "mode": "deep",
  "max_history_messages": 100
}
```

**2) 单轮对话**
- `POST /chat`
```json
{
  "message": "你好",
  "thread_id": "user_session_1",
  "stream_report_tokens": true
}
```
响应：
```json
{
  "answer": "...",
  "steps": 5
}
```

**3) 流式对话（SSE）**
- `POST /chat_stream_tokens`
请求体同 `/chat`。
SSE 事件类型：
- `type: "title"`
- `type: "token"`
- `type: "deepthought"`
- `type: "done"`
- `type: "error"`

**4) 会话列表**
- `GET /api/conversations`
最多返回 200 条，按 `updated_at` 排序。

**5) 会话消息**
- `GET /api/conversations/{conversation_key}/messages`
最多返回 500 条。

**6) 重置智能体**
- `POST /reset`
清空 checkpoint 并重建 agent。

## Agent 流程（来自 `agent.py`）
- LangGraph 节点：
  - `intent` -> `agent` / `agent_report`
  - `agent` -> `tools` -> `agent`
  - `agent_report` -> `report` -> `report_final` -> END
- 内置系统提示词针对“道路病害”业务场景
- 工具调用基于 `tool_calls` 判断是否继续
- 报告场景使用 `generate_report_parallel`，最终要求输出 `<custom-filecard>`

## 数据库表
服务启动时自动创建：
- `conversations(thread_id, title, created_at, updated_at)`
- `messages(id, thread_id, role, content, status, extra_info, created_at)`

## 备注
- MCP 连接失败不会中断服务，会打印日志并忽略 MCP 工具。
- 调整 `api.py` 中的 `max_history_messages` 可修改历史消息保留长度。

## 文件修改详情

- #修改了venv\Lib\site-packages\langchain_classic\agents\output_parsers下的解释器文件：
- /react_single_input.py
- /__init__.py
-
- #修改了venv\Lib\site-packages\langchain_community\chat_models目录下的模型文件：
- /tongyi.py

