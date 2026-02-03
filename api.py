from dotenv import load_dotenv
load_dotenv()
import asyncio
import threading
import json
from datetime import datetime, timezone, date
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fastapi import FastAPI, Body, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import create_react_langgraph_agent
from local_llm import llm
from sql_tool_demo import sql_agent_tool
from ragflow_tool import ragflow_search_tool
from report_tool import report_parallel_tool, set_token_emitter, reset_token_emitter

from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import os
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

class AsyncPostgresSaverWrapper(PostgresSaver):
    """
    将同步 PostgresSaver 包装成支持异步方法的版本。
    通过 run_in_executor 在线程池中执行同步方法，避免 Windows 上 psycopg 异步驱动的兼容性问题。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def aget_tuple(self, config):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        loop = asyncio.get_event_loop()
        func = partial(self.list, config, filter=filter, before=before, limit=limit)
        # list 返回生成器，需要转成列表
        result = await loop.run_in_executor(self._executor, lambda: list(func()))
        for item in result:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        loop = asyncio.get_event_loop()
        func = partial(self.put, config, checkpoint, metadata, new_versions)
        return await loop.run_in_executor(self._executor, func)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        loop = asyncio.get_event_loop()
        func = partial(self.put_writes, config, writes, task_id, task_path)
        return await loop.run_in_executor(self._executor, func)


class BufferedPostgresSaverWrapper(AsyncPostgresSaverWrapper):
    """
    仅在显式提交时落盘检查点。
    运行过程中缓存最新检查点和对应 writes，避免每个节点都写数据库。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_lock = threading.Lock()
        self._buffer: dict[str, dict] = {}

    def put(self, config, checkpoint, metadata, new_versions):
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        with self._buffer_lock:
            self._buffer[thread_id] = {
                "config": config,
                "checkpoint": checkpoint,
                "metadata": metadata,
                "new_versions": new_versions,
                "writes": [],
            }
        return next_config

    def put_writes(self, config, writes, task_id, task_path=""):
        thread_id = config["configurable"]["thread_id"]
        with self._buffer_lock:
            entry = self._buffer.get(thread_id)
            if entry is None:
                entry = {
                    "config": config,
                    "checkpoint": None,
                    "metadata": None,
                    "new_versions": None,
                    "writes": [],
                }
                self._buffer[thread_id] = entry
            entry["writes"].append((writes, task_id, task_path))
        return None

    def commit(self, thread_id: str) -> bool:
        with self._buffer_lock:
            entry = self._buffer.pop(thread_id, None)
        if not entry or entry.get("checkpoint") is None:
            return False
        config = entry["config"]
        checkpoint = entry["checkpoint"]
        metadata = entry["metadata"]
        new_versions = entry["new_versions"]

        next_config = super().put(config, checkpoint, metadata, new_versions)
        for writes, task_id, task_path in entry.get("writes", []):
            super().put_writes(next_config, writes, task_id, task_path)
        return True

    async def acommit(self, thread_id: str) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.commit, thread_id)


# 同步 PostgresSaver 的连接池（避免 Windows 异步事件循环兼容性问题）
connection_pool: ConnectionPool | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, connection_pool
    # 创建同步连接池用于 PostgresSaver，设置 autocommit 避免事务块问题
    connection_pool = ConnectionPool(
        conninfo=PG_URL_PSYCOPG,
        kwargs={"autocommit": True}  # 让 setup() 中的 CREATE INDEX CONCURRENTLY 能正常执行
    )
    # 使用包装器支持异步方法
    memory = BufferedPostgresSaverWrapper(connection_pool)
    # memory = BufferedPostgresSaverWrapper(connection_pool)
    memory.setup()  # 同步调用，创建 checkpoint 表
    await ensure_agent()
    yield
    # 关闭连接池
    connection_pool.close()

app = FastAPI(title="LangGraph Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加 CORS 中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 指向数据库的地址，lcjsdtc为密码，127.0.0.1:8976为端口号，postgres为数据库的名称，没有指明 schema，使用默认 public。
# SQLAlchemy 使用 asyncpg 驱动（兼容 Windows ProactorEventLoop）
PG_URL_ASYNCPG = "postgresql+asyncpg://postgres:lcjsdtc@127.0.0.1:8976/postgres"
# AsyncPostgresSaver 需要纯 psycopg 连接字符串（不带 sqlalchemy 前缀）
PG_URL_PSYCOPG = "postgresql://postgres:lcjsdtc@127.0.0.1:8976/postgres"
async_engine: AsyncEngine = create_async_engine(PG_URL_ASYNCPG, pool_pre_ping=True)

memory: PostgresSaver | None = None
agent_app = None


# 消息历史裁剪配置（保留最近N条消息，防止上下文过长）
max_history_messages: int = 100


async def load_tools() -> list:
    tools: list = []
    tools_report:list = []
    try:
        mcp_client = MultiServerMCPClient(
            connections={
                "local_mcp": {
                    "url": "http://127.0.0.1:8080/",
                    "transport": "sse",
                }
            }
        )
        tool_1 = await mcp_client.get_tools()
        tools.extend(tool_1)
        tools_report.extend(tool_1)
    except Exception:
        print("mcp连接失败")
        pass

    tools.append(sql_agent_tool)
    tools.append(ragflow_search_tool)
    tools_report.append(report_parallel_tool)

    return tools,tools_report

#创建额外需要的表
async def init_conversation_tables():
    async with async_engine.begin() as conn:
        #存放对话列表
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))
        #存放对话详情
        await conn.execute(text(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT DEFAULT 'success',
                extra_info JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        ))

#在数据库中写入thread_id和标题
async def upsert_conversation(thread_id: str, title: str):
    async with async_engine.begin() as conn:
        await conn.execute(text(
            """
            INSERT INTO conversations(thread_id, title)
            VALUES (:thread_id, :title)
            ON CONFLICT (thread_id)
            DO UPDATE SET
              updated_at = NOW(),
              title = COALESCE(conversations.title, EXCLUDED.title);
            """
        ), {"thread_id": thread_id, "title": title})

#写入消息列表
async def insert_message(thread_id: str, role: str, content: str, status: str = "success", extra_info: Optional[dict] = None):
    async with async_engine.begin() as conn:
        await conn.execute(text(
            """
            INSERT INTO messages(thread_id, role, content, status, extra_info)
            VALUES (:thread_id, :role, :content, :status, :extra_info)
            """
        ), {
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "status": status,
            "extra_info": json.dumps(extra_info) if extra_info else None,
        })

#判断thread_id在数据库中存不存在
async def conversation_exists(thread_id: str) -> bool:
    async with async_engine.begin() as conn:
        res = await conn.execute(text(
            """
            SELECT 1 FROM conversations WHERE thread_id = :thread_id LIMIT 1
            """
        ), {"thread_id": thread_id})
        return res.first() is not None


async def save_chat_history(thread_id: str, user_content: str, assistant_content: str):
    title = (user_content or "新对话").strip()[:50]
    await upsert_conversation(thread_id, title)
    await insert_message(thread_id, "user", user_content)
    await insert_message(thread_id, "assistant", assistant_content)

async def ensure_agent() -> None:
    global agent_app, memory
    if agent_app is None:
        await init_conversation_tables()
        # memory is already initialized in lifespan with PostgresSaver
        tools,tools_report = await load_tools()
        agent_app, memory = create_react_langgraph_agent(
            llm, tools,tools_report, memory=memory, max_history_messages=max_history_messages
        )

class ModeRequest(BaseModel):
    deep_thinking: bool

class HistoryConfigRequest(BaseModel):
    max_messages: int  # 保留的最大历史消息数

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "user_session_1"
    #控制report_tool.py的print是否返回给前端
    stream_report_tokens: bool = True

class ChatResponse(BaseModel):
    answer: str
    steps: int

class ResetRequest(BaseModel):
    thread_id: Optional[str] = None  # 如果为 None，则清空所有会话

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "mode": "deep" ,
        "max_history_messages": max_history_messages
    }
#保留功能，设定裁断次数
# @app.post("/history_config")
# async def set_history_config(req: HistoryConfigRequest):
#     global max_history_messages, agent_app
    
#     if req.max_messages < 5:
#         return {"ok": False, "error": "max_messages 不能小于 5"}
    
#     if req.max_messages > 100:
#         return {"ok": False, "error": "max_messages 不能大于 100"}
    
#     max_history_messages = req.max_messages
    
#     await ensure_agent()
#     agent_app, _ = create_react_langgraph_agent(
#         llm, await load_tools(), deep_thinking=deep_thinking, 
#         memory=memory, max_history_messages=max_history_messages
#     )
    
#     return {
#         "ok": True, 
#         "max_history_messages": max_history_messages,
#         "message": f"已更新历史消息保留数量为 {max_history_messages} 条"
#     }

#保留功能，切换深度学习
# @app.post("/mode")
# async def set_mode(req: ModeRequest):
#     global deep_thinking, agent_app
#     deep_thinking = req.deep_thinking
#     await ensure_agent()
#     agent_app, _ = create_react_langgraph_agent(
#         llm, await load_tools(), deep_thinking=deep_thinking, memory=memory, max_history_messages=max_history_messages
#     )
#     return {"ok": True, "mode": "deep" if deep_thinking else "fast"}

#保留功能，回溯检查点
# @app.get("/checkpoints")
# async def list_checkpoints():
#     await ensure_agent()
#     cps = []
#     try:
#         cps = list(memory.storage.keys()) if memory else []
#     except Exception:
#         cps = []
#     return {"count": len(cps), "ids": cps}

#重置智能体
@app.post("/reset")
async def reset_agent():
    """
    完全重置智能体 - 清空所有会话历史，重新创建智能体和内存
    使用场景：想要完全清空所有对话历史，重新开始
    """
    global agent_app, memory

    # 重新创建 memory（这会清空所有检查点）
    if connection_pool is None:
        # 若生命周期外调用，重新创建连接池
        pool = ConnectionPool(conninfo=PG_URL_PSYCOPG, kwargs={"autocommit": True})
    else:
        pool = connection_pool

    memory = BufferedPostgresSaverWrapper(pool)
    memory.setup()
    
    # 重新加载工具
    tools,tools_report = await load_tools()
    
    # 重新创建智能体
    agent_app, memory = create_react_langgraph_agent(
        llm, tools,tools_report, memory=memory, max_history_messages=max_history_messages
    )
    
    return {
        "ok": True,
        "message": "智能体已完全重置，所有会话历史已清空",
    }
@app.get("/time")
async def list_time():
    time = ["2023","2024","2025"]
    return {
        "code": 200,
        "type": "time",
        "data": {"time": time}
    }

@app.get("/place")
async def list_place():
    place = ["上城区"]
    return {
        "code": 200,
        "type": "place",
        "data": {"place": place}
    }
#单条消息
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    await ensure_agent()
    config = {"configurable": {"thread_id": req.thread_id or "user_session_1"}}
    result = await agent_app.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    # Extract final AIMessage if available
    answer = ""
    steps = 0
    if isinstance(result, dict) and "messages" in result:
        messages = result["messages"]
        steps = len(messages)
        for i, msg in enumerate(messages, 1):
            if type(msg).__name__ == "AIMessage" and i == len(messages):
                answer = msg.content or ""
    if not answer:
        # Fallback: serialize result
        answer = str(result)
    if memory and hasattr(memory, "acommit"):
        await memory.acommit(config["configurable"]["thread_id"])
    await save_chat_history(config["configurable"]["thread_id"], req.message, answer)
    return ChatResponse(answer=answer, steps=steps)


@app.get("/api/conversations")
async def list_conversations():
    await ensure_agent()
    async with async_engine.begin() as conn:
        rows = (await conn.execute(text(
            """
            SELECT thread_id, COALESCE(title, '新对话') AS label,
                   created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT 200
            """
        ))).fetchall()

    today = date.today()
    conversations = []
    for r in rows:
        created = r.created_at
        group = None
        if created.date() == today:
            group = "today"
        elif (today - created.date()).days == 1:
            group = "yesterday"
        conversations.append({
            "key": r.thread_id,
            "label": r.label,
            "group": group,
            "createdAt": r.created_at.isoformat(),
            "updatedAt": r.updated_at.isoformat()
        })

    return {
        "code": 200,
        "message": "success",
        "data": {"conversations": conversations}
    }


@app.get("/api/conversations/{conversation_key}/messages")
async def list_messages(conversation_key: str):
    await ensure_agent()
    async with async_engine.begin() as conn:
        rows = (await conn.execute(text(
            """
            SELECT id, role, content, status, created_at
            FROM messages
            WHERE thread_id = :thread_id
            ORDER BY created_at ASC
            LIMIT 500
            """
        ), {"thread_id": conversation_key})).fetchall()

    messages = []
    for r in rows:
        messages.append({
            "id": str(r.id),
            "message": {
                "role": "user" if r.role == "user" else "assistant",
                "content": r.content
            },
            "status": r.status or "success",
            "timestamp": r.created_at.isoformat()
        })

    return {
        "code": 200,
        "message": "success",
        "data": {
            "conversationKey": conversation_key,
            "messages": messages
        }
    }

#流式消息
@app.post("/chat_stream_tokens")
async def chat_stream_tokens(req: ChatRequest):
    await ensure_agent()
    
    async def event_generator():
        config = {"configurable": {"thread_id": req.thread_id or "user_session_1"}}
        thread_id = config["configurable"]["thread_id"]
        go_out = 'not'
        #在 report_tool 与 SSE 输出之间缓冲消息，避免阻塞主流式事件。
        token_queue = asyncio.Queue()
        #report_tool 的回调函数，将进度消息写入队列。
        async def push_token(message: str):
            await token_queue.put(message)
        precontent = ''

        token_ctx = None
        try:
            #判断是否启动回调
            if req.stream_report_tokens:
                token_ctx = set_token_emitter(push_token, enabled=True)

            current_content = ""
            if not await conversation_exists(thread_id):
                event_data = {
                    "type": "title",
                    "content": req.message
                }
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 流式调用
            async for result in agent_app.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config={"configurable": {"thread_id": thread_id}},
                version="v2"    
            ):
                # 把队列里的进度消息转换成 SSE type=token 推给前端。
                while not token_queue.empty():
                    token_msg = token_queue.get_nowait()
                    event_data = {
                        "type": "token",
                        "content": token_msg
                    }
                    current_content +=token_msg
                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                event_type = result['event']  
                if event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'agent':
                    go_out = 'true'
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'tools':
                    go_out = 'tools'
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'report_final':
                    go_out = 'true'
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] not in ['agent','tools','report_final']:
                    go_out = 'not'
                if result.get("event") == "on_chat_model_end":
                    data = result.get("data", {})
                    output = data.get("output", {})
                    
                    # 检测是否有tool_calls
                    if hasattr(output, 'tool_calls') and output.tool_calls:
                        for tool_call in output.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            # 工具正在调用中
                            if tool_name == 'sql_search':
                                event_data = {
                                    "type": "deepthought",
                                    "content": f"正在查询数据库\n"
                                }
                            elif tool_name == 'ragflow_search':
                                event_data = {
                                    "type": "deepthought",
                                    "content": f"正在查询知识库\n"
                                }
                            elif tool_name == 'get_current_time':
                                event_data = {
                                    "type": "deepthought",
                                    "content": f"正在获取当前时间\n"
                                }
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                if event_type == "on_chat_model_stream" and go_out=='tools':
                    nowcontent = result['data']["chunk"].content
                    if 'Final Answer:' in nowcontent:
                        go_out = 'deep'
                        event_data = {
                                "type": "deepthought",
                                "content":'查询完毕，'
                            }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        continue
                    elif 'Final Answer:' not in precontent and 'Final Answer:' in (precontent + nowcontent):
                        go_out = 'deep'
                        event_data = {
                                "type": "deepthought",
                                "content":'查询完毕，'
                            }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        continue
                    precontent = nowcontent

                # 流式输出
                if event_type == "on_chat_model_stream" and go_out=='true':
                    chunk = result['data']["chunk"]
                    if hasattr(chunk, "content"):
                        if chunk.content:
                            reasoning_debug = None
                            if hasattr(chunk, "additional_kwargs"):
                                reasoning_debug = chunk.additional_kwargs.get("reasoning_content")
                            if reasoning_debug:
                                print(f"[debug reasoning_content] {reasoning_debug}")
                            # 将连续的两个换行符替换为一个
                            finallyoutput = chunk.content
                            current_content += finallyoutput
                            event_data = {
                                "type": "token",
                                "content": finallyoutput
                            }
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                elif event_type == "on_chat_model_stream" and go_out=='deep':
                    chunk_deep = result['data']["chunk"]
                    if hasattr(chunk_deep, "content") and chunk_deep.content:
                        # 将连续的两个换行符替换为一个
                        processed_content_deep = chunk_deep.content.replace("\n\n", "\n")
                        processed_content_deep = processed_content_deep.replace("\n-","\n·")
                        # 移除常见 Markdown 标题符号，避免前端展示异常
                        markdown_chars = "#*`>"
                        cleaned_content = processed_content_deep.translate(str.maketrans('', '', markdown_chars))
                        finallyoutput = cleaned_content.replace('\nAction','Action')
                        event_data = {
                            "type": "deepthought",
                            "content":finallyoutput
                        }
                        yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # 发送完成信号
            final_data = {
                "type": "done",
                #"answer": current_content,
            }
            print(current_content)
            #显式提交结束之后更新memory
            if memory and hasattr(memory, "acommit"):
                await memory.acommit(config["configurable"]["thread_id"])
            try:
                await save_chat_history(config["configurable"]["thread_id"], req.message, current_content)
            except Exception as save_err:
                print(f"保存会话历史失败: {save_err}")
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # 在后端日志中打印错误信息
            print("❌ 流式对话错误:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"详细堆栈:\n{error_traceback}")
            
            error_data = {
                "type": "error",
                "message": str(e),
                "traceback": error_traceback
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        #结束后清空回调返回状态，防止影响下一次请求
        finally:
            if token_ctx:
                reset_token_emitter(*token_ctx)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

