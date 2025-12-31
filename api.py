import asyncio
import json
from typing import Optional

from fastapi import FastAPI, Body, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import create_react_langgraph_agent
from local_llm import llm
from sql_tool_demo import sql_agent_tool

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_agent()
    yield

app = FastAPI(title="LangGraph Agent API", lifespan=lifespan)

# 添加 CORS 中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory: MemorySaver | None = None
agent_app = None

deep_thinking: bool = True

# 消息历史裁剪配置（保留最近N条消息，防止上下文过长）
max_history_messages: int = 100

async def load_tools() -> list:
    tools: list = []
    try:
        mcp_client = MultiServerMCPClient(
            connections={
                "local_mcp": {
                    "url": "http://localhost:8080/sse",
                    "transport": "sse",
                }
            }
        )
        tool_1 = await mcp_client.get_tools()
        tools.extend(tool_1)
    except Exception:
        pass
    try:
        mcp_client2 = MultiServerMCPClient(
            connections={
                "local_mcp": {
                    "url": "http://127.0.0.1:8080/",
                    "transport": "sse",
                }
            }
        )
        tool_2 = await mcp_client2.get_tools()
        tools.extend(tool_2)
    except Exception:
        print("mcp连接失败")
        pass

    tools.append(sql_agent_tool)

    return tools

async def ensure_agent() -> None:
    global agent_app, memory
    if agent_app is None:
        memory = MemorySaver()
        tools = await load_tools()
        agent_app, memory = create_react_langgraph_agent(
            llm, tools, deep_thinking=deep_thinking, memory=memory, max_history_messages=max_history_messages
        )

class ModeRequest(BaseModel):
    deep_thinking: bool

class HistoryConfigRequest(BaseModel):
    max_messages: int  # 保留的最大历史消息数

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "user_session_1"

class ChatResponse(BaseModel):
    answer: str
    steps: int

class ResetRequest(BaseModel):
    thread_id: Optional[str] = None  # 如果为 None，则清空所有会话

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "mode": "deep" if deep_thinking else "fast",
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
    memory = MemorySaver()
    
    # 重新加载工具
    tools = await load_tools()
    
    # 重新创建智能体
    agent_app, memory = create_react_langgraph_agent(
        llm, tools, deep_thinking=deep_thinking, memory=memory, max_history_messages=max_history_messages
    )
    
    return {
        "ok": True,
        "message": "智能体已完全重置，所有会话历史已清空",
        "mode": "deep" if deep_thinking else "fast"
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
    return ChatResponse(answer=answer, steps=steps)

#流式消息
@app.post("/chat_stream_tokens")
async def chat_stream_tokens(req: ChatRequest):
    # await ensure_agent()
    
    async def event_generator():
        config = {"configurable": {"thread_id": req.thread_id or "user_session_1"}}
        go_out = True
        try:
            current_content = ""
            
            # 流式调用
            async for result in agent_app.astream_events(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
                version="v2"
            ):
                event_type = result['event']  
                if event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'agent':
                    go_out = True
                elif event_type == "on_chat_model_start" and result['metadata']["langgraph_node"] == 'tools':
                    go_out = False
                # 检测工具调用 - 在 agent 节点检查 tool_calls
                if result.get("event") == "on_chat_model_end":
                    data = result.get("data", {})
                    output = data.get("output", {})
                    
                    # 检测是否有tool_calls
                    if hasattr(output, 'tool_calls') and output.tool_calls:
                        for tool_call in output.tool_calls:
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            
                            # 工具正在别调用中
                            event_data = {
                                "type": "deepthought",
                                "content": f"正在调用工具: {tool_name},{tool_args}"
                            }
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                
                # 流式输出
                if event_type == "on_chat_model_stream" and go_out==True:
                    chunk = result['data']["chunk"]
                    if hasattr(chunk, "content"):
                        if chunk.content:
                            # 将连续的两个换行符替换为一个
                            finallyoutput = chunk.content
                            current_content += finallyoutput
                            event_data = {
                                "type": "token",
                                "content": finallyoutput
                            }
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                elif event_type == "on_chat_model_stream" and go_out==False:
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
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


