from local_llm import llm
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import create_langgraph_agent
import asyncio
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 配置 MCP 客户端
mcp_client = MultiServerMCPClient(
    connections={
        "local_mcp": {
            "url": "http://localhost:8080/sse",
            "transport": "sse",
        }
    }
)

# 获取工具列表
tools = asyncio.run(mcp_client.get_tools())
print(f"SQL Agent 可用工具数: {len(tools)}")

sql_system_prompt = """
"""


# 创建 SQL 查询智能体
sql_agent = create_langgraph_agent(llm, tools, system_prompt=sql_system_prompt, need_checkpoints=False)


def make_langgraph_tool(agent_app, name: str = 'langgraph_agent', description: str = '调用 LangGraph 智能体'):
    @tool(name, description=description)
    def agent_tool(query: str) -> str:
        # 创建配置
        config = {"configurable": {"thread_id": "tool_session"}}
        
        # 调用智能体
        result = agent_app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        # 提取最后一条消息作为结果
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            return str(last_message)
        
        return "智能体执行失败"
    
    return agent_tool


# 创建 SQL 查询工具
sql_agent_tool = make_langgraph_tool(
    sql_agent,
    name='sql_search',
    description='这是一个自动生成 SQL 语句并查询数据库的工具,只需要提供查询需求即可,会自动生成 SQL 并返回结果'
)
