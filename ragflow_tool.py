from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent,create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from local_llm import llm
import asyncio
import threading
import os
mcpurl = os.getenv("MCP_URL", "http://127.0.0.1:8080/")
MCP_CONNECTIONS = {
    "local_mcp": {
        "url": mcpurl,
        "transport": "sse",
    }
}
_cached_mcp_tools = None
_cached_sql_agent = None

front_url = os.getenv("front_url_file", 'http://192.168.46.210:8081/')

#react智能体（无历史）- 用于调用ragflow工具
def agent_react_without_history(llm,tools,prompt):
    # 使用支持 JSON 参数的输出解析器
    from langchain_classic.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser
    from langchain_classic.agents.output_parsers import ChineseReActOutputParser
    chinese_parser = ChineseReActOutputParser(debug=False) #调试模式开启
    agent = create_react_agent(
        llm, 
        tools, 
        prompt,
        #output_parser=ReActJsonSingleInputOutputParser(),
        output_parser=chinese_parser
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        #异常格式处理，选择True的话如果格式异常会进行循环修改
        handle_parsing_errors=True
    )
    # 直接返回 agent_executor，不包装历史记录（避免流式调用冲突）
    return agent_executor

#获得工具列表
async def get_tools_async(force_refresh: bool = False):
    global _cached_mcp_tools
    if _cached_mcp_tools is not None and not force_refresh:
        return _cached_mcp_tools
    mcp_client = MultiServerMCPClient(connections=MCP_CONNECTIONS)
    _cached_mcp_tools = await mcp_client.get_tools()
    return _cached_mcp_tools

tools = []
tool_names = []

prompt_ragflow = PromptTemplate.from_template(
    """
# 你是一个数据整理的专家，专注于将输入的内容调用工具进行查询并将结果整理后返回
## 角色工具
你拥有的工具为
{tools}，{tool_names}
注意，你只能使用名为'search_ragflow'的工具，别的工具都不许调用
注意，不要连续输出两个换行符'\n'
必须严格按照以下格式回答：

Question: 需要回答的问题
Thought: 我需要做什么
Action: 工具名称（必须是search_ragflow）
Action Input: {{}}
Observation: 工具返回的结果
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 任务
1. 根据【用户问题】调用工具'search_ragflow'查询结果
2. 每个分块的文件名存放在工具返回内容的"filename"元素当中，你需要将相同文件名的分块进行整理
3. 对于分块的文件名的内容，你需要拼接成一个完整的链接，比如：、xx.pdf,我需要你拼接一个前缀:{front_url},最终结果为：{front_url}/xx.pdf
4. 'search_ragflow'工具的返回每个分块会有一个page元素，其作用是指明返回内容来自源文件的详细信息，其内容是一个数字，示例：
"page": n
你都只需要将n与任务3得到的文件链接进行拼接（示例中为a1),在任务3的链接结尾增加一个'#'+数字,组成一个对应文件对应分块的链接存放到原则2的“指向链接”中
示例最终返回的结果为：{front_url}/xx.pdf#page=a1，这很重要！！
5. 你需要将分块的源文件进行整理，将相同源文件的分块汇合到一起，以"分块1"、"分块2"进行区分，详情见原则2


-原则1：每个指向链接必须是{front_url}+文件名+数字的组合，即{front_url}/xx.pdf#page=a1，必须返回完整拼接的链接！这很重要！只进行简单的拼接即可。
-原则2：每个分块都有对应的页码，需要将页码和分块一一对应，返回内容需要结构分明结构为：
    a文件的分块内容：
        分块1：.....
        页码1：
        指向链接1：
        分块2：
        页码2：
        指向链接2：
        .....
    a文件名：
    参考文件的链接：....
    b文件的分块内容：
        分块1：
        页码1：
        指向链接1：
        分块2：
        页码2：
        指向链接2：
        .....
    b文件名：
-原则3：除了我需要你返回的分块信息，不要有其他的返回，不要返回工具是怎么调用的，你在后台调用即可，不需要返回调用逻辑，只返回给我分块信息即可。可以简单赘述“成功查询到知识库内容，相关内容为："

现在开始：

Question: {input}
Thought:{agent_scratchpad}"""
)

async def create_ragflow_agent():
    global _cached_sql_agent, tools, tool_names
    if _cached_sql_agent is not None:
        return _cached_sql_agent

    tools = await get_tools_async()
    tool_names = [t.name for t in tools]
    tools_text = "\n".join(
        [f"{t.name}: {getattr(t, 'description', '')}" for t in tools]
    )
    prompt = prompt_ragflow.partial(
        tools=tools_text,
        tool_names=", ".join(tool_names),
        front_url=front_url,  # 添加前端 URL 前缀
    )
    _cached_sql_agent = agent_react_without_history(llm, tools, prompt)
    return _cached_sql_agent

@tool(
    "ragflow_search",
    description="这是一个自己调用ragflow中所存储的知识库的工具，用于获得知识库的信息",
)
async def ragflow_search_tool(input: str) -> str:
    agent = await create_ragflow_agent()
    # 移除 session_id 配置，因为 agent 已经不再使用历史记录
    result = await agent.ainvoke({"input": input})
    return result.get("output", str(result))
