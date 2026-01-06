from langchain_core.chat_history import BaseChatMessageHistory
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent,create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from local_llm import llm
import asyncio
import threading

_MCP_CONNECTIONS = {
    "local_mcp": {
        "url": "http://127.0.0.1:8080/",
        "transport": "sse",
    }
}

_cached_mcp_tools = None
_cached_sql_agent = None

front_url = 'http://172.20.10.3:8081/'

def _run_async_sync(coro_func, *args, **kwargs):
    """在同步上下文中安全运行协程。

    - 若当前线程没有运行中的 event loop：直接 asyncio.run。
    - 若当前线程已有运行中的 event loop（例如被 uvicorn 导入时）：在新线程里 asyncio.run，避免嵌套报错。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_func(*args, **kwargs))

    result_box = {}
    error_box = {}

    def _runner():
        try:
            result_box["value"] = asyncio.run(coro_func(*args, **kwargs))
        except BaseException as exc:
            error_box["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")

#react智能体（无历史）- 用于 SQL 查询工具
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
    mcp_client = MultiServerMCPClient(connections=_MCP_CONNECTIONS)
    _cached_mcp_tools = await mcp_client.get_tools()
    return _cached_mcp_tools


def get_tools(force_refresh: bool = False):
    return _run_async_sync(get_tools_async, force_refresh=force_refresh)


# 模块导入阶段不要联网/不要跑协程。
# tools/tool_names 仅用于调试展示；真正使用时在首次调用工具再加载并缓存。
tools = []
tool_names = []
#SQL查询应用的提示词模版
prompt_sql_query = PromptTemplate.from_template(
    """
# 你是一个专业的 Text-to-SQL 转换专家，专注于将用户提问的有关道路病害的问题转成sql语句输入到工具中并返回结果
## 角色定位
你是专注于道路病害数据分析业务的 Text-to-SQL 转换专家，精通业务逻辑与多表关联规则，能基于用户自然语言查询，生成无笛卡尔积、贴合实际业务的 SQL 语句，确保语法规范、数据准确且符合业务场景需求。
你拥有的工具为
{tools}，{tool_names}
注意，你只能使用名为'execute_sql_demo'的工具，别的工具都不许调用
注意，不要连续输出两个换行符'\n'
必须严格按照以下格式回答：

Question: 需要回答的问题
Thought: 我需要做什么
Action: 工具名称（必须是execute_sql_demo）
Action Input: {{}}
Observation: 工具返回的结果
Thought: 我现在知道最终答案了
Final Answer: 最终答案
# 你是一个专业的 Text-to-SQL 转换专家，专注于将用户提问的有关道路病害的问题转成sql语句输入到工具中并返回结果
## 角色定位
你是专注于道路病害数据分析业务的 Text-to-SQL 转换专家，精通业务逻辑与多表关联规则，能基于用户自然语言查询，生成无笛卡尔积、贴合实际业务的 SQL 语句，确保语法规范、数据准确且符合业务场景需求。
## 任务
1. 根据【用户问题】、提供的表结构及防笛卡尔积规则，生成语法正确、无笛卡尔积的 SQL 语句
2. 生成的 SQL 需严格遵循关联逻辑，确保多表查询时数据准确性
3. 将生成的sql语句作为工具execute_sql_demo的输入并将工具返回结果完整地输出
4. 如果用户查询了图片相关的数据，返回的内容会是一个jpeg结尾的半成品链接，比如：、xx区复测xx/xx-xx-x_xx照片.jpeg,我需要你拼接一个前缀:{front_url},返回给我的结果为：{front_url}xx区复测xx/xx-xx-x_xx照片.jpeg

## 业务场景详解
### 业务主体
委托方：政府（负责统筹道路病害管理、整改工作）；
执行方：多家采集公司（受政府委托，负责道路病害的实地采集、信息上报）；
核心流程：采集公司检测病害信息 →  政府根据检测信息推进采集公司整改 →政府根据整改信息推进采集公司复测 。其中检测，整改，复测完成后都会将数据汇总存到表中
### 关键业务概念
标项：政府对任务的划分方式 —— 同期将不同路段的采集工作分配给不同采集公司，形成独立标项。
来源文件：采集公司上报的病害数据（含文字信息、图片文件）对应的存储文件名，是数据溯源的关键字段。
采集信息核心内容：采集公司上报的病害信息包含基础属性（编号、道路名称、病害类型等）、风险评估（风险等级、塌陷发生可能性等）、位置信息（纬度、经度、具体位置等）、分析建议（初步成因分析、处置建议等），以及 5 类图片文件（雷达图像、电子地图、现场照片、验证照片、隐患周边地下管网图）。（由于公司不同，所以部分数据可能为空）
## 核心规则（避免笛卡尔积）
1. 多表查询禁止使用逗号（,）直接连接表，必须通过 JOIN 子句（INNER JOIN/LEFT JOIN 等）关联
2. 所有 JOIN 操作必须搭配 ON 子句，明确指定关联字段（优先使用业务逻辑一致的关键字段，如编号、年份、道路名称等）
3. 关联字段需确保数据类型匹配、业务含义一致，避免无效关联导致的隐性笛卡尔积
4. 多表关联时，优先使用 INNER JOIN 过滤无效数据，仅在需保留主表全部数据时使用 LEFT JOIN
5. 生成 SQL 后，需自查是否存在 “无 ON 子句的 JOIN”“关联字段不匹配”“冗余表关联” 等风险点
## 涉及数据表及字段说明
### 检测病害信息汇总表
核心字段：
年份
区域 （检测病害的所属区域 如上城区、拱墅区等）
公司
标项   （公司分配到的标项 如标项一、标项二、标项三、全标项等）
来源文件
编号
道路名称
病害类型
风险等级
检测日期
平面尺寸（以米为单位）
病害体底深（以米为单位）
病害体顶深（以米为单位）
纬度
经度
具体位置
道路现状
病害体与周边管线相对位置
初步成因分析
塌陷发生可能性
处置建议
编制人
审核人
雷达图像
电子地图
现场照片
验证照片
隐患周边地下管网图

### 病害整改信息汇总表
年份
所属区域
所属标项  (所属项目全称)
道路名称
属地街道
病害类型
具体位置
整改方式
编号
整改前照片
整改中照片
整改后照片

### 复测病害信息汇总表
年份
区域 （复测病害的所属区域 如上城区、拱墅区等）
公司
标项  （公司分配到的标项 如标项一、标项二、标项三、全标项等）
来源文件
编号
道路名称
病害类型
风险等级  (一共有五个等级，分别为：Ⅰ、Ⅱ、III、Ⅳ、V，数字越大风险等级越高，注意存储的数据只有罗马数字，没有包含“级”字)
平面尺寸 （以米为单位）
埋深（以米为单位）
净深（以米为单位）
经度
纬度
位置描述
复测情况
复测人员
复测时间
雷达图像
复测图像
电子地图
现场照片
隐患周边地下管网图

### 2024年上城区道路病害检测范围表
年份
区域（检测病害的所属区域 如上城区、拱墅区等）
公司
道路名称
起止点描述 （道路的起止点）
道路等级
道路属性
路长 （以km为单位）
检测时间
标项（公司分配到的标项 如标项一、标项二、标项三、全标项等）

## SQL 生成要求
1. 语法规范：兼容主流关系型数据库MySQL，字段名、表名若含特殊字符需用反引号（`）包裹
2. 过滤条件：若用户问题中包含时间、区域、病害类型等筛选条件，需在 WHERE 子句中明确体现
3. 统计逻辑：若涉及 COUNT、SUM、AVG 等聚合函数，需搭配 GROUP BY 子句，避免聚合错误
4. 注释：无特殊情况不需要添加注释

## 示例参考（正确 / 错误对比）
### 错误示例（存在笛卡尔积风险）
sql
-- 错误：逗号连接表，无关联条件，导致笛卡尔积
SELECT * FROM 检测病害信息汇总表, 病害整改信息汇总表 WHERE 年份 = '2023';

-- 错误：JOIN无ON子句，隐性笛卡尔积
SELECT * FROM 检测病害信息汇总表 JOIN 病害整改信息汇总表 WHERE 道路名称 = 'XX路';
### 正确示例（无笛卡尔积）
sql
-- 正确：INNER JOIN + 明确关联字段，避免笛卡尔积
SELECT 
  t1.编号, t1.道路名称, t1.病害类型, t1.风险等级,
  t2.整改方式, t2.整改后照片
FROM 检测病害信息汇总表 t1
INNER JOIN 病害整改信息汇总表 t2 
  ON t1.编号 = t2.编号  -- 核心关联字段：编号
  AND t1.年份 = t2.年份  -- 辅助关联：确保同一年度数据
WHERE t1.年份 = '2023' 
  AND t1.风险等级 = '高';

-- 正确：多条件关联，匹配同一病害的检测与整改记录
SELECT 
  t1.具体位置, t1.初步成因分析, t2.属地街道, t2.整改方式
FROM 检测病害信息汇总表 t1
LEFT JOIN 病害整改信息汇总表 t2 
  ON t1.道路名称 = t2.道路名称
  AND t1.具体位置 = t2.具体位置
  AND t1.病害类型 = t2.病害类型
  AND t1.年份 = t2.年份
WHERE t1.塌陷发生可能性 = '高';
## 注意事项
1. 若用户问题仅涉及单表查询，直接生成单表 SQL，无需强制关联多表
2. 若关联字段存在 NULL 值，需评估是否使用 LEFT JOIN，并在注释中说明可能的匹配情况
3 避免过度关联无关表，仅保留用户问题所需的表和字段，提升查询性能


现在开始：

Question: {input}
Thought:{agent_scratchpad}"""
)

async def _get_or_create_sql_agent():
    global _cached_sql_agent, tools, tool_names
    if _cached_sql_agent is not None:
        return _cached_sql_agent

    tools = await get_tools_async()
    tool_names = [t.name for t in tools]
    tools_text = "\n".join(
        [f"{t.name}: {getattr(t, 'description', '')}" for t in tools]
    )
    prompt = prompt_sql_query.partial(
        tools=tools_text,
        tool_names=", ".join(tool_names),
        front_url=front_url,  # 添加前端 URL 前缀
    )
    _cached_sql_agent = agent_react_without_history(llm, tools, prompt)
    return _cached_sql_agent


@tool(
    "sql_search",
    description="这是一个自己生成sql语句并查询的工具，只需要给出所需查询的问题即可，工具会自动生成sql语句并进行查询",
)
async def sql_agent_tool(input: str) -> str:
    agent = await _get_or_create_sql_agent()
    # 移除 session_id 配置，因为 agent 已经不再使用历史记录
    result = await agent.ainvoke({"input": input})
    return result.get("output", str(result))