"""
并行报告生成工具 - 将报告拆分成8个任务并行执行
"""
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from local_llm import llm
from sql_tool_demo import sql_agent_tool  # 导入 SQL Agent 工具
import asyncio
import threading
from contextvars import ContextVar
from typing import Awaitable, Callable, Optional
import os
mcpurl = os.getenv("MCP_URL", "http://127.0.0.1:8080/")
MCP_CONNECTIONS = {
    "local_mcp": {
        "url": mcpurl,
        "transport": "sse",
    }
}

_cached_mcp_tools = None
#定义一个函数能够向api端发送消息
_token_emitter: ContextVar[Optional[Callable[[str], Awaitable[None]]]] = ContextVar(
    "report_token_emitter",
    default=None,
)
#这里是一个判断
_emit_enabled: ContextVar[bool] = ContextVar("report_emit_enabled", default=True)
# 作用：把“进度回调函数”写入上下文变量，并记录是否启用推送。
# 返回值：旧的上下文 token（token_emitter、token_enabled），用于之后恢复。
# 用途：在一次请求开始时注册“推送通道”，并决定本次是否向前端发送。
def set_token_emitter(emitter: Callable[[str], Awaitable[None]], enabled: bool = True):
    token_emitter = _token_emitter.set(emitter)
    token_enabled = _emit_enabled.set(enabled)
    return token_emitter, token_enabled
# 作用：把上下文变量恢复到调用前的状态。
# 用途：在请求结束时清理，防止并发请求之间“串台”或遗留回调。
def reset_token_emitter(token_emitter, token_enabled):
    _token_emitter.reset(token_emitter)
    _emit_enabled.reset(token_enabled)
#总调用，如果_emit_enabled为true则返回到api端，反之则直接print
async def emit_progress(message: str) -> None:
    if not _emit_enabled.get():
        print(message)
        return
    emitter = _token_emitter.get()
    if emitter:
        await emitter(message)
    else:
        print(message)

def _run_async_sync(coro_func, *args, **kwargs):
    """在同步上下文中安全运行协程"""
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

def create_chapter_agent(llm, tools, prompt):
    """创建章节生成 Agent"""
    from langchain_classic.agents.output_parsers import ChineseReActOutputParser
    chinese_parser = ChineseReActOutputParser(debug=False)
    agent = create_react_agent(llm, tools, prompt, output_parser=chinese_parser)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=15,
        handle_parsing_errors=True
    )
    return agent_executor

async def get_tools_async(force_refresh: bool = False):
    """获取 MCP 工具列表"""
    global _cached_mcp_tools
    if _cached_mcp_tools is not None and not force_refresh:
        return _cached_mcp_tools
    
    tools = []
    
    # 加载 MCP 工具（只加载需要的）
    try:
        mcp_client = MultiServerMCPClient(connections=MCP_CONNECTIONS)
        mcp_tools = await mcp_client.get_tools()
        
        # 只保留 get_current_time 和 search_ragflow
        for tool in mcp_tools:
            if tool.name in ['get_current_time', 'search_ragflow']:
                tools.append(tool)
    except Exception as e:
        print(f"⚠️ MCP 工具加载失败: {e}")
    
    # 添加 SQL Agent 工具（封装好的）
    tools.append(sql_agent_tool)
    
    _cached_mcp_tools = tools
    return _cached_mcp_tools

# 第1章提示词
prompt_chapter_1 = PromptTemplate.from_template(
    """
你负责生成道路病害报告的第一章，你必须使用工具获取数据，然后生成报告。

## 可用工具
{{tools}}

工具名称：{{tool_names}}

## 重要：你必须按照以下格式回答

Question: 用户的问题
Thought: 我需要做什么（用中文思考）
Action: 工具名称（从上面的工具列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 工作步骤
1. 视情况调用 get_current_time 获取时间，调用sql_search获取这一年探测范围的道路数量，调用时需说明在检测范围表中找并需要去重。
2. 调用 sql_search 查询标项信息、负责公司、区域范围等
3. 调用 search_ragflow 获取背景信息，调用sql_search获取这一年检测了多少病害。
4. 调用 sql_search 查询这一年每个标项检测病害的日期都是什么时候被检测出来的，检测时间段为每个月的最早天数——最晚天数（如7.5——7.15，12.1——12.20）若某个标项有只有月份没有具体日期的数据，那你在列这个标项这个月份的时候只列出月份即可如（xx年xx月），若某个公司某月份只有1天，在列这个月份的时候只列出这一天即可。)
5. 调用sql_search查询各个公司检测了哪些道路，调用sql_search查询病害数量top4的道路。
6. 收集完所有数据后，在 Final Answer 中输出完整报告（包含模板但是不包含模板中的问题和要求）
## 报告生成规则
1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
2. 根据各模块问题按要求逐项回答，不要遗漏
3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
4. 使用 Markdown 表格展示数据，表格格式必须正确：
   - 注意：表头和分隔行必须分开，不能混在一起！
5. 大标题和小标题后必须空一行（Markdown格式）
6. 语气正式客观，分析以数据为依据，逻辑清晰
7. 输出的报告中只要有问题的答案即可，不要把问题也输出
8. 检测时间段需全部包含在报告中，不能只包含部分数据（检测时间的格式不同，需要你自行处理）
9. 输出内容时不要把报告模板里的问题也输出出去，只输出在模板里的回答即可。
10. 输出表格时不要把各个标项的数据混在一起。

## 报告模板（必须逐条回答并合成，但是不要把问题也输出出去）

# 一、年度工作概述    
本年度道路病害整体检测、复测和整改工作的开展情况是怎样的？请简要说明本年度一共检测了多少条道路，检测出多少病害，主要检测哪些类型，其中哪些道路较为严重（列出不超过4个，但是不要把这句不超过4个的提示也输出），再说明标项、负责的公司（同一个标项可能有不同公司）、检测的区域（如上城区）和检测的时间段。（用一张表格记录标项、公司、区域、检测时间段的数据）(检测时间需要通过数据库查全，你需要通过数据库得到检测病害的日期是什么时候被检测出来的，然后得到每个月最晚天数与最早天数，最后以xx年xx月xx日————xx年xx月xx日的格式在表格中呈现，若跨月，则需以逗号分开，如2024年7月2日————2024年7月23日，2024年10月1日———2024年10月7日)

现在开始输出报告模板（不要把我的问题和要求也输出出去）：
Question: {{input}}
Thought:{{agent_scratchpad}}""",
    template_format="jinja2"
)

# 第2章提示词
prompt_chapter_2 = PromptTemplate.from_template(
    """
你负责生成道路病害报告的第二章，你必须使用工具获取数据，然后生成报告。

## 可用工具
{{tools}}

工具名称：{{tool_names}}

## 重要：你必须按照以下格式回答

Question: 用户的问题
Thought: 我需要做什么（用中文思考）
Action: 工具名称（从上面的工具列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 工作步骤
1. 调用 sql_search 查询这一年每个标项检测病害的日期都是什么时候被检测出来的
2. 调用 sql_search 查询各标段工作量数据
3. 通过工具搜索需要的数据，章节2.1计算工作天数时不用考虑日期只有月份的情况，只需要考虑日期有具体日期的情况，但是要说明这种只有月份的数据情况，章节2.2需要考虑这种只有月份的情况。
4. 收集完所有数据后，在 Final Answer 中输出完整报告

## 报告生成规则
1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
2. 根据各模块问题按要求逐项回答，不要遗漏
3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
4. 使用 Markdown 表格展示数据，表格格式必须正确：
   - 注意：表头和分隔行必须分开，不能混在一起！
5. 大标题和小标题后必须空一行
6. 语气正式客观，分析以数据为依据，逻辑清晰
7. 输出的报告中只要有问题的答案即可，不要把问题也输出
8. 工作天数需要查全（检测时间的格式不同，需要你自行处理），你需要查询各公司检测病害被检测的具体时间，然后计算出工作天数。
9. 图和表二选一使用，不要既输出表又输出图表（在图和表中选择更为合适的数据载体）

## 图片回答格式
-原则1：注意必须包含完整的标签，严格按照示例的格式，只能修改示例内部的内容，格式不能进行修改！
-原则2：注意不要将示例内容进行输出！只输出经过你修改过后的整个标签内容
-原则3：注意以图表的格式输出的时候必须添加文本分析，不要只输出一个图表，必须图文并茂的进行输出！

### 统计数据可视化
当你需要用图片展示统计类数据时，使用图表组件展示：如年度趋势对比使用折线图，病害类型分布占比使用饼图或柱状图，排名对比使用柱状图等
-数据解释（仅用于模型理解）不需要输出：
'[]'中为图表数据，每个数据项为一个对象，对象中包含x轴数据和y轴数据。x轴数据键为name，y轴数据键为value。
axisXTitle为x轴标题，axisYTitle为y轴标题。
type为图表类型，可选值为line,pie,bar。

-示例格式：
仅仅将以下markdown返回:
#### 年度病害数量趋势图
<custom-chart 
  axisXTitle="年份" 
  axisYTitle="病害数量(个)" 
  type="line">
  [{"name":2013,"value":59},{"name":2014,"value":64},{"name":2015,"value":60}]
</custom-chart>

## 报告模板（必须逐条回答并合成）

# 二、工作量情况分析

## 2.1 工作量概述
（需要用表格）本年度道路病害检测的工作量如何？请列出检测的总里程数，覆盖道路数，所耗工作天数(需要得到具体天数)和日均检测里程数。(这里与下面的工作天数你需要通过数据库得到每个标项检测病害的日期都是什么时候被检测出来的，然后计算多个月内每个月最晚天数与最早天数的差值，最后将差值相加即可得到工作天数（如7月有7.2，7.6，7.8，工作天数为7.2——7.8，即7天）,若某个标项有只有月份没有具体日期的数据，那你在列这个标项的时候只列出月份即可)

## 2.2 各标段工作量
（需要用表格）本年度各标段检测的工作量如何？请说明本年度各标段检测的里程数、覆盖道路数、工作天数(需要得到具体天数，若某个标项有只有月份没有具体日期的数据，那你需要在工作天数中说明这部分数据只提到了月份没有提到具体日期，并且日均检测里程也不需要计算，用"/"表示)和日均检测里程等，并以图表形式对比分析各标段在工作量和工作效率上的情况。

## 2.3 检测覆盖情况
用图表展现并说明本年度道路病害检测的道路种类在整体及各标段的数量和占比各是多少？（这里的道路种类指主干路、次干路、支路等）

现在开始：
Question: {{input}}
Thought:{{agent_scratchpad}}""",
    template_format="jinja2"
)

# 第3.1-3.3章提示词
prompt_chapter_3_1_3 = PromptTemplate.from_template(
    """
你负责生成道路病害报告的第三章第1-3节，你必须使用工具获取数据，然后生成报告。

## 可用工具
{{tools}}

工具名称：{{tool_names}}

## 重要：你必须按照以下格式回答

Question: 用户的问题
Thought: 我需要做什么（用中文思考）
Action: 工具名称（从上面的工具列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 工作步骤
1. 调用 sql_search 查询病害总数和各标段病害数量
2. 调用 sql_search 查询病害数量前十的道路
3. 调用 sql_search 查询各种类型病害的数量
4. 收集完所有数据后，在 Final Answer 中输出完整报告

## 报告生成规则
1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
2. 根据各模块问题按要求逐项回答，不要遗漏
3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
4. 使用 Markdown 表格展示数据
   - 注意：表头和分隔行必须分开，不能混在一起！
5. 大标题和小标题后必须空一行
6. 语气正式客观，分析以数据为依据，逻辑清晰
7. 输出的报告中只要有问题的答案即可，不要把问题也输出
8. 图和表二选一使用，不要既输出表又输出图表（在图和表中选择更为合适的数据载体）

## 图片回答格式
-原则1：注意必须包含完整的标签，严格按照示例的格式，只能修改示例内部的内容，格式不能进行修改！
-原则2：注意不要将示例内容进行输出！只输出经过你修改过后的整个标签内容
-原则3：注意以图表的格式输出的时候必须添加文本分析，不要只输出一个图表，必须图文并茂的进行输出！

### 统计数据可视化
当你需要用图片展示统计类数据时，使用图表组件展示：如年度趋势对比使用折线图，病害类型分布占比使用饼图或柱状图，排名对比使用柱状图等
-数据解释（仅用于模型理解）不需要输出：
'[]'中为图表数据，每个数据项为一个对象，对象中包含x轴数据和y轴数据。x轴数据键为name，y轴数据键为value。
axisXTitle为x轴标题，axisYTitle为y轴标题。
type为图表类型，可选值为line,pie,bar。

-示例格式：
仅仅将以下markdown返回:
#### 年度病害数量趋势图
<custom-chart 
  axisXTitle="年份" 
  axisYTitle="病害数量(个)" 
  type="line">
  [{"name":2013,"value":59},{"name":2014,"value":64},{"name":2015,"value":60}]
</custom-chart>

## 报告模板（必须逐条回答并合成）

## 3.1 病害量概述
本年度道路病害总数及各标段病害数量分别是多少？请以图表形式展示各标段病害情况。

## 3.2 高发道路分析
请以图表形式列出本年度病害数量前十的道路及其病害数量，并详细分析这些高发道路的病害分布特征，如是否集中于特定路段、与周边设施关联性等。

## 3.3 病害类型分布
用图表展示和分析一下各种类型病害的具体数量和占比情况。（是对这种病害的分布情况，数量占比进行分析，而不是分析这种病害）

现在开始：
Question: {{input}}
Thought:{{agent_scratchpad}}""",
    template_format="jinja2"
)

# 第3.4-3.5章提示词
prompt_chapter_3_4_5 = PromptTemplate.from_template(
    """
你负责生成道路病害报告的第三章第4-5节，你必须使用工具获取数据，然后生成报告。

## 可用工具
{{tools}}

工具名称：{{tool_names}}

## 重要：你必须按照以下格式回答

Question: 用户的问题
Thought: 我需要做什么（用中文思考）
Action: 工具名称（从上面的工具列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 工作步骤
1. 调用 sql_search 查询各类病害对应的风险等级
2. 调用 sql_search 查询不同风险等级的病害数量
3. 调用 sql_search 查询高风险病害数量最多的道路
4. 调用 search_ragflow 获取病害成因相关信息
5. 收集完所有数据后，在 Final Answer 中输出完整报告

## 报告生成规则
1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
2. 根据各模块问题按要求逐项回答，不要遗漏
3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
4. 使用 Markdown 表格展示数据
   - 注意：表头和分隔行必须分开，不能混在一起！
6. 语气正式客观，分析以数据为依据，逻辑清晰
7. 输出的报告中只要有问题的答案即可，不要把问题也输出
8. 图和表二选一使用，不要既输出表又输出图表（在图和表中选择更为合适的数据载体）
9. 对于同一块内容（数据），在图和表中二选一使用（在图和表中选择更为合适的数据载体，而不是输出同时包含这部分数据的图和表）
10. 列表的格式为：（注意输出列表前需空一行）
xxxx：

    - 项目1
    - 项目2
    - 项目3
    ...
## 图片回答格式
-原则1：注意必须包含完整的标签，严格按照示例的格式，只能修改示例内部的内容，格式不能进行修改！
-原则2：注意不要将示例内容进行输出！只输出经过你修改过后的整个标签内容
-原则3：注意以图表的格式输出的时候必须添加文本分析，不要只输出一个图表，必须图文并茂的进行输出！

### 统计数据可视化
当你需要用图片展示统计类数据时，使用图表组件展示：如年度趋势对比使用折线图，病害类型分布占比使用饼图或柱状图，排名对比使用柱状图等
-数据解释（仅用于模型理解）不需要输出：
'[]'中为图表数据，每个数据项为一个对象，对象中包含x轴数据和y轴数据。x轴数据键为name，y轴数据键为value。
axisXTitle为x轴标题，axisYTitle为y轴标题。
type为图表类型，可选值为line,pie,bar。

-示例格式：
仅仅将以下markdown返回:
#### 年度病害数量趋势图
<custom-chart 
  axisXTitle="年份" 
  axisYTitle="病害数量(个)" 
  type="line">
  [{"name":2013,"value":59},{"name":2014,"value":64},{"name":2015,"value":60}]
</custom-chart>

## 报告模板（必须逐条回答并合成）

## 3.4 风险等级评估

### 3.4.1 病害风险等级列表
说明本年度检测出的各类道路病害分别对应的风险等级是什么？

### 3.4.2 风险等级分布
分别用图表表示并分析不同风险等级的病害数量、占比情况，以及高风险病害数量最多的道路。

## 3.5 病害成因分析

### 3.5.1 报告成因汇总
综合本年度的检测报告分析，主要成因有哪些？列举出来。

### 3.5.2 成因总结
结合所有检测报告和上述列举的成因，总结一下给出本年度道路病害的成因分析。

现在开始：
Question: {{input}}
Thought:{{agent_scratchpad}}""",
    template_format="jinja2"
)

# 第4.1章提示词
prompt_chapter_4_1 = PromptTemplate.from_template(
    """
你负责生成报告的第四章第1节。

## 可用工具
{{tools}}

工具名称：{{tool_names}}

## 重要：你必须按照以下格式回答

Question: 用户的问题
Thought: 我需要做什么（用中文思考）
Action: 工具名称（从上面的工具列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 工作步骤
1. 调用 sql_search 查询整改病害数量和整改情况
2. 调用 sql_search 查询复测病害数量和复测情况（要有按编号去重和不去重的两次查询）
3. 收集完所有数据后，在 Final Answer 中输出完整报告

## 报告生成规则
1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
2. 根据各模块问题按要求逐项回答，不要遗漏
3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
4. 使用 Markdown 表格展示数据
   - 注意：表头和分隔行必须分开，不能混在一起！
5. 大标题和小标题后必须空一行
6. 语气正式客观，分析以数据为依据，逻辑清晰
7. 输出的报告中只要有问题的答案即可，不要把问题也输出

## 报告模板（必须逐条回答并合成）

## 4.1 整改情况
请给出本年度已完成整改的道路病害数量及整改率（整改病害数/总病害数），并说明尚未整改病害的数量及未整改的原因（原因由你来说明，你可以说类似以下的话：“等级Ⅳ以上的病害过于严重所以必须及时整改”，“等级Ⅲ的病害视严重程度进行整改，对于影响较重的部分需要即使整改，影响较轻的需要经常巡查”，“等级Ⅱ和Ⅰ的病害暂不整改，若情况恶化再进行整改”）。（业务逻辑：病害等级Ⅳ以上的必定会进行整改，病害等级为Ⅲ的视病害情况进行整改，Ⅱ和Ⅰ暂不整改。）

## 4.2 复测情况
本年度检测出的道路病害中，有多少处完成了复测，其复测率是多少？（是否每个被检测的道路病害都有对应复测记录，是否进行多次复测（一个编号是否有多次复测记录））

现在开始：
Question: {{input}}
Thought:{{agent_scratchpad}}""",
    template_format="jinja2"
)

# 第4.2章提示词
# prompt_chapter_4_2 = PromptTemplate.from_template(
#     """
# 你负责生成报告的第四章第2节。

# ## 可用工具
# {{tools}}

# 工具名称：{{tool_names}}

# ## 重要：你必须按照以下格式回答

# Question: 用户的问题
# Thought: 我需要做什么（用中文思考）
# Action: 工具名称（从上面的工具列表中选择）
# Action Input: 工具的输入参数
# Observation: 工具返回的结果
# ... (可以重复多次 Thought/Action/Observation)
# Thought: 我现在知道最终答案了
# Final Answer: 最终答案

# ## 工作步骤
# 1. 调用 sql_search 查询复测病害数量
# 2. 调用 sql_search 查询复测病害的类型
# 3. 收集完所有数据后，在 Final Answer 中输出完整报告

# ## 报告生成规则
# 1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
# 2. 根据各模块问题按要求逐项回答，不要遗漏
# 3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
# 4. 使用 Markdown 表格展示数据
#    - 注意：表头和分隔行必须分开，不能混在一起！
# 5. 大标题和小标题后必须空一行
# 6. 语气正式客观，分析以数据为依据，逻辑清晰
# 7. 输出的报告中只要有问题的答案即可，不要把问题也输出

# ## 报告模板（必须逐条回答并合成）

# ## 4.2 复测情况分析
# 请提供本年度各标项完成复测的道路病害数量（去重和不去重）和占比（需去重）并分析。

# 现在开始：
# Question: {{input}}
# Thought:{{agent_scratchpad}}""",
#     template_format="jinja2"
# )

# 第5章提示词
prompt_chapter_5 = PromptTemplate.from_template(
    """
你负责生成报告的第五章。

## 可用工具
{{tools}}

工具名称：{{tool_names}}

## 重要：你必须按照以下格式回答

Question: 用户的问题
Thought: 我需要做什么（用中文思考）
Action: 工具名称（从上面的工具列表中选择）
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (可以重复多次 Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

## 工作步骤
1. 调用 sql_search 查询各标段检测公司的工作数据，调用sql_search获取这一年各公司在各标项的检测道路长度，调用时需说明在检测范围表中找。
2. 查询知识库获取你需要的数据和各公司检测的设备配置，调用sql_search获取这一年各公司在各标项检测出的病害数量。
3. 收集完所有数据后，在 Final Answer 中输出完整报告

## 报告生成规则
1. 严格遵循模板中的章节结构与标题层级展开，不得增删、合并或修改章节
2. 根据各模块问题按要求逐项回答，不要遗漏
3. 若数据缺失，请用"【暂无相关数据】"标注，不得空缺或编造内容
4. 使用 Markdown 表格展示数据
   - 注意：表头和分隔行必须分开，不能混在一起！
5. 大标题和小标题后必须空一行
6. 语气正式客观，分析以数据为依据，逻辑清晰
7. 输出的报告中只要有问题的答案即可，不要把问题也输出
8. 工作天数需要查全（检测时间的格式不同，需要你自行处理）
9. 对于同一块内容（数据），在图和表中二选一使用（在图和表中选择更为合适的数据载体，而不是输出同时包含这部分数据的图和表）
10. 不要去查工作天数，评价纬度不要与工作天数，工作效率有关

## 图片回答格式
-原则1：注意必须包含完整的标签，严格按照示例的格式，只能修改示例内部的内容，格式不能进行修改！
-原则2：注意不要将示例内容进行输出！只输出经过你修改过后的整个标签内容
-原则3：注意以图表的格式输出的时候必须添加文本分析，不要只输出一个图表，必须图文并茂的进行输出！

### 统计数据可视化
当你需要用图片展示统计类数据时，使用图表组件展示：如年度趋势对比使用折线图，病害类型分布占比使用饼图或柱状图，排名对比使用柱状图等
-数据解释（仅用于模型理解）不需要输出：
'[]'中为图表数据，每个数据项为一个对象，对象中包含x轴数据和y轴数据。x轴数据键为name，y轴数据键为value。
axisXTitle为x轴标题，axisYTitle为y轴标题。
type为图表类型，可选值为line,pie,bar。

-示例格式：
仅仅将以下markdown返回:
#### 年度病害数量趋势图
<custom-chart 
  axisXTitle="年份" 
  axisYTitle="病害数量(个)" 
  type="line">
  [{"name":2013,"value":59},{"name":2014,"value":64},{"name":2015,"value":60}]
</custom-chart>

## 以下为评价标准，但是知识库中没有的内容你不能参考：
三、检测作业要求
（一）检测区域。对检测路段开展实地勘查，制定检测计划书。现场应对城市断面开展全覆盖检测，包括车行道、非机动车道、人行道、公交车站台等，对临时占用道路设施的区域如停车泊位、施工区域等也应覆盖检测。检测深度应符合合同要求。
（二）检测过程。实时记录检测过程，形成过程台账，包括检测日期、检测路段、现场人员、工作照片、全断面GPS航迹、检测数据谱图原始记录、内窥影像等。
（三）安全作业。检测车辆及人员遵守安全文明作业相关要求。车辆行驶应符合交通规则，人员应佩戴安全帽、穿着反光背心。采用手推设备检测车行道的，检测区域要进行安全围护，并配备安全员进行瞭望，确保作业安全。
（四）检测结果。检测单位对检测中发现的重大安全隐患，按照应急预案立即封闭交通并及时上报。
四、项目技术要求
（一）基本要求
1.道路雷达探测项目采用探地雷达检测方法为主，其他检测方法为辅。
2.在检测过程中如查明已形成严重隐患的土体病害时，立即以电话与书面形式通知甲方。
3.以逐条道路列表形式描述所检测出的各类病害的属性、平面位置、埋深、大小等情况，对病害严重区域配以影像资料。
4.逐条道路的道路平面简图，在图上标明各类病害所在位置。
5.对各类病害进行初步成因分析并提出处理方法建议。
6.形成所有核定检测区域的测线布置图及雷达图谱。
7.对检测数据进行整理分析，并出具检测报告。对检测报告中，疑似空洞目标经钻探验证的准确率要达到95%以上。
8.对整改完毕的问题点位进行复测，检测整改工作是否到位。

（二）技术要求
1.能探测到的道路地面以下土体病害一般具有下列基本条件：
A、土体病害的几何尺寸与其埋藏深度或探测距离之比不应小于1/5；
B、土体病害对激发的异常场应能够从干扰背景中分辨出来。
2.道路雷达检测项目投入的仪器设备应满足性能稳定、结构合理、构件牢固可靠、防潮、抗震和绝缘性能良好的要求。仪器设备应定期进行检查、校准和保养。
3.探地雷达主机主要性能和技术指标应符合下列规定：
A、系统增益不小于150dB；
B、信噪比不低于120dB，最大动态范围不低于150dB；
C、系统应具有可选的信号叠加、时窗、实时滤波、增益、点测或连续测量、位置标记等功能；
D、计时误差不应大于1.0ns；
E、最小采样间隔应达到0.5ns，A/D转换不应低于16bit；
F、工作温度：-10℃~40℃;
G、具有现场数据处理功能。
4.探地雷达天线选择应符合下列要求：
A、地面探测时应同时配置不少于两种不同频率的天线；
B、具有屏蔽功能。
5.探地雷达法的测线布设应符合下列要求：
A、测线布设应覆盖整个探测区域；
B、在路面探测地下土体病害时，应同时布设两种（含）不同频率的天线进行连续测试。两种天线测试时测线间距不宜大于2m。
6.采用车载进行检测时，车速应小于10km/h。
7.测线之间应有必要重叠，保障道路全范围充分检测。
8.应对检测到的异常区域进行详查，并采用相应的检测方法验证或核实检测结果。
（四）设备要求
1.应在合同期内自有至少1台车载检测设备及2台手推式检测设备，要能够满足多台设备同时快速作业需求，严禁借用、租赁设备。
2.车行道检测以车载三维地质雷达为主，辅以其他地球物理勘探设备。人行道检测可采用手推雷达设备。
3.车载雷达探测频率不高于400MHz。手推地质雷达配备不少于3种频率的天线，频带范围能够覆盖100-600MHz。所有检测设备应提供GPS定位轨迹
记录。
4.实地采用设备需与投标文件中设备对应，特殊情况需要更换设备的，应更换参数相同或更优设备，并提前三天书面告知采购单位。
5.同时中标多个项目的，检测设备不得在跨城区、县（市）项目间混用。
五、人员配备要求
1.项目负责人：配备1名；
2.每个检测项目至少配备1个车载班组及2个手推班组开展同步作业。单个检测项目配备不少于7人（检测技术人员具备相关项目岩土类或测绘类中级工程师及以上职称），其中必须有1名物探高级工程师(物探、岩土或地球物理类)；
3.检测项目负责人和技术负责人应常驻现场陪同检测班组开展作业，现场出勤率均不得低于70%。每次现场检测时，项目负责人和技术负责人至少有1人在现场。
4.项目组人员应于检测单位签订合同并按规定交纳社保。
5.检测单位如同时中标多个项目，现场项目人员不得在跨区、县（市）项目间混用。现场人员有变动的应提前三天前书面通知采购单位。

## 报告模板（必须逐条回答并合成）

# 五、工作情况评价
用图表展示和分析各标段的各个检测公司在本年度工作中的整体表现如何？其中哪些标段表现较好，哪些标段存在不足？

现在开始：
Question: {{input}}
Thought:{{agent_scratchpad}}""",
    template_format="jinja2"
)

# 第6章提示词（基于前面章节内容生成，不使用工具）
prompt_chapter_6 = PromptTemplate.from_template(
    """
你负责生成报告的第六章。这是报告的总结章节，需要基于前面章节的分析内容进行综合总结和建议。

## 前面章节的完整内容
{{previous_chapters}}

## 重要说明
你已经拥有前面所有章节的完整内容和数据，无需再使用任何工具查询。请直接基于上述内容进行分析和总结。

请直接在下面输出完整的第六章内容，格式如下：

# 六、结论与下阶段工作建议

## 6.1 结论
综合本年度道路病害的检测、整改和复测情况及前面章节分析的内容，给出年度总结。

## 6.2 建议
请基于前面章节分析的病害成因与分布特征、各标段工作表现以及整体工作推进情况，为下阶段道路病害相关工作的检测、整改、复测及统筹规划提供系统性建议，并明确下一年度应重点关注的区域或道路。

## 报告生成规则
1. 严格遵循上述章节结构与标题层级展开，不得增删、合并或修改章节,不要在末尾加上通过上述措施之类的话。
2. 结论要基于前面章节的具体数据和分析，引用具体数字和事实
3. 建议须具体、可落地，禁止空话，要针对前面发现的问题提出解决方案
5. 大标题和小标题后必须空一行
5. 语气正式客观，分析以数据为依据，逻辑清晰

现在请直接输出第六章的完整内容：""",
    template_format="jinja2"
)

async def generate_chapter_1(year: str, region: str, tools):
    """生成第1章（使用Agent）"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节1 开始生成',description: ' ',},")
    
    tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = prompt_chapter_1.partial(tools=tools_text, tool_names=tool_names)
    agent = create_chapter_agent(llm, tools, prompt)
    
    input_text = f"生成{year}年{region}的年度工作概述"
    result = await agent.ainvoke({"input": input_text})
    
    elapsed = time.time() - start_time
    await emit_progress(f"""{{title: '✅ 章节1 完成',description: '耗时: {elapsed:.2f}秒',}},""")
    return result.get("output", "")

async def generate_chapter_2(year: str, region: str, tools):
    """生成第2章"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节2 开始并行生成',description: ' ',},")
    
    tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = prompt_chapter_2.partial(tools=tools_text, tool_names=tool_names)
    agent = create_chapter_agent(llm, tools, prompt)
    
    input_text = f"生成{year}年{region}的工作量情况分析"
    result = await agent.ainvoke({"input": input_text})
    
    elapsed = time.time() - start_time
    await emit_progress(f"{{title: '✅ 章节2 完成',description: '耗时: {elapsed:.2f}秒',}},")
    return result.get("output", "")

async def generate_chapter_3_1_3(year: str, region: str, tools):
    """生成第3.1-3.3节"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节3.1-3.3 开始并行生成',description: ' ',},")
    
    tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = prompt_chapter_3_1_3.partial(tools=tools_text, tool_names=tool_names)
    agent = create_chapter_agent(llm, tools, prompt)
    
    input_text = f"生成{year}年{region}的病害量概述、高发道路分析和病害类型分布"
    result = await agent.ainvoke({"input": input_text})
    
    elapsed = time.time() - start_time
    await emit_progress(f"{{title: '✅ 章节3.1-3.3 完成',description: '耗时: {elapsed:.2f}秒',}},")
    return result.get("output", "")

async def generate_chapter_3_4_5(year: str, region: str, tools):
    """生成第3.4-3.5节"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节3.4-3.5 开始并行生成',description: ' ',},")
    
    tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = prompt_chapter_3_4_5.partial(tools=tools_text, tool_names=tool_names)
    agent = create_chapter_agent(llm, tools, prompt)
    
    input_text = f"生成{year}年{region}的风险等级评估和病害成因分析"
    result = await agent.ainvoke({"input": input_text})
    
    elapsed = time.time() - start_time
    await emit_progress(f"{{title: '✅ 章节3.4-3.5 完成',description: '耗时: {elapsed:.2f}秒',}},")
    return result.get("output", "")

async def generate_chapter_4_1(year: str, region: str, tools):
    """生成第4.1节"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节4.1 开始并行生成',description: ' ',},")
    
    tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = prompt_chapter_4_1.partial(tools=tools_text, tool_names=tool_names)
    agent = create_chapter_agent(llm, tools, prompt)
    
    input_text = f"生成{year}年{region}的病害处理跟踪"
    result = await agent.ainvoke({"input": input_text})
    
    elapsed = time.time() - start_time
    await emit_progress(f"{{title: '✅ 章节4.1 完成',description: '耗时: {elapsed:.2f}秒',}},")
    return result.get("output", "")

# async def generate_chapter_4_2(year: str, region: str, tools):
#     """生成第4.2节"""
#     import time
#     start_time = time.time()
#     await emit_progress("{title: '⏱️  章节4.2 开始并行生成',description: ' ',},")
    
#     tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
#     tool_names = ", ".join([t.name for t in tools])
#     prompt = prompt_chapter_4_2.partial(tools=tools_text, tool_names=tool_names)
#     agent = create_chapter_agent(llm, tools, prompt)
    
#     input_text = f"生成{year}年{region}的复测合格率分析"
#     result = await agent.ainvoke({"input": input_text})
    
#     elapsed = time.time() - start_time
#     await emit_progress(f"{{title: '✅ 章节4.2 完成',description: '耗时: {elapsed:.2f}秒',}},")
#     return result.get("output", "")

async def generate_chapter_5(year: str, region: str, tools):
    """生成第5章"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节5开始并行生成',description: ' ',},")
    
    tools_text = "\n".join([f"{t.name}: {getattr(t, 'description', '')}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = prompt_chapter_5.partial(tools=tools_text, tool_names=tool_names)
    agent = create_chapter_agent(llm, tools, prompt)
    
    input_text = f"生成{year}年{region}的工作情况评价"
    result = await agent.ainvoke({"input": input_text})
    
    elapsed = time.time() - start_time
    await emit_progress(f"{{title: '✅ 章节5 完成',description: '耗时: {elapsed:.2f}秒',}},")
    return result.get("output", "")

async def generate_chapter_6(year: str, region: str, previous_chapters: str):
    """生成第6章（基于前面章节内容，不使用工具）"""
    import time
    start_time = time.time()
    await emit_progress("{title: '⏱️  章节6 开始并行生成',description: ' ',},")
    
    # 章节6不需要工具，直接使用LLM基于前面内容生成
    prompt = prompt_chapter_6.partial(previous_chapters=previous_chapters)
    
    # 直接调用LLM，不使用Agent
    result = await llm.ainvoke(prompt.format())
    
    elapsed = time.time() - start_time
    await emit_progress(f"{{title: '✅ 章节6 完成',description: '耗时: {elapsed:.2f}秒',}},")
    return result.content if hasattr(result, 'content') else str(result)

async def clean_previous_chapters(previous_chapters: str) -> str:
    """整理前面章节的内容，清理格式并提取Final Answer"""
    import time
    start_time = time.time()
    await emit_progress("{title: '✅ 开始整理前面章节的内容 ',description: ' ',}")
    
    clean_prompt = PromptTemplate.from_template(
        """
你是一个报告格式整理专家。请整理以下报告章节内容，只修复格式问题，不要修改任何实际内容。

## 整理要求：
1. 确保Markdown格式正确，特别是表格格式，表头和分隔行必须分开。
2. 保持章节标题和内容的完整性
3. 不要修改任何实际内容，只修改格式
4. 只输出整理后的内容，不要输出任何解释或说明

## 需要整理的内容：
{previous_chapters}
"""
    )
    
    result = await llm.ainvoke(clean_prompt.format(previous_chapters=previous_chapters))
    
    # 调试输出
    cleaned_content = result.content if hasattr(result, 'content') else str(result)
    elapsed = time.time() - start_time
    await emit_progress(f"{{title:'✅ 内容整理完成',description: '耗时: {elapsed:.2f}秒',}}")
    return cleaned_content

async def generate_report_parallel(year: str, region: str):
    """并行生成完整报告"""
    import time
    total_start = time.time()
    
    await emit_progress(f"""<custom-chain>
[{{title: '🚀 开始并行生成 {year}年{region} 报告...',description: ' ',}},""")
    
    # 加载工具
    tools = await get_tools_async()
    
    # 错开启动，避免 API 限流
    await emit_progress("{title: '📊 错开启动各章节生成...',description: ' ',}")
    
    async def delayed_task(delay, task_func, *args):
        """延迟启动任务"""
        await asyncio.sleep(delay)
        return await task_func(*args)
    # 第一阶段：并行生成章节1-5（每个任务间隔1.5秒启动）
    await emit_progress("{title: '📝 第一阶段：并行生成章节1-5...',description: ' ',}")
    
    # 定义任务列表（包含任务函数、延迟、章节名称）
    tasks = [
        (generate_chapter_1, 0, "1"),
        (generate_chapter_2, 1.5, "2"),  # 第二章已注释
        (generate_chapter_3_1_3, 3, "3.1-3.3"),
        (generate_chapter_3_4_5, 4.5, "3.4-3.5"),
        (generate_chapter_4_1, 6, "4.1"),
        # (generate_chapter_4_2, 7.5, "4.2"),
        (generate_chapter_5, 9, "5"),  # 第五章已注释
    ]
    # 创建异步任务
    async_tasks = []
    for task_func, delay, chapter_name in tasks:
        if delay == 0:
            async_tasks.append(task_func(year, region, tools))
        else:
            async_tasks.append(delayed_task(delay, task_func, year, region, tools))
    
    # 并行执行任务
    results_1_5 = await asyncio.gather(*async_tasks, return_exceptions=True)
    
    for i, result in enumerate(results_1_5):
        if isinstance(result, Exception):
            actual_chapter = tasks[i][2]
            await emit_progress(f"{{title: '⚠️ 章节 {actual_chapter} 生成失败',description: '{result}',}},")
            results_1_5[i] = f"\n\n# 章节 {actual_chapter} 生成失败\n【生成过程中出现错误】\n\n"
    
    # 汇总前面章节的内容（动态构建）
    previous_chapters_parts = []
    
    # 添加第一章
    if len(results_1_5) > 0:
        previous_chapters_parts.append(results_1_5[0])
    
    # 添加第二章（如果存在）
    for i, (_, _, chapter_name) in enumerate(tasks):
        if chapter_name == "2" and i < len(results_1_5):
            previous_chapters_parts.append(results_1_5[i])
    
    # 添加第三章
    previous_chapters_parts.append("# 三、病害情况分析")
    for i, (_, _, chapter_name) in enumerate(tasks):
        if "3." in chapter_name and i < len(results_1_5):
            previous_chapters_parts.append(results_1_5[i])
    
    # 添加第四章
    previous_chapters_parts.append("# 四、处理情况分析")
    for i, (_, _, chapter_name) in enumerate(tasks):
        if "4." in chapter_name and i < len(results_1_5):
            previous_chapters_parts.append(results_1_5[i])
    
    # 添加第五章（如果存在）
    for i, (_, _, chapter_name) in enumerate(tasks):
        if chapter_name == "5" and i < len(results_1_5):
            previous_chapters_parts.append(results_1_5[i])
    
    # 合并所有章节内容
    previous_chapters = "\n".join(previous_chapters_parts)
    # 整理前面章节的内容，清理格式并提取Final Answer
    await emit_progress("{title: '📝 整理前面章节内容',description: ' ',}")
    previous_chapters = await clean_previous_chapters(previous_chapters)
    # 第二阶段：基于前面章节内容生成章节6（不使用工具）
    await emit_progress("{title: '📝 第二阶段：基于前面章节生成章节6',description: ' ',}")
    
    chapter_6 = await generate_chapter_6(year, region, previous_chapters)

    if isinstance(chapter_6, Exception):
        await emit_progress(f"{{title: '⚠️ 章节6 生成失败',description: '{chapter_6}',}},")
        chapter_6 = "\n\n# 六、结论与2026年工作建议\n【生成过程中出现错误】\n\n"

    total_elapsed = time.time() - total_start
    await emit_progress(f"{{title: '生成完成',description: '总耗时: {total_elapsed:.2f}秒',}},")
    # 汇总完整报告
    await emit_progress("{title: '📝 汇总完整报告',description: ' ',}")
    final_report = f"""# {year}年{region}道路病害年度报告
{previous_chapters}
{chapter_6}
"""

    # 打印整个报告内容，用于调试
    # print(final_report)

    # 渲染图表并生成 PDF（使用 markdown-pdf）
    await emit_progress("{title: '📊 开始渲染图表并生成 PDF.',description: ' ',}")
    from pdf_generator import PDFGenerator
    import os
    
    # 创建 PDF 生成器
    pdf_generator = PDFGenerator(static_dir="static/reports")
    
    # 生成 PDF 或 Markdown 文件
    file_path = pdf_generator.markdown_to_pdf(final_report)
    
    # 生成下载链接
    filename = os.path.basename(file_path)
    download_url = f"http://192.168.19.199:8000/static/reports/{filename}"

    # 判断文件类型
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.pdf':
        await emit_progress(f"""{{title: '✅ PDF 生成完成',description: '{download_url}',}}]
</custom-chain>""")
    else:
        await emit_progress(f"""{{title: '⚠️ PDF 生成失败，已保存为 {file_ext} 文件',description: ' {download_url}',}}]
</custom-chain>""")

    return download_url

@tool(
    "generate_report_parallel",
    description="并行生成道路病害年度报告并返回下载链接。输入如'生成2024年上城区病害报告'。优先返回PDF，失败则返回Markdown文件。",
)
async def report_parallel_tool(input: str) -> str:
    """
    道路病害年度报告生成工具
    
    Args:
        input: 报告需求，如"生成2024年上城区病害报告"
    
    Returns:
        PDF 或 Markdown 文件下载链接
    """
    # 解析输入
    import re
    # 匹配年份（4位数字）
    year_match = re.search(r'(\d{4})', input)
    # 匹配中文区域（不包含"年"字）
    region_match = re.search(r'(?:年|度)([^年度\s]+区)', input)
    if not region_match:
        region_match = re.search(r'([\u4e00-\u9fa5]{2,4}区)', input)
    
    year = year_match.group(1) if year_match else "2024"
    region = region_match.group(1) if region_match else "上城区"
    
    print(f"📋 解析结果: 年份={year}, 区域={region}")
    
    return await generate_report_parallel(year, region)
