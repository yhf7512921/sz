"""
图表渲染模块 - 使用 pyecharts 生成图表并转换为图片
"""
import re
import json
import base64
import os
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Pie
from pyecharts.globals import CurrentConfig, ThemeType

# 优先使用本地 ECharts 资源，找不到则回退到 CDN
_LOCAL_ECHARTS_DIR = os.environ.get(
    "ECharts_LOCAL_DIR",
    os.path.join(os.path.dirname(__file__), "assets", "echarts", "")
)
if os.path.exists(_LOCAL_ECHARTS_DIR):
    CurrentConfig.ONLINE_HOST = _LOCAL_ECHARTS_DIR.replace("\\", "/")
else:
    CurrentConfig.ONLINE_HOST = "https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/"

# 最大并发数
MAX_WORKERS = 4


class ChartRenderer:
    """图表渲染器"""
    
    def __init__(self, output_dir: str = "temp_charts"):
        """
        初始化图表渲染器
        
        Args:
            output_dir: 临时图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def _render_single_chart(self, i: int, total: int, attrs: str, data_json: str) -> Tuple[str, Optional[str]]:
        """
        渲染单个图表
        
        Args:
            i: 图表索引
            total: 总图表数
            attrs: 图表属性
            data_json: 图表数据
        
        Returns:
            (chart_tag, img_base64): 图表标签和渲染后的图片Base64
        """
        max_retries = 6
        retry_delay = 8  # 秒
        
        chart_start = time.time()
        attrs_dict = self._parse_attrs(attrs)
        chart_type = attrs_dict.get('type', 'unknown')
        chart_tag = f'<custom-chart {attrs}>{data_json}</custom-chart>'
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"📊 [{i}/{total}] 第 {attempt+1} 次重试渲染图表 (类型: {chart_type})...")
                    time.sleep(retry_delay)
                else:
                    print(f"📊 [{i}/{total}] 开始渲染图表 (类型: {chart_type})...")
                
                # 解析数据
                data = json.loads(data_json)
                x_data = [item['name'] for item in data]
                y_data = [item['value'] for item in data]
                
                # 生成图表
                chart = self._build_chart(attrs_dict, x_data, y_data)
                
                # 渲染为图片
                img_base64 = self._render_to_base64(chart)
                
                chart_elapsed = time.time() - chart_start
                if img_base64:
                    print(f"✅ [{i}/{total}] 图表渲染完成，耗时: {chart_elapsed:.2f}秒")
                    return chart_tag, img_base64
                else:
                    print(f"⚠️ [{i}/{total}] 图表渲染失败，准备重试...")
                    continue
                
            except Exception as e:
                chart_elapsed = time.time() - chart_start
                print(f"⚠️ [{i}/{total}] 图表渲染失败: {e}，耗时: {chart_elapsed:.2f}秒")
                if attempt < max_retries - 1:
                    print(f"📊 [{i}/{total}] 将在 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"❌ [{i}/{total}] 达到最大重试次数，渲染失败")
                    return chart_tag, None
        
        return chart_tag, None
    
    def parse_custom_chart(self, markdown_text: str) -> str:
        """
        解析 Markdown 中的 custom-chart 标签并替换为图片
        
        Args:
            markdown_text: 包含 custom-chart 标签的 Markdown 文本
        
        Returns:
            替换后的 Markdown 文本
        """
        pattern = r'<custom-chart\s+([^>]+)>(.*?)</custom-chart>'
        matches = re.findall(pattern, markdown_text, re.DOTALL)
        
        total_charts = len(matches)
        print(f"📊 找到 {total_charts} 个图表需要渲染")
        
        if total_charts == 0:
            return markdown_text
        
        # 准备渲染任务
        tasks = []
        for i, (attrs, data_json) in enumerate(matches, 1):
            tasks.append((i, total_charts, attrs, data_json))
        
        # 并行渲染图表
        chart_results = []
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_charts)) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self._render_single_chart, *task): task for task in tasks}
            
            # 收集结果
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    chart_results.append(result)
                except Exception as e:
                    print(f"❌ 任务执行失败: {e}")
        
        # 替换图表标签为图片
        for chart_tag, img_base64 in chart_results:
            if img_base64:
                img_tag = f'![图表](data:image/png;base64,{img_base64})'
                markdown_text = markdown_text.replace(chart_tag, img_tag)
        
        return markdown_text
    
    def _parse_attrs(self, attrs: str) -> Dict[str, str]:
        """
        解析标签属性
        
        Args:
            attrs: 属性字符串
        
        Returns:
            属性字典
        """
        attrs_dict = {}
        for attr in attrs.split():
            if '=' in attr:
                key, value = attr.split('=', 1)
                attrs_dict[key.strip()] = value.strip('"\'')
        return attrs_dict
    
    def _build_chart(self, attrs_dict: Dict[str, str], x_data: List, y_data: List):
        """
        构建 ECharts 图表
        
        Args:
            attrs_dict: 图表属性
            x_data: x 轴数据
            y_data: y 轴数据
        
        Returns:
            pyecharts 图表对象
        """
        chart_type = attrs_dict.get('type', 'line')
        axis_x_title = attrs_dict.get('axisXTitle', '')
        axis_y_title = attrs_dict.get('axisYTitle', '')
        
        if chart_type == 'line':
            chart = self._build_line_chart(x_data, y_data, axis_x_title, axis_y_title)
        elif chart_type == 'bar':
            chart = self._build_bar_chart(x_data, y_data, axis_x_title, axis_y_title)
        elif chart_type == 'pie':
            chart = self._build_pie_chart(x_data, y_data, axis_y_title)
        else:
            # 默认使用折线图
            chart = self._build_line_chart(x_data, y_data, axis_x_title, axis_y_title)
        
        return chart
    
    def _build_line_chart(self, x_data: List, y_data: List, axis_x_title: str, axis_y_title: str) -> Line:
        """
        构建折线图
        
        Args:
            x_data: x 轴数据
            y_data: y 轴数据
            axis_x_title: x 轴标题
            axis_y_title: y 轴标题
        
        Returns:
            Line 图表对象
        """
        c = (
            Line(init_opts=opts.InitOpts(
                width="2000px",
                height="1200px",
                theme=ThemeType.WHITE
            ))
            .add_xaxis(x_data)
            .add_yaxis(
                series_name="数值",
                y_axis=y_data,
                symbol="circle",
                symbol_size=8,
                is_smooth=True,
                label_opts=opts.LabelOpts(is_show=False),
                linestyle_opts=opts.LineStyleOpts(width=3)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=axis_y_title,
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=18,
                        font_weight="bold"
                    )
                ),
                xaxis_opts=opts.AxisOpts(
                    name=axis_x_title,
                    name_location="middle",
                    name_gap=30,
                    axislabel_opts=opts.LabelOpts(
                        font_size=12,
                        rotate=0 if len(x_data) <= 10 else 30
                    )
                ),
                yaxis_opts=opts.AxisOpts(
                    name=axis_y_title,
                    name_location="middle",
                    name_gap=50,
                    axislabel_opts=opts.LabelOpts(font_size=12)
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="cross"
                ),
                legend_opts=opts.LegendOpts(
                    pos_left="5%",
                    orient="vertical"
                )
            )
        )
        return c
    
    def _build_bar_chart(self, x_data: List, y_data: List, axis_x_title: str, axis_y_title: str) -> Bar:
        """
        构建柱状图
        
        Args:
            x_data: x 轴数据
            y_data: y 轴数据
            axis_x_title: x 轴标题
            axis_y_title: y 轴标题
        
        Returns:
            Bar 图表对象
        """
        c = (
            Bar(init_opts=opts.InitOpts(
                width="1000px",
                height="600px",
                theme=ThemeType.WHITE
            ))
            .add_xaxis(x_data)
            .add_yaxis(
                series_name="数值",
                y_axis=y_data,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#5470c6",
                    border_radius=[4, 4, 0, 0]
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=axis_y_title,
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=18,
                        font_weight="bold"
                    )
                ),
                xaxis_opts=opts.AxisOpts(
                    name=axis_x_title,
                    name_location="middle",
                    name_gap=30,
                    axislabel_opts=opts.LabelOpts(
                        font_size=12,
                        rotate=0 if len(x_data) <= 10 else 30
                    )
                ),
                yaxis_opts=opts.AxisOpts(
                    name=axis_y_title,
                    name_location="middle",
                    name_gap=50,
                    axislabel_opts=opts.LabelOpts(font_size=12)
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type="shadow"
                ),
                legend_opts=opts.LegendOpts(
                    pos_left="5%",
                    orient="vertical"
                )
            )
        )
        return c
    
    def _build_pie_chart(self, x_data: List, y_data: List, title: str) -> Pie:
        """
        构建饼图
        
        Args:
            x_data: 类别名称
            y_data: 数值
            title: 图表标题
        
        Returns:
            Pie 图表对象
        """
        # 构建饼图数据
        pie_data = [[x, y] for x, y in zip(x_data, y_data)]
        
        c = (
            Pie(init_opts=opts.InitOpts(
                width="1000px",
                height="600px",
                theme=ThemeType.WHITE
            ))
            .add(
                series_name="数值",
                data_pair=pie_data,
                radius=["40%", "70%"],
                label_opts=opts.LabelOpts(
                    formatter="{b}: {d}%"
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=title,
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=18,
                        font_weight="bold"
                    )
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="item",
                    formatter="{a} <br/>{b}: {c} ({d}%)"
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical",
                    pos_left="5%",
                    pos_top="center"
                )
            )
        )
        return c
    
    def _render_to_base64(self, chart) -> str:
        """
        将图表渲染为 base64 编码的图片
        
        Args:
            chart: pyecharts 图表对象
        
        Returns:
            base64 编码的图片字符串
        """
        # 生成临时 HTML 文件
        temp_html = os.path.join(self.output_dir, f"temp_{os.urandom(8).hex()}.html")
        
        # 渲染为 HTML 文件
        chart.render(temp_html)
        
        # 读取并修改 HTML，确保 echarts 被正确加载
        with open(temp_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 在 head 中添加 echarts 引用（如果不存在）
        echarts_script = '<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>'
        if 'echarts.min.js' not in html_content:
            html_content = html_content.replace('<head>', f'<head>\n    {echarts_script}')
        
        # 写回文件
        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 使用 snapshot-selenium 渲染为图片
        try:
            from pyecharts.render import make_snapshot
            from snapshot_selenium import snapshot
            
            temp_png = temp_html.replace('.html', '.png')
            
            # 渲染为图片
            make_snapshot(snapshot, temp_html, temp_png, delay=10)
            
            # 读取图片并编码
            with open(temp_png, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode()
            
            # 清理临时文件
            if os.path.exists(temp_html):
                os.remove(temp_html)
            if os.path.exists(temp_png):
                os.remove(temp_png)
            
            return img_base64
            
        except ImportError:
            # 如果没有安装 snapshot-selenium，尝试 snapshot-phantomjs
            try:
                from pyecharts.render import make_snapshot
                from snapshot_phantomjs import snapshot
                
                temp_png = temp_html.replace('.html', '.png')
                make_snapshot(snapshot, temp_html, temp_png)
                
                # 读取图片并编码
                with open(temp_png, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                
                # 清理临时文件
                if os.path.exists(temp_html):
                    os.remove(temp_html)
                if os.path.exists(temp_png):
                    os.remove(temp_png)
                
                return img_base64
                
            except ImportError:
                # 如果都没有安装，清理临时文件并返回 None
                print("⚠️ 未安装 snapshot-selenium 或 snapshot-phantomjs，无法渲染图表")
                if os.path.exists(temp_html):
                    os.remove(temp_html)
                return None


def render_charts_in_markdown(markdown_text: str, output_dir: str = "temp_charts") -> str:
    """
    渲染 Markdown 中的所有图表
    
    Args:
        markdown_text: 包含 custom-chart 标签的 Markdown 文本
        output_dir: 临时图表输出目录
    
    Returns:
        替换后的 Markdown 文本
    """
    renderer = ChartRenderer(output_dir)
    return renderer.parse_custom_chart(markdown_text)
