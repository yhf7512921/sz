"""
PDF 生成模块 - 将 Markdown 转换为 PDF
"""
import os
import uuid
import time
from typing import Optional
import markdown
import pdfkit
from jinja2 import Template
from chart_renderer import render_charts_in_markdown

# 导入Base64编码的字体数据
try:
    from font_base64 import FONT_BASE64
    HAS_FONT_BASE64 = True
except ImportError:
    FONT_BASE64 = None
    HAS_FONT_BASE64 = False


class PDFGenerator:
    """PDF 生成器"""
    
    def __init__(self, static_dir: str = "static/reports"):
        """
        初始化 PDF 生成器
        
        Args:
            static_dir: 静态文件存储目录
        """
        self.static_dir = static_dir
        os.makedirs(static_dir, exist_ok=True)
        
        # 配置 wkhtmltopdf 路径
        wkhtmltopdf_path = r'E:\langchain_study\pdftool\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        self.config = config
    
    def _create_html_template(self) -> Template:
        """创建 HTML 模板"""
        # 生成 @font-face 规则
        if HAS_FONT_BASE64 and FONT_BASE64:
            font_face_rule = """
        @font-face {
            font-family: 'FangSong';
            src: url('data:font/truetype;base64," + FONT_BASE64 + "') format('truetype');
            font-weight: normal;
            font-style: normal;
        }
            """
        else:
            font_face_rule = """
        @font-face {
            font-family: 'FangSong';
            src: url('file:///C:/WINDOWS/FONTS/SIMFANG.TTF') format('truetype');
            font-weight: normal;
            font-style: normal;
        }
            """
        
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>报告</title>
    <style>
""" + font_face_rule + """
        body {
            font-family: 'FangSong', "SimSun", "仿宋", "宋体", "KaiTi", "楷体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", serif;
            line-height: 1.8;
            margin: 40px;
            padding: 0;
            color: #333;
            font-size: 12pt;
        }
        
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: bold;
            page-break-after: avoid;
        }
        
        h1 {
            text-align: center;
            font-size: 28pt;
            margin-bottom: 30px;
            color: #2c3e50;
        }
        
        h2 {
            font-size: 20pt;
            margin-top: 30px;
            margin-bottom: 16px;
            padding-left: 12px;
            border-left: 5px solid #3498db;
            color: #2c3e50;
        }
        
        h3 {
            font-size: 16pt;
            margin-top: 16px;
            margin-bottom: 14px;
            color: #34495e;
        }
        
        h4 {
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 12px;
            color: #34495e;
        }
        
        h5 {
            font-size: 12pt;
            margin-top: 18px;
            margin-bottom: 10px;
            color: #34495e;
        }
        
        h6 {
            font-size: 11pt;
            margin-top: 16px;
            margin-bottom: 10px;
            color: #34495e;
        }
        
        p {
            margin: 8px 0;
            text-align: justify;
            text-indent: 2em;
            line-height: 1.5;
            font-size: 12pt;
            font-family: 'FangSong', "SimSun", "仿宋", "宋体", "KaiTi", "楷体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", serif;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        
        br {
            display: block;
            margin: 8px 0;
            content: "";
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 12pt;
            font-family: 'FangSong', "SimSun", "仿宋", "宋体", "KaiTi", "楷体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }
        
        tr {
            page-break-inside: avoid;
            page-break-after: avoid;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: center;
            border: 1px solid #2980b9;
            page-break-inside: avoid;
            page-break-after: avoid;
            font-size: 12pt;
        }
        
        td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: center;
            page-break-inside: avoid;
            font-size: 12pt;
            font-family: 'FangSong', "SimSun", "仿宋", "宋体", "KaiTi", "楷体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", serif;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        tr:hover {
            background-color: #e9ecef;
        }
        
        ul, ol {
            margin: 8px 0;
            padding-left: 30px;
            font-family: 'FangSong', "SimSun", "仿宋", "宋体", "KaiTi", "楷体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", serif;
        }

        li {
            margin: 4px 0;
            line-height: 1.4;
            font-family: 'FangSong', "SimSun", "仿宋", "宋体", "KaiTi", "楷体", "SimHei", "黑体", "Microsoft YaHei", "微软雅黑", serif;
        }
        
        img {
            max-width: 80%;
            height: auto;
            display: block;
            margin: 12px auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        blockquote {
            margin: 20px 0;
            padding: 12px 20px;
            border-left: 5px solid #3498db;
            background-color: #f8f9fa;
            color: #555;
            font-style: italic;
        }
        
        hr {
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 0.9em;
        }
        
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
        }
        
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .page-break {
            page-break-after: always;
        }
    </style>
</head>
<body>
    {{ content }}
</body>
</html>
        """
        # 修复模板中的Base64数据插入
        if HAS_FONT_BASE64 and FONT_BASE64:
            template_str = template_str.replace('" + FONT_BASE64 + "', FONT_BASE64)
        return Template(template_str)
    
    def markdown_to_pdf(self, markdown_text: str, output_path: Optional[str] = None) -> str:
        """
        将 Markdown 转换为 PDF
        
        Args:
            markdown_text: Markdown 文本
            output_path: 输出 PDF 路径，如果不指定则自动生成
        
        Returns:
            PDF 文件路径
        """
        total_start = time.time()
        
        # 1. 渲染图表
        print("📊 开始渲染图表...")
        chart_start = time.time()
        markdown_with_images = render_charts_in_markdown(markdown_text)
        chart_elapsed = time.time() - chart_start
        print(f"✅ 图表渲染完成，耗时: {chart_elapsed:.2f}秒")
        
        # 2. 将 Markdown 转换为 HTML（支持表格、代码块等）
        print("📝 开始转换 Markdown 为 HTML...")
        md_start = time.time()
        md = markdown.Markdown(extensions=[
            'tables',
            'fenced_code',
            'codehilite',
            'toc',
            'nl2br',
            'sane_lists'
        ])
        html_content = md.convert(markdown_with_images)
        md_elapsed = time.time() - md_start
        print(f"✅ Markdown 转换完成，耗时: {md_elapsed:.2f}秒")
        
        # 3. 使用 Jinja2 模板包装 HTML（添加样式）
        print("🎨 开始包装 HTML 模板...")
        template_start = time.time()
        template = self._create_html_template()
        full_html = template.render(content=html_content)
        template_elapsed = time.time() - template_start
        print(f"✅ HTML 模板包装完成，耗时: {template_elapsed:.2f}秒")
        
        # 4. 生成 PDF 路径
        if output_path is None:
            output_path = os.path.join(
                self.static_dir,
                f"report_{uuid.uuid4()}_{int(time.time())}.pdf"
            )
        
        # 5. 使用 pdfkit 生成 PDF
        print("📄 开始生成 PDF...")
        pdf_start = time.time()
        try:
            options = {
                'encoding': 'UTF-8',
                'quiet': '',
                'margin-top': '20mm',
                'margin-right': '20mm',
                'margin-bottom': '20mm',
                'margin-left': '20mm',
                'enable-local-file-access': None,
                'no-stop-slow-scripts': None,
                'disable-smart-shrinking': None
            }
            
            if self.config:
                pdfkit.from_string(full_html, output_path, options=options, configuration=self.config)
            else:
                pdfkit.from_string(full_html, output_path, options=options)
            
            pdf_elapsed = time.time() - pdf_start
            print(f"✅ PDF 生成成功，耗时: {pdf_elapsed:.2f}秒")
            print(f"✅ PDF 文件路径: {output_path}")
            
            total_elapsed = time.time() - total_start
            print(f"⏱️  PDF生成总耗时: {total_elapsed:.2f}秒")
            print(f"   - 图表渲染: {chart_elapsed:.2f}秒 ({chart_elapsed/total_elapsed*100:.1f}%)")
            print(f"   - Markdown转换: {md_elapsed:.2f}秒 ({md_elapsed/total_elapsed*100:.1f}%)")
            print(f"   - HTML模板: {template_elapsed:.2f}秒 ({template_elapsed/total_elapsed*100:.1f}%)")
            print(f"   - PDF生成: {pdf_elapsed:.2f}秒 ({pdf_elapsed/total_elapsed*100:.1f}%)")
            
            return output_path
        except Exception as e:
            # 如果 pdfkit 失败，尝试备用方案
            print(f"⚠️ PDF 生成失败，尝试备用方案: {e}")
            # 备用方案：直接保存为 .md 文件
            md_path = output_path.replace('.pdf', '.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_with_images)
            print(f"📝 已保存为 Markdown 文件: {md_path}")
            return md_path
    
    def generate_report(self, markdown_content: str, base_url: str = "http://localhost:8000") -> dict:
        """
        生成报告并返回下载链接
        
        Args:
            markdown_content: Markdown 格式的报告内容
            base_url: 基础 URL
        
        Returns:
            包含报告信息的字典
        """
        try:
            # 生成 PDF 或 Markdown 文件
            file_path = self.markdown_to_pdf(markdown_content)
            
            # 获取文件名和扩展名
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1]
            
            # 根据文件类型返回不同的 URL
            if file_ext == '.pdf':
                download_url = f"{base_url}/static/reports/{filename}"
                message = "报告生成成功"
            else:
                download_url = f"{base_url}/static/reports/{filename}"
                message = f"报告已生成（{file_ext} 格式，PDF 生成失败）"
            
            # 返回下载链接
            return {
                "success": True,
                "report_id": os.path.splitext(filename)[0],
                "download_url": download_url,
                "file_path": file_path,
                "file_type": file_ext,
                "message": message
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "报告生成失败"
            }


def generate_pdf_report(markdown_content: str, static_dir: str = "static/reports", base_url: str = "http://localhost:8000") -> dict:
    """
    生成 PDF 报告的便捷函数
    
    Args:
        markdown_content: Markdown 格式的报告内容
        static_dir: 静态文件存储目录
        base_url: 基础 URL
    
    Returns:
        包含报告信息的字典
    """
    generator = PDFGenerator(static_dir)
    return generator.generate_report(markdown_content, base_url)


def test_font():
    """
    测试字体是否正确加载
    生成一个简单的测试PDF来验证仿宋字体是否正常工作
    """
    test_markdown = """# 字体测试

## 正文测试
这是一段测试文字，用于验证仿宋字体是否正确显示。字体大小应该是小四号（12pt）。

## 表格测试
| 字体名称 | 字体大小 | 显示效果 |
|---------|---------|---------|
| 仿宋 | 12pt | 测试文字 |
| 宋体 | 12pt | 测试文字 |
| 楷体 | 12pt | 测试文字 |

## 列表测试

### 测试1：无标题的列表
- 第一项测试文字
- 第二项测试文字
- 第三项测试文字

### 测试2：有标题的列表
2024年：
- 第一项测试文字
- 第二项测试文字
- 第三项测试文字

### 测试3：有标题带空行的列表
2024年：

- 第一项测试文字
- 第二项测试文字
- 第三项测试文字

## 混合测试
正文段落应该使用仿宋字体，大小为12pt。表格和列表也应该使用相同的字体设置。
"""
    
    print("🔍 开始字体测试...")
    generator = PDFGenerator("static/reports")
    result = generator.markdown_to_pdf(test_markdown, "static/reports/font_test.pdf")
    print(f"✅ 字体测试完成，PDF文件: {result}")
    return result


if __name__ == "__main__":
    test_font()
