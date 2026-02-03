from mcp.server.fastmcp import FastMCP
import mysql.connector
from typing import List, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv

# 加载环境变量（必须在导入 RagflowRetriever 之前）
load_dotenv()

from ragflow_retriever import RagflowRetriever

# 创建MCP实例时设置网络配置
mcp = FastMCP(
    "Demo",
    host="0.0.0.0",  # 允许外部连接
    port=8080,        # 指定端口
    sse_path="/"      # SSE端点设置为根路径
)
#这两个是固定的

# 数据库配置
DB_CONFIG_demo = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "sz"
}


def get_db_connection_people():
    """获取数据库连接"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG_people)
        return connection
    except mysql.connector.Error as e:
        raise Exception(f"数据库连接失败: {e}")
    
def get_db_connection_roadsick():
    """获取数据库连接"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG_roadsick)
        return connection
    except mysql.connector.Error as e:
        raise Exception(f"数据库连接失败: {e}")

def get_db_connection_demo():
    """获取数据库连接"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG_demo)
        return connection
    except mysql.connector.Error as e:
        raise Exception(f"数据库连接失败: {e}")

# 初始化 RAGFlow 检索器
try:
    ragflow_retriever = RagflowRetriever()
except Exception as e:
    ragflow_retriever = None
    print(f"RAGFlow 检索器初始化失败: {e}")

@mcp.tool()
def execute_sql_demo(sql: str) -> str:
    """
    执行SQL查询语句（仅支持SELECT），支持多条SQL（以分号分隔）
    
    Args:
        sql: SQL查询语句，多条SQL用分号分隔
        
    Returns:
        查询结果的JSON字符串
    """
    try:
        connection = get_db_connection_demo()
        cursor = connection.cursor(dictionary=True)
        
        # 分割多条SQL语句
        sql_statements = [stmt.strip() for stmt in sql.split(';') if stmt.strip()]
        results = []
        
        for i, single_sql in enumerate(sql_statements):
            if not single_sql:
                continue
            
            # 检查是否为SELECT语句
            if not single_sql.strip().upper().startswith('SELECT'):
                results.append({
                    "statement_index": i + 1,
                    "sql": single_sql,
                    "type": "ERROR",
                    "error": "仅支持SELECT查询，不允许执行INSERT、UPDATE、DELETE等操作"
                })
                continue
                
            cursor.execute(single_sql)
            statement_results = cursor.fetchall()
            results.append({
                "statement_index": i + 1,
                "sql": single_sql,
                "type": "SELECT",
                "results": statement_results
            })
        
        cursor.close()
        connection.close()
        
        return json.dumps(results, ensure_ascii=False, indent=2, default=str)
            
    except mysql.connector.Error as e:
        return f"SQL执行错误: {e}"
    except Exception as e:
        return f"执行错误: {e}"

@mcp.tool()
def get_current_time() -> str:
    """
    获取当前时间
    
    Returns:
        当前时间的字符串表示
    """
    from datetime import datetime
    current_time = datetime.now()
    return f"当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"

@mcp.tool()
def search_ragflow(query: str, top_k: int = 8) -> str:
    """
    在 RAGFlow 知识库中检索相关文档
    
    Args:
        query: 检索问题
        top_k: 返回结果数量，默认为4
        
    Returns:
        检索结果的JSON字符串，包含文档内容和元数据
    """
    if ragflow_retriever is None:
        return json.dumps({
            "error": "RAGFlow 检索器未初始化，请检查环境变量配置"
        }, ensure_ascii=False)
    
    try:
        from langchain_community.chat_models import ChatZhipuAI
        from langchain_core.messages import HumanMessage
        
        # 初始化 GLM 模型
        llm = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0,
            zhipuai_api_key="48552c97d20f4eff96a683eff58834df.Eomco0kx1bKJ9LwS"
        )
        
        # Query 改写
        rewrite_prompt = f"""请将下面的用户问题改写成更适合知识库检索的形式。
要求：
1. 补全省略的信息
2. 使用专业术语
3. 保持问题的核心意图
4. 只返回改写后的问题，不要其他解释

原始问题：{query}
"""
        rewritten_query = llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()
        
        # 使用改写后的问题检索文档
        docs = ragflow_retriever.invoke(rewritten_query, top_k=top_k)
        
        # 格式化结果
        results = []
        for doc in docs:
            list = doc.metadata.get("positions", [])
            results.append({
                "content": doc.page_content,
                "metadata": {
                    "filename": doc.metadata.get("filename", "未知文件"),
                    "score": doc.metadata.get("score", 0),
                    "vector_similarity": doc.metadata.get("vector_similarity", 0),
                    "term_similarity": doc.metadata.get("term_similarity", 0),
                    "document_id": doc.metadata.get("document_id", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "doc_type": doc.metadata.get("doc_type", ""),
                    "positions": list,
                    "page": list[0][0]
                }
            })
        
        response = {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "results": results
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"检索失败: {str(e)}"
        }, ensure_ascii=False)

if __name__ == "__main__":
    # 使用SSE协议，网络配置已在实例创建时设置
    mcp.run(transport="sse")