"""
RAGFlow 知识库检索 -> LangChain Retriever
环境变量：
  RAGFLOW_API_BASE    RAGFlow API 基地址
  RAGFLOW_TOKEN       API密钥
  RAGFLOW_DATASET_ID  对应知识库的ID
"""

import os
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin
import httpx         
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import asyncio

logger = logging.getLogger(__name__)

def _make_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))

class RagflowRetriever(BaseRetriever):
    base_url: str   = os.getenv("RAGFLOW_API_BASE", "http://localhost:8000")
    api_key: str    = os.getenv("RAGFLOW_TOKEN", "ragflow-9ppdsyOXmGF7shW2VjKgovwnSpAg2Rq4628JgbQQoY0")
    dataset_id: str = os.getenv("RAGFLOW_DATASET_ID", "04f1f520d4c811f0a5cd0242ac1b0006")
    top_k: int      = 8
    verify_ssl: bool = True
    timeout: int    = 30

    def __init__(self, **data):
        super().__init__(**data)
        if not self.api_key:
            raise ValueError("RAGFLOW_TOKEN is required")
        if not self.dataset_id:
            raise ValueError("RAGFLOW_DATASET_ID is required")
       
        self._client = httpx.Client(
            base_url=_make_url(self.base_url, ""),
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def _get_relevant_documents(
        self, query: str, *, top_k: Optional[int] = None, **kwargs
    ) -> List[Document]:
        top_k = top_k or self.top_k
        url = _make_url(self.base_url, "api/v1/retrieval")
        
        payload = {
            "question": query,
            "dataset_ids": [self.dataset_id],
            "top_k": top_k,
        }
        
        # 可选的额外参数
        if "search_method" in kwargs:
            payload["search_method"] = kwargs["search_method"]
        if "rerank" in kwargs:
            payload["rerank"] = kwargs["rerank"]
        if "similarity_threshold" in kwargs:
            payload["similarity_threshold"] = kwargs["similarity_threshold"]
        
        try:
            r = self._client.post(url, json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("[RAGFlow] HTTP %s: %s", e.response.status_code, e.response.text[:200])
            return []
        except httpx.RequestError as e:
            logger.error("[RAGFlow] Request failed: %s", e)
            return []

        data = r.json()
        if data.get("code") != 0:
            logger.error("[RAGFlow] Business error: %s", data.get("message"))
            return []

        # 处理返回的数据结构
        chunks_data = data.get("data", {})
        if isinstance(chunks_data, dict):
            chunks = chunks_data.get("chunks", [])
        else:
            chunks = chunks_data if isinstance(chunks_data, list) else []

        # 强制裁剪，避免服务端返回超过 top_k
        if isinstance(chunks, list) and top_k is not None:
            chunks = chunks[:top_k]

        docs = []
        for ch in chunks:
            if isinstance(ch, dict):
                text = ch.get("content") or ch.get("text") or ""
                if not text:
                    continue
                
                # 提取PDF文件名和其他元数据
                meta = self._extract_metadata(ch)
                meta["score"] = ch.get("similarity", ch.get("score", 0))
                meta["vector_similarity"] = ch.get("vector_similarity", 0)
                meta["term_similarity"] = ch.get("term_similarity", 0)
                
                docs.append(Document(page_content=text, metadata=meta))
        
        
        return docs
    
    def _extract_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """提取并整理文档元数据"""
        metadata = {}
        
        # 提取PDF文件名
        if "document_keyword" in chunk:
            metadata["filename"] = chunk["document_keyword"]
        elif "doc_name" in chunk:
            metadata["filename"] = chunk["doc_name"]
        else:
            metadata["filename"] = "未知文件"
        
        # 提取文档ID
        if "document_id" in chunk:
            metadata["document_id"] = chunk["document_id"]
        
        # 提取切片ID
        if "id" in chunk:
            metadata["chunk_id"] = chunk["id"]
        
        # 提取文档类型
        if "doc_type_kwd" in chunk:
            metadata["doc_type"] = chunk["doc_type_kwd"]
        
        # 提取图像ID
        if "image_id" in chunk:
            metadata["image_id"] = chunk["image_id"]
        
        # 提取位置信息
        if "positions" in chunk:
            metadata["positions"] = chunk["positions"]
        
        # 提取其他可能相关的字段
        for key in ["dataset_id", "highlight"]:
            if key in chunk:
                metadata[key] = chunk[key]
        
        return metadata

    async def _aget_relevant_documents(
        self, query: str, *, top_k: Optional[int] = None, **kwargs
    ) -> List[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query, top_k=top_k, **kwargs)