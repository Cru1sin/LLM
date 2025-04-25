from typing import List
import numpy as np
from openai import OpenAI

class EmbeddingRetriever:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.llm = OpenAI()
        self.vector_store = []
        
    async def embed_document(self, document: str) -> List[float]:
        """将文档转换为嵌入向量"""
        response = await self.llm.embeddings.create(
            model=self.model,
            input=document
        )
        return response.data[0].embedding
        
    async def embed_query(self, query: str) -> List[float]:
        """将查询转换为嵌入向量"""
        return await self.embed_document(query)
        
    def add_to_store(self, embedding: List[float], document: str):
        """将文档和其嵌入向量添加到向量存储"""
        self.vector_store.append({
            "embedding": embedding,
            "document": document
        })
        
    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """检索最相关的文档"""
        similarities = []
        for item in self.vector_store:
            similarity = np.dot(query_embedding, item["embedding"])
            similarities.append((similarity, item["document"]))
            
        # 按相似度排序并返回top_k个文档
        similarities.sort(reverse=True)
        return [doc for _, doc in similarities[:top_k]] 