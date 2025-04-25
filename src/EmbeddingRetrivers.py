from VectorStore import VectorStore
from typing import List
import requests
import os
from utils.logtitles import log_title


class EmbeddingRetriever:
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self.vector_store = VectorStore()

    async def embed(self, document: str) -> List[float]:
        url = os.getenv("EMBEDDING_BASE_URL")+"/embeddings"
        payload = {
            "model": self.embedding_model,
            "input": document,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": "Bearer "+os.getenv("EMBEDDING_KEY"),
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        return response.json()["data"][0]["embedding"]
    
    async def embed_query(self, query: str) -> List[float]:
        return await self.embed(query)
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = await self.embed(documents)
        self.vector_store.add_item(embeddings, documents)
        return embeddings
    
    async def embed_konwledge(self, konwledge_path: str):
        for file in os.listdir(konwledge_path):
            if file.endswith('.md'):
                with open(os.path.join(konwledge_path, file), 'r') as f:
                    content = f.read()
                    await self.embed_documents(content)
    
    async def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = await self.embed_query(query)
        return await self.vector_store.search(query_embedding, top_k)

    async def retrieve_context(self, query: str):
        log_title("Retrieving Context")
        context = await self.retrieve(query)
        return context
