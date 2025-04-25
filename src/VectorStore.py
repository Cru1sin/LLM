import dataclasses
from typing import List, Optional
import numpy as np

@dataclasses.dataclass
class VetcorStoreItem:
    embedding: List[float]
    document: str

class VectorStore:
    def __init__(self):
        self.vector_store: List[VetcorStoreItem] = []

    async def add_item(self, embedding: List[float], document: str):
        self.vector_store.append(VetcorStoreItem(embedding, document))

    async def search(self, query_embedding: List[float], top_k: int = 10) -> List[VetcorStoreItem]:
        similarities = [self.cosine_similarity(item.embedding, query_embedding) for item in self.vector_store]
        top_k_indices = np.argsort(similarities)[-top_k:]
        return [self.vector_store[i] for i in top_k_indices]

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dotProduct = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return dotProduct / (norm1 * norm2)
    
    
    
