"""
vector_search.py
================
Vector-RAG search: tìm kiếm ngữ nghĩa trên Milvus.

Tách từ logic search trong main.py run_query().
"""

from src.knowledge.embedding import EmbeddingManager
from src.storage.milvus_store import MilvusManager


class VectorSearchManager:
    """Tìm kiếm ngữ nghĩa (semantic search) trên Milvus."""

    def __init__(self, embedder: EmbeddingManager, milvus_db: MilvusManager):
        self.embedder = embedder
        self.milvus_db = milvus_db

    def search(self, query: str, limit: int = 3) -> str:
        """
        Tìm kiếm Vector-RAG: embed query → search Milvus → format context.

        Returns:
            Chuỗi context chứa các kết quả tìm thấy, hoặc chuỗi rỗng.
        """
        query_vector = self.embedder.encode(query)
        results = self.milvus_db.search(query_vector, limit=limit)

        ctx_lines = []
        for hits in results:
            for hit in hits:
                e = hit['entity']
                ctx_lines.append(
                    f"  - [{e['timestamp']}] [{e.get('video_id','')} / {e.get('camera_id','')}] {e['caption']}"
                )
        return "\n".join(ctx_lines)
