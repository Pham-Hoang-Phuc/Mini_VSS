"""
hybrid_retriever.py
===================
Hybrid Retrieval: kết hợp Vector-RAG + Graph-RAG.

Pipeline (theo vss_context.md - Mục 5):
  1. Intent Analysis: phân tích câu hỏi → "Tìm kiếm hình ảnh" hay "Truy vấn logic"
  2. Parallel Search: Vector (Milvus) + Graph (ArcadeDB)
  3. Graph Reasoning: kết nối obj_id từ Milvus với hành động lịch sử trong Graph
  4. Answer Synthesis: LLM tổng hợp từ cả 2 nguồn + video URL làm bằng chứng

TODO:
  - Implement Graph Reasoning (cross-reference Milvus results ↔ ArcadeDB entities)
  - Integrate MinIO URLs vào câu trả lời
"""

from src.retrieval.vector_search import VectorSearchManager
from src.retrieval.graph_search import GraphSearchManager


class HybridRetriever:
    """
    Kết hợp Vector-RAG + Graph-RAG + Graph Reasoning.

    Usage tương lai:
        retriever = HybridRetriever(vector_search, graph_search)
        context = retriever.retrieve(query, use_vector=True, use_graph=True)
    """

    def __init__(self, vector_search: VectorSearchManager, graph_search: GraphSearchManager):
        self.vector_search = vector_search
        self.graph_search = graph_search

    def retrieve(self, query: str, use_vector: bool = True, use_graph: bool = True) -> dict:
        """
        Truy xuất thông tin từ cả 2 nguồn.

        Returns:
            {
                "vector_context": str,  # Kết quả từ Milvus
                "graph_context": str,   # Kết quả từ ArcadeDB
            }
        """
        vector_context = ""
        graph_context = ""

        if use_vector:
            vector_context = self.vector_search.search(query)

        if use_graph:
            graph_context = self.graph_search.query(query)

        return {
            "vector_context": vector_context,
            "graph_context": graph_context,
        }
