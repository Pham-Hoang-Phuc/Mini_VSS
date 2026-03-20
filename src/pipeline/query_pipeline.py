"""
query_pipeline.py
=================
CA-RAG Query Pipeline: phân tích câu hỏi, truy xuất song song, tổng hợp đáp án.

Tách từ main.py run_query().
"""

from src.retrieval.vector_search import VectorSearchManager
from src.retrieval.graph_search import GraphSearchManager
from src.models.llm import GeminiManager


def run_query(query: str,
              vector_search: VectorSearchManager,
              graph_search: GraphSearchManager,
              ai: GeminiManager) -> str:
    """
    CA-RAG query pipeline:
      1. Route  → chọn Vector / Graph / cả hai
      2. Retrieve song song
      3. Synthesize → câu trả lời cuối
    """
    print(f"\n{'─'*55}")
    print(f"  Câu hỏi: {query}")
    print(f"{'─'*55}")

    # ── Bước 1: Query Parsing / Routing ───────────────────────
    route = ai.route_query(query)
    use_vector = route.get("use_vector", True)
    use_graph  = route.get("use_graph",  True)
    reason     = route.get("reason", "")

    tools_used = []
    if use_vector: tools_used.append("Vector-RAG")
    if use_graph:  tools_used.append("Graph-RAG")
    print(f"  [Router] Công cụ: {' + '.join(tools_used) or 'Không xác định'}")
    if reason:
        print(f"  [Router] Lý do: {reason}")

    # ── Bước 2: Parallel Retrieval ────────────────────────────
    vector_context = ""
    graph_context  = ""

    if use_vector:
        print("  → [Vector-RAG] Đang tìm kiếm Milvus...")
        vector_context = vector_search.search(query)
        if vector_context:
            print(f"  → [Vector-RAG] Tìm thấy kết quả.")
        else:
            print("  → [Vector-RAG] Không có kết quả.")

    if use_graph:
        print("  → [Graph-RAG] Đang truy vấn ArcadeDB...")
        graph_context = graph_search.query(query)
        if graph_context:
            print("  → [Graph-RAG] Có kết quả từ đồ thị.")
        else:
            print("  → [Graph-RAG] Không có kết quả từ đồ thị.")

    # ── Bước 3: Synthesize ────────────────────────────────────
    print("  → [LLM] Đang tổng hợp câu trả lời...")
    if use_vector and use_graph:
        answer = ai.generate_answer_combined(query, vector_context, graph_context)
    elif use_vector:
        answer = ai.generate_answer(query, vector_context)
    elif use_graph:
        answer = graph_context if graph_context else "Graph-RAG không tìm thấy kết quả."
    else:
        answer = "Không xác định được công cụ phù hợp."

    return answer
