"""
llm.py
======
GeminiManager: tích hợp Gemini SDK cho CA-RAG pipeline.

Methods:
  - route_query(query)                             → phân tích câu hỏi, chọn Vector/Graph
  - generate_answer(query, context)                → trả lời từ Vector-RAG
  - generate_answer_combined(query, v_ctx, g_ctx)  → tổng hợp từ cả hai nguồn
"""

import json
from google import genai
from config import Config


class GeminiManager:
    def __init__(self):
        self.client = genai.Client(
            api_key=Config.GEMINI_API_KEY,)
        #     http_options={"api_version": "v1"}
        # )
        self.model = Config.LLM_MODEL_NAME

    # ──────────────────────────────────────────────────────────
    # [CA-RAG] Phân tích câu hỏi → chọn công cụ
    # ──────────────────────────────────────────────────────────
    def route_query(self, query: str) -> dict:
        """
        Phân tích câu hỏi để quyết định dùng Vector-RAG, Graph-RAG, hay cả hai.

        Returns:
            {
                "use_vector": bool,   # True → dùng Milvus similarity search
                "use_graph":  bool,   # True → dùng Neo4j cypher reasoning
                "reason":     str     # giải thích ngắn (debug)
            }
        """
        prompt = f"""Bạn là bộ định tuyến (router) cho hệ thống truy vấn video thông minh.
Nhiệm vụ: phân tích câu hỏi và quyết định nên dùng công cụ nào.

Công cụ có sẵn:
1. vector_search  – Tìm kiếm theo ngữ nghĩa (semantic search) trên Milvus.
   Phù hợp với: câu hỏi tìm "cảnh nào", "đoạn nào", mô tả nội dung video chung.
   Ví dụ: "Tìm cảnh có xe nâng", "Lúc nào có người trong kho?"

2. graph_query – Truy vấn đồ thị logic (Neo4j Cypher) về entity, hành động, quan hệ.
   Phù hợp với: câu hỏi về MỐI QUAN HỆ, AI LÀM GÌ, ai liên quan đến máy móc nào.
   Ví dụ: "Xe nâng làm gì?", "Công nhân QC kiểm tra gì?", "Đội bảo trì bảo trì cái gì?"

Trả lời ĐÚNG theo định dạng JSON sau, không thêm gì khác:
{{
  "use_vector": true hoặc false,
  "use_graph":  true hoặc false,
  "reason": "giải thích ngắn gọn"
}}

Câu hỏi: "{query}"
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            text = response.text.strip()
            # Bóc JSON từ markdown code block nếu có
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            result = json.loads(text)
            # Đảm bảo luôn có đủ keys
            result.setdefault("use_vector", True)
            result.setdefault("use_graph",  True)
            result.setdefault("reason",     "")
            return result
        except Exception as e:
            # Fallback: dùng cả hai khi parse lỗi
            print(f"  [Router] ⚠ Parse lỗi ({e}), fallback → dùng cả hai.")
            return {"use_vector": True, "use_graph": True, "reason": "fallback"}

    # ──────────────────────────────────────────────────────────
    # [Vector-RAG] Trả lời từ Milvus context
    # ──────────────────────────────────────────────────────────
    def generate_answer(self, query: str, context: str) -> str:
        """Tổng hợp câu trả lời từ kết quả Vector-RAG (Milvus)."""
        prompt = f"""Bạn là trợ lý AI thông minh cho hệ thống tìm kiếm video.
Dưới đây là các đoạn video tìm thấy dựa trên câu hỏi (kết quả tìm kiếm ngữ nghĩa):

{context}

Câu hỏi: "{query}"

Hãy trả lời tự nhiên, lịch sự. Nêu rõ mốc thời gian (timestamp). 
Nếu không tìm thấy thông tin phù hợp, hãy thông báo không tìm thấy."""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text

    # ──────────────────────────────────────────────────────────
    # [CA-RAG] Tổng hợp từ cả Vector + Graph context
    # ──────────────────────────────────────────────────────────
    def generate_answer_combined(
        self, query: str, vector_context: str, graph_context: str
    ) -> str:
        """
        Tổng hợp câu trả lời cuối cùng từ cả hai nguồn:
          - vector_context: kết quả semantic search từ Milvus
          - graph_context : kết quả graph reasoning từ Neo4j
        """
        parts = []
        if vector_context.strip():
            parts.append(f"[Kết quả tìm kiếm ngữ nghĩa - Milvus]\n{vector_context}")
        if graph_context.strip():
            parts.append(f"[Kết quả truy vấn đồ thị logic - Neo4j]\n{graph_context}")

        if not parts:
            return "Không tìm thấy thông tin liên quan trong video."

        combined = "\n\n".join(parts)

        prompt = f"""Bạn là trợ lý AI thông minh cho hệ thống phân tích video nhà máy (CA-RAG).
Bạn nhận được thông tin từ hai nguồn: tìm kiếm ngữ nghĩa (Vector-RAG) và truy vấn đồ thị logic (Graph-RAG).

{combined}

Câu hỏi của người dùng: "{query}"

Hãy tổng hợp thông tin từ cả hai nguồn để đưa ra câu trả lời toàn diện, tự nhiên.
Nêu rõ mốc thời gian khi có. Ưu tiên thông tin từ Graph-RAG cho câu hỏi về quan hệ/hành động,
ưu tiên Vector-RAG cho câu hỏi tìm kiếm theo mô tả."""

        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return response.text