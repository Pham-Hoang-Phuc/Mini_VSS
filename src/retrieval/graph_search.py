"""
graph_search.py
===============
Graph-RAG query: truy vấn đồ thị ArcadeDB bằng ngôn ngữ tự nhiên.

Giữ logic query + setup_chain từ GraphRAGManager trong graph_rag.py.
"""

import os
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from configs.config import Config
from src.storage.arcadedb_store import ArcadeDBGraph

warnings.filterwarnings("ignore")

# Đặt GOOGLE_API_KEY cho langchain_google_genai
os.environ.setdefault("GOOGLE_API_KEY", Config.GEMINI_API_KEY or "")


# ──────────────────────────────────────────────────────────────
# Cypher Generation Prompt (tiếng Việt, căn chỉnh cho video chunks)
# ──────────────────────────────────────────────────────────────
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
- Analyze the question and extract relevant graph components dynamically.
- Use only the relationship types and properties from the provided schema.
- The schema is based on a graph structure with nodes and relationships:
{schema}
- Return only the generated Cypher query. No explanation, no comments.
- Use toLower() for case-insensitive matching.
- In the graph, all nodes have only one property: name.


Examples:
# Xe nâng xuất hiện trong video lúc nào?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'xe nâng'
RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS doi_tuong

# Công nhân làm gì ở cửa ra vào?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'công nhân'
      AND toLower(b.name) CONTAINS 'cửa ra vào'
RETURN a.name AS nhan_vien, type(r) AS hanh_dong, b.name AS dia_diem

# Đội bảo trì thực hiện công việc gì?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'bảo trì'
RETURN a.name AS doi_bao_tri, type(r) AS hanh_dong, b.name AS doi_tuong

# Nhân viên QC kiểm tra gì?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'qc' OR toLower(a.name) CONTAINS 'chất lượng'
RETURN a.name AS nhan_vien, type(r) AS hanh_dong, b.name AS doi_tuong

# Máy CNC được sử dụng như thế nào?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(b.name) CONTAINS 'cnc' OR toLower(a.name) CONTAINS 'cnc'
RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS may_moc

The question is:
{question}
"""


class GraphSearchManager:
    """
    Truy vấn đồ thị ArcadeDB bằng ngôn ngữ tự nhiên.
    Sử dụng GraphCypherQAChain (LangChain) để sinh Cypher từ câu hỏi.
    """

    def __init__(self, graph: ArcadeDBGraph):
        self.graph = graph
        self._chain = None  # lazy init sau khi build xong graph

    def setup_chain(self):
        """Refresh schema và khởi tạo GraphCypherQAChain."""
        self.graph.refresh_schema()

        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE,
        )
        cypher_llm = ChatGoogleGenerativeAI(model=Config.CYPHER_LLM_MODEL, temperature=0)
        qa_llm     = ChatGoogleGenerativeAI(model=Config.QA_LLM_MODEL,     temperature=0.3)

        self._chain = GraphCypherQAChain.from_llm(
            llm=qa_llm,
            cypher_llm=cypher_llm,
            graph=self.graph,
            verbose=True,           # in ra Cypher query để debug
            cypher_prompt=prompt,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,  # trả về cả Cypher đã dùng
        )
        print("  [GraphSearch] ✓ GraphCypherQAChain đã sẵn sàng.")

    def query(self, question: str) -> str:
        """
        Truy vấn đồ thị bằng ngôn ngữ tự nhiên.
        Trả về chuỗi câu trả lời hoặc chuỗi rỗng nếu lỗi.
        """
        if self._chain is None:
            self.setup_chain()
        try:
            response = self._chain.invoke({"query": question})
            return response.get("result", "")
        except Exception as e:
            print(f"  [GraphSearch] ✗ Lỗi truy vấn: {e}")
            return ""
