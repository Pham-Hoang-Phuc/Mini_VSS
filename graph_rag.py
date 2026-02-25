"""
graph_rag.py
============
Module Graph-RAG: quản lý toàn bộ logic Neo4j cho CA-RAG pipeline.
Tái sử dụng patterns từ graph_rag_complete.py, đóng gói thành class.

Chức năng:
  - extract_and_store(): gọi Gemini trích xuất entity/relation từ caption,
    MERGE vào Neo4j kèm property timestamp.
  - query(): trả lời câu hỏi dựa trên đồ thị (GraphCypherQAChain).
"""

import os
import re
import warnings
from openai import OpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.prompt import PromptTemplate
from config import Config

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
RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS doi_tuong,
       a.timestamp AS thoi_gian

# Công nhân làm gì ở cửa ra vào?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'công nhân'
      AND toLower(b.name) CONTAINS 'cửa ra vào'
RETURN a.name AS nhan_vien, type(r) AS hanh_dong, b.name AS dia_diem,
       a.timestamp AS thoi_gian

# Đội bảo trì thực hiện công việc gì?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'bảo trì'
RETURN a.name AS doi_bao_tri, type(r) AS hanh_dong, b.name AS doi_tuong,
       a.timestamp AS thoi_gian

# Nhân viên QC kiểm tra gì?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'qc' OR toLower(a.name) CONTAINS 'chất lượng'
RETURN a.name AS nhan_vien, type(r) AS hanh_dong, b.name AS doi_tuong,
       a.timestamp AS thoi_gian

# Máy CNC được sử dụng như thế nào?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(b.name) CONTAINS 'cnc' OR toLower(a.name) CONTAINS 'cnc'
RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS may_moc,
       a.timestamp AS thoi_gian

The question is:
{question}
"""


class GraphRAGManager:
    """
    Quản lý Graph-RAG: kết nối Neo4j, trích xuất entity/relation từ caption,
    và truy vấn đồ thị bằng ngôn ngữ tự nhiên.
    """

    def __init__(self):
        self.kg = Neo4jGraph(
            url=Config.NEO4J_URI,
            username=Config.NEO4J_USERNAME,
            password=Config.NEO4J_PASSWORD,
            database=Config.NEO4J_DATABASE,
        )
        self._chain = None  # lazy init sau khi build xong graph
        print("  [GraphRAG] ✓ Kết nối Neo4j thành công.")

    # ──────────────────────────────────────────────────────────
    # 1. Trích xuất Entity & Relationship bằng Gemini
    # ──────────────────────────────────────────────────────────
    def _extract_entities_and_relationships(self, text: str) -> str:
        """Gọi Gemini (OpenAI-compatible) để trích xuất từ caption."""
        client = OpenAI(
            api_key=Config.GEMINI_API_KEY,
            base_url=Config.GEMINI_BASE_URL,
        )

        prompt = (
            f"Extract entities (nodes) and their relationships (edges) from the text below. "
            f"Entities and relationships MUST be in Vietnamese.\n"
            f"Follow this exact format:\n\n"
            f"Entities:\n"
            f"- {{Entity}}: {{Type}}\n\n"
            f"Relationships:\n"
            f"- ({{Entity1}}, {{RelationshipType}}, {{Entity2}})\n\n"
            f"Text:\n\"{text}\"\n\n"
            f"Output:\nEntities:\n- {{Entity}}: {{Type}}\n...\n\n"
            f"Relationships:\n- ({{Entity1}}, {{RelationshipType}}, {{Entity2}})\n"
        )

        response = client.chat.completions.create(
            model=Config.GRAPH_LLM_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            response_format={"type": "text"},
            temperature=1,
            max_tokens=2048,
            top_p=1,
        )
        return response.choices[0].message.content

    # ──────────────────────────────────────────────────────────
    # 2. Parse output LLM
    # ──────────────────────────────────────────────────────────
    def _parse_llm_output(self, result: str):
        """Parse output của Gemini thành entity_dict và relationship_list."""
        entity_pattern = r"- (.+): (.+)"
        entities = re.findall(entity_pattern, result)
        entity_dict = {e.strip(): t.strip() for e, t in entities}

        relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
        relationships = re.findall(relationship_pattern, result)
        relationship_list = [
            (s.strip(), r.strip().replace(" ", "_").upper(), o.strip())
            for s, r, o in relationships
        ]
        return entity_dict, relationship_list

    # ──────────────────────────────────────────────────────────
    # 3. MERGE vào Neo4j (kèm timestamp và camera_id)
    # ──────────────────────────────────────────────────────────
    def _merge_to_neo4j(self, relationships: list):
        """MERGE nodes và relationships vào Neo4j (chỉ lưu tạo hạt (Entity) và quan hệ)."""
        with self.kg._driver.session() as session:
            for subject, relation, obj in relationships:
                cypher = f"""
                MERGE (a:Entity {{name: $subject}})
                MERGE (b:Entity {{name: $object}})
                MERGE (a)-[:`{relation}`]->(b)
                """
                session.run(cypher, subject=subject, object=obj)

    # ──────────────────────────────────────────────────────────
    # PUBLIC: extract_and_store (gọi từ ingestion pipeline)
    # ──────────────────────────────────────────────────────────
    def extract_and_store(self, caption: str):
        """
        Trích xuất entity/relation từ caption và MERGE vào Neo4j.
        Neo4j chỉ lưu nội dung (Entity + Relationship), không lưu metadata.
        Trả về số lượng relationship được thêm (0 nếu lỗi).
        """
        try:
            raw = self._extract_entities_and_relationships(caption)
            entity_dict, rel_list = self._parse_llm_output(raw)

            if rel_list:
                self._merge_to_neo4j(rel_list)
                print(f"    [GraphRAG] +{len(rel_list)} relationships | {len(entity_dict)} entities")
            else:
                print(f"    [GraphRAG] ⚠ Không tìm thấy relationship trong chunk này.")
            return len(rel_list)
        except Exception as e:
            print(f"    [GraphRAG] ✗ Lỗi: {e}")
            return 0

    # ──────────────────────────────────────────────────────────
    # PUBLIC: setup_chain (gọi sau khi ingestion xong)
    # ──────────────────────────────────────────────────────────
    def setup_chain(self):
        """Refresh schema và khởi tạo GraphCypherQAChain."""
        self.kg.refresh_schema()

        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=CYPHER_GENERATION_TEMPLATE,
        )
        cypher_llm = ChatGoogleGenerativeAI(model=Config.CYPHER_LLM_MODEL, temperature=0)
        qa_llm     = ChatGoogleGenerativeAI(model=Config.QA_LLM_MODEL,     temperature=0.3)

        self._chain = GraphCypherQAChain.from_llm(
            llm=qa_llm,
            cypher_llm=cypher_llm,
            graph=self.kg,
            verbose=True,           # in ra Cypher query để debug
            cypher_prompt=prompt,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,  # trả về cả Cypher đã dùng
        )
        print("  [GraphRAG] ✓ GraphCypherQAChain đã sẵn sàng.")

    # ──────────────────────────────────────────────────────────
    # PUBLIC: query (gọi từ query pipeline)
    # ──────────────────────────────────────────────────────────
    def query(self, question: str) -> str:
        """
        Truy vấn đồ thị bằng ngôn ngữ tự nhiên.
        Trả về chuỗi câu trả lời hoặc chuỗi rỗng nếu lỗi.
        """
        if self._chain is None:
            self.setup_chain()
        try:
            response = self._chain.invoke({"query": question})

            # # ── In ra Cypher query để debug ────────────────────
            # steps = response.get("intermediate_steps", [])
            # for step in steps:
            #     if isinstance(step, dict) and "query" in step:
            #         cypher = step["query"].strip()
            #         print(f"\n  ┌─ [Graph-RAG] Cypher Query ──────────────────")
            #         print(f"  │  {cypher}")
            #         print(f"  └─────────────────────────────────────────────\n")
            #         break

            return response.get("result", "")
        except Exception as e:
            print(f"  [GraphRAG] ✗ Lỗi truy vấn: {e}")
            return ""

    # ──────────────────────────────────────────────────────────
    # PUBLIC: clear_graph (tiện dùng khi reset)
    # ──────────────────────────────────────────────────────────
    def clear_graph(self):
        """Xóa toàn bộ nodes và relationships trong Neo4j."""
        with self.kg._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("  [GraphRAG] ✓ Đã xóa toàn bộ graph.")
