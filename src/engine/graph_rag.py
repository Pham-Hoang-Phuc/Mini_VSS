"""
graph_rag.py
============
Module Graph-RAG: quản lý toàn bộ logic ArcadeDB cho CA-RAG pipeline.

Chức năng:
  - extract_and_store(): gọi Gemini trích xuất entity/relation từ caption,
    MERGE vào ArcadeDB kèm property timestamp.
  - query(): trả lời câu hỏi dựa trên đồ thị (GraphCypherQAChain).
"""

import os
import re
import warnings
import requests
from requests.auth import HTTPBasicAuth
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs.graph_store import GraphStore
from configs.config import Config

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


# ──────────────────────────────────────────────────────────────
# ArcadeDBGraph — wrapper dùng HTTP/JSON API của ArcadeDB
# (thay thế langchain_neo4j.Neo4jGraph bằng HTTP API của ArcadeDB)
# ──────────────────────────────────────────────────────────────
class ArcadeDBGraph(GraphStore):
    """
    Kết nối ArcadeDB qua HTTP/JSON API (port 2480).
    Implement interface tương thích với GraphCypherQAChain:
      - thuộc tính `schema`
      - method `query(cypher)`
      - method `refresh_schema()`
    """

    def __init__(self, host: str, port: str, username: str, password: str, database: str):
        self.base_url = f"http://{host}:{port}"
        self.database = database
        self.auth     = HTTPBasicAuth(username, password)
        self.schema   = ""
        self._ensure_database()
        self._ensure_schema()
        self.refresh_schema()

    # ── Nội bộ ────────────────────────────────────────────────

    def _command(self, command: str, language: str = "sql", params: dict = None) -> list:
        """Gọi POST /api/v1/command/{db} và trả về list records."""
        payload = {"language": language, "command": command}
        if params:
            payload["params"] = params

        resp = requests.post(
            f"{self.base_url}/api/v1/command/{self.database}",
            json=payload,
            auth=self.auth,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", [])

    def _ensure_database(self):
        """Tạo database nếu chưa tồn tại."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/exists/{self.database}",
                auth=self.auth,
                timeout=10,
            )
            if resp.status_code == 200 and resp.json().get("result") is True:
                return
        except Exception:
            pass

        # Tạo database mới
        try:
            requests.post(
                f"{self.base_url}/api/v1/server",
                json={"command": f"create database {self.database}"},
                auth=self.auth,
                timeout=10,
            )
        except Exception as e:
            print(f"  [ArcadeDB] ⚠ Không thể tạo database: {e}")

    def _ensure_schema(self):
        """Tạo vertex type Entity nếu chưa tồn tại."""
        try:
            self._command("CREATE VERTEX TYPE Entity IF NOT EXISTS")
        except Exception:
            pass
        try:
            self._command("CREATE PROPERTY Entity.name IF NOT EXISTS STRING")
        except Exception:
            pass
        # Index unique trên Entity.name
        try:
            self._command("CREATE INDEX IF NOT EXISTS ON Entity (name) UNIQUE")
        except Exception:
            pass

    def _cypher(self, cypher: str, params: dict = None) -> list:
        """Chạy openCypher query."""
        return self._command(cypher, language="cypher", params=params)

    # ── Public interface ───────────────────────────────────────

    def refresh_schema(self):
        """Cập nhật schema string dùng cho LLM prompt."""
        try:
            all_types = self._command("SELECT name, type FROM schema:types")
            v_names = [r["name"] for r in all_types if r.get("type") == "vertex"]
            e_names = [r["name"] for r in all_types if r.get("type") == "edge"]
            self.schema = (
                f"Node properties: {', '.join(v_names) or 'Entity (name: STRING)'}\n"
                f"Relationship types: {', '.join(e_names) or '(dynamic)'}\n"
                f"All nodes are of type Entity with property `name`."
            )
            # Cập nhật structured schema cho GraphCypherQAChain
            self._structured_schema = {
                "node_props": {v: [{"property": "name", "type": "STRING"}] for v in v_names},
                "rel_props":  {e: [] for e in e_names},
                "relationships": []
            }
        except Exception:
            self.schema = (
                "Node properties: Entity (name: STRING)\n"
                "Relationship types: (dynamic)\n"
                "All nodes are of type Entity with property `name`."
            )
            self._structured_schema = {
                "node_props": {"Entity": [{"property": "name", "type": "STRING"}]},
                "rel_props":  {},
                "relationships": []
            }

    @property
    def get_schema(self) -> str:
        """Trả về schema string — bắt buộc cho GraphStore."""
        return self.schema

    @property
    def get_structured_schema(self) -> dict:
        """Trả về structured schema — bắt buộc cho GraphCypherQAChain."""
        return self._structured_schema

    def add_graph_documents(self, graph_documents, *args, **kwargs):
        """Stub — không dùng, ingestion được xử lý riêng trong GraphRAGManager."""
        pass

    def query(self, cypher: str) -> list:
        """Chạy openCypher query và trả về list records (dict)."""
        return self._cypher(cypher)



# ──────────────────────────────────────────────────────────────
# GraphRAGManager
# ──────────────────────────────────────────────────────────────
class GraphRAGManager:
    """
    Quản lý Graph-RAG: kết nối ArcadeDB, trích xuất entity/relation từ caption,
    và truy vấn đồ thị bằng ngôn ngữ tự nhiên.
    """

    def __init__(self):
        self.kg = ArcadeDBGraph(
            host=Config.ARCADEDB_HOST,
            port=Config.ARCADEDB_PORT,
            username=Config.ARCADEDB_USERNAME,
            password=Config.ARCADEDB_PASSWORD,
            database=Config.ARCADEDB_DATABASE,
        )
        self._chain = None  # lazy init sau khi build xong graph
        print("  [GraphRAG] ✓ Kết nối ArcadeDB thành công.")

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

    # ────────────────────────────────────────────────────────
    # 3. MERGE vào ArcadeDB (dùng openCypher)
    # ────────────────────────────────────────────────────────
    def _merge_to_arcadedb(self, relationships: list):
        """MERGE nodes và relationships vào ArcadeDB qua openCypher MERGE."""
        for subject, relation, obj in relationships:
            cypher = f"""
            MERGE (a:Entity {{name: $subject}})
            MERGE (b:Entity {{name: $obj}})
            MERGE (a)-[:`{relation}`]->(b)
            """
            try:
                self.kg._cypher(cypher, params={"subject": subject, "obj": obj})
            except Exception as e:
                print(f"    [GraphRAG] ⚠ Lỗi khi thêm ({subject})-[{relation}]->({obj}): {e}")

    # ──────────────────────────────────────────────────────────
    # PUBLIC: extract_and_store (gọi từ ingestion pipeline)
    # ──────────────────────────────────────────────────────────
    def extract_and_store(self, caption: str):
        """
        Trích xuất entity/relation từ caption và MERGE vào ArcadeDB.
        ArcadeDB chỉ lưu nội dung (Entity + Relationship), không lưu metadata.
        Trả về số lượng relationship được thêm (0 nếu lỗi).
        """
        try:
            raw = self._extract_entities_and_relationships(caption)
            entity_dict, rel_list = self._parse_llm_output(raw)

            if rel_list:
                self._merge_to_arcadedb(rel_list)
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
            return response.get("result", "")
        except Exception as e:
            print(f"  [GraphRAG] ✗ Lỗi truy vấn: {e}")
            return ""

    # ──────────────────────────────────────────────────────────
    # PUBLIC: clear_graph (tiện dùng khi reset)
    # ──────────────────────────────────────────────────────────
    def clear_graph(self):
        """Xóa toàn bộ vertices và edges trong ArcadeDB."""
        try:
            self.kg._command("DELETE FROM Entity")
        except Exception:
            pass
        # Xoá tất cả edge types động
        try:
            edge_types = self.kg._command(
                "SELECT name FROM schema:types WHERE type = 'EDGE'"
            )
            for et in edge_types:
                name = et.get("name", "")
                if name:
                    try:
                        self.kg._command(f"DELETE FROM `{name}`")
                    except Exception:
                        pass
        except Exception:
            pass
        print("  [GraphRAG] ✓ Đã xóa toàn bộ graph.")
