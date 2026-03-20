"""
arcadedb_store.py
=================
ArcadeDB connection layer — wrapper HTTP/JSON API.

Tách từ graph_rag.py: chỉ chứa database connection logic,
không chứa business logic (entity extraction, query chain).
"""

import requests
from requests.auth import HTTPBasicAuth
from langchain_community.graphs.graph_store import GraphStore
from configs.config import Config


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
        """Stub — không dùng, ingestion được xử lý riêng trong EntityExtractor."""
        pass

    def query(self, cypher: str) -> list:
        """Chạy openCypher query và trả về list records (dict)."""
        return self._cypher(cypher)

    def clear_all(self):
        """Xóa toàn bộ vertices và edges trong ArcadeDB."""
        try:
            self._command("DELETE FROM Entity")
        except Exception:
            pass
        # Xoá tất cả edge types động
        try:
            edge_types = self._command(
                "SELECT name FROM schema:types WHERE type = 'EDGE'"
            )
            for et in edge_types:
                name = et.get("name", "")
                if name:
                    try:
                        self._command(f"DELETE FROM `{name}`")
                    except Exception:
                        pass
        except Exception:
            pass
        print("  [ArcadeDB] ✓ Đã xóa toàn bộ graph.")
