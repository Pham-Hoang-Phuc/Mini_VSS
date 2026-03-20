"""
entity_extractor.py
===================
Module trích xuất Entity & Relationship từ caption (VLM output).

Pipeline (theo vss_context.md - Mục 3, Bước 3):
  - Input: Structured Caption (từ VLM)
  - Sử dụng LLM để bóc tách thành các cặp (Subject) - [Action] -> (Object)
  - MERGE vào ArcadeDB

Logic được tách từ graph_rag.py (extraction + parse + merge).
"""

import re
from openai import OpenAI
from configs.config import Config
from src.storage.arcadedb_store import ArcadeDBGraph


class EntityExtractor:
    """
    Trích xuất entity/relation từ caption và MERGE vào ArcadeDB.
    """

    def __init__(self, graph: ArcadeDBGraph):
        self.graph = graph

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
                self.graph._cypher(cypher, params={"subject": subject, "obj": obj})
            except Exception as e:
                print(f"    [EntityExtractor] ⚠ Lỗi khi thêm ({subject})-[{relation}]->({obj}): {e}")

    # ──────────────────────────────────────────────────────────
    # PUBLIC: extract_and_store
    # ──────────────────────────────────────────────────────────
    def extract_and_store(self, caption: str) -> int:
        """
        Trích xuất entity/relation từ caption và MERGE vào ArcadeDB.
        Trả về số lượng relationship được thêm (0 nếu lỗi).
        """
        try:
            raw = self._extract_entities_and_relationships(caption)
            entity_dict, rel_list = self._parse_llm_output(raw)

            if rel_list:
                self._merge_to_arcadedb(rel_list)
                print(f"    [EntityExtractor] +{len(rel_list)} relationships | {len(entity_dict)} entities")
            else:
                print(f"    [EntityExtractor] ⚠ Không tìm thấy relationship trong chunk này.")
            return len(rel_list)
        except Exception as e:
            print(f"    [EntityExtractor] ✗ Lỗi: {e}")
            return 0
