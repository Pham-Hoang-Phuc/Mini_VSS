"""
graph_rag_complete.py
=====================
Pipeline hoàn chỉnh:
  1. Đọc dữ liệu thô (CSV hoặc dùng data mẫu)
  2. Dùng Gemini (OpenAI-compatible endpoint) để trích xuất Entity & Relationship từ văn bản
  3. Đẩy Nodes + Relationships vào Neo4j
  4. Truy vấn đồ thị bằng LangChain GraphCypherQAChain + Gemini

Giữ nguyên các patterns từ hehe.py:
  - extract_entities_and_relationships()  → gọi Gemini qua openai client
  - process_llm_out()                     → parse output LLM thành list
  - add_relationships_to_neo4j()          → MERGE vào Neo4j
  - CYPHER_GENERATION_TEMPLATE            → prompt ví dụ tiếng Việt
  - prettyCypherChain()                   → wrap GraphCypherQAChain
"""

# ============================================================
# 0. IMPORTS & CONFIG
# ============================================================
import os
import re
import textwrap
import warnings
import pandas as pd
from dotenv import load_dotenv

# Gemini qua OpenAI-compatible API
from openai import OpenAI

# LangChain + Neo4j
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.prompt import PromptTemplate

# Warning control
warnings.filterwarnings("ignore")

# Load biến môi trường từ .env
load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
NEO4J_URI       = os.getenv("NEO4J_URI")
NEO4J_USERNAME  = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE", "neo4j")

# Map GEMINI_API_KEY → GOOGLE_API_KEY (cần cho langchain_google_genai)
os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY or "")


# ============================================================
# 1. KẾT NỐI NEO4J
# ============================================================
# kg object dùng xuyên suốt: vừa để MERGE nodes/relationships (qua driver),
# vừa được truyền vào GraphCypherQAChain (qua LangChain Neo4jGraph)
kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)


# ============================================================
# 2. DỮ LIỆU ĐẦU VÀO
# ============================================================
# Thay đường dẫn CSV thực tế nếu có; nếu không có thì dùng data mẫu bên dưới.
CSV_PATH = "data.csv"   # ← đổi thành đường dẫn file CSV của bạn

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Đọc file CSV hoặc trả về DataFrame mẫu nếu file không tồn tại."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Đảm bảo có cột 'information'
        if "information" not in df.columns:
            raise ValueError(f"File CSV '{csv_path}' phải có cột 'information'.")
        print(f"Đã tải {len(df)} dòng từ '{csv_path}'.")
    else:
        print(f"Không tìm thấy '{csv_path}', sử dụng data mẫu điện thoại.")
        sample_data = [
            {"caption": "Công nhân bắt đầu vào ca, mặc đồ bảo hộ và kiểm tra danh sách chấm công tại cửa ra vào", "timestamp": "00:00-00:30"},
            {"caption": "Xe nâng (forklift) vận chuyển các kiện hàng nguyên liệu từ kho vào khu vực sản xuất chính", "timestamp": "00:31-01:15"},
            {"caption": "Nhóm kỹ thuật viên đang tập trung quanh máy CNC để hiệu chỉnh thông số kỹ thuật", "timestamp": "01:16-02:00"},
            {"caption": "Dây chuyền lắp ráp hoạt động ổn định, các công nhân thao tác gắn chip lên bảng mạch", "timestamp": "02:01-03:00"},
            {"caption": "Nhân viên quản lý chất lượng (QC) kiểm tra ngẫu nhiên các sản phẩm trên băng chuyền bằng kính hiển vi", "timestamp": "03:01-03:45"},
            {"caption": "Một công nhân đang dọn dẹp khu vực đóng gói và dán nhãn lên các thùng carton thành phẩm", "timestamp": "03:46-04:30"},
            {"caption": "Đội bảo trì thực hiện kiểm tra định kỳ các đường ống khí nén dọc hành lang xưởng", "timestamp": "04:31-05:00"}
        ]
        df = pd.DataFrame(sample_data)
        print(f"Sử dụng {len(df)} dòng data mẫu.")
    return df


# ============================================================
# 3. TRÍCH XUẤT ENTITY & RELATIONSHIP (giữ nguyên pattern hehe.py)
# ============================================================
def extract_entities_and_relationships(text: str) -> str:
    """
    Gửi văn bản tới Gemini (qua OpenAI-compatible endpoint),
    trả về string chứa Entities và Relationships.
    """
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=GEMINI_BASE_URL,
    )

    prompt = (
        f"Extract entities (nodes) and their relationships (edges) from the text below."
        f"Entities and relationships MUST be in Vietnamese\n"
        f"Follow this format:\n\n"
        f"Entities:\n"
        f"- {{Entity}}: {{Type}}\n\n"
        f"Relationships:\n"
        f"- ({{Entity1}}, {{RelationshipType}}, {{Entity2}})\n\n"
        f"Text:\n\"{text}\"\n\n"
        f"Output:\nEntities:\n- {{Entity}}: {{Type}}\n...\n\n"
        f"Relationships:\n- ({{Entity1}}, {{RelationshipType}}, {{Entity2}})\n"
    )

    response = client.chat.completions.create(
        model="gemma-3-27b-it",   # ← model Gemini, đổi tùy ý
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            },
        ],
        response_format={"type": "text"},
        temperature=1,
        max_tokens=2048,
        top_p=1,
    )

    return response.choices[0].message.content


# ============================================================
# 4. PARSE OUTPUT LLM (giữ nguyên pattern hehe.py)
# ============================================================
def process_llm_out(result: str):
    """
    Parse chuỗi output của Gemini thành:
      - entity_list: list tên entities
      - relationship_list: list (subject, RELATION_TYPE, object)
    """
    response = result

    # Extract entities
    entity_pattern = r"- (.+): (.+)"
    entities = re.findall(entity_pattern, response)
    entity_dict = {entity.strip(): entity_type.strip() for entity, entity_type in entities}
    entity_list = list(entity_dict.keys())

    # Extract relationships
    relationship_pattern = r"- \(([^,]+), ([^,]+), ([^)]+)\)"
    relationships = re.findall(relationship_pattern, response)
    relationship_list = [
        (subject.strip(), relation.strip().replace(" ", "_").upper(), object_.strip())
        for subject, relation, object_ in relationships
    ]

    # In ra để debug
    print("Entities:")
    for entity, entity_type in entity_dict.items():
        print(f"  {entity}: {entity_type}")

    print("\nRelationships:")
    for subject, relation, object_ in relationship_list:
        print(f"  ({subject}, {relation}, {object_})")

    return entity_list, relationship_list


# ============================================================
# 5. ĐẨY VÀO NEO4J (giữ nguyên pattern hehe.py)
# ============================================================
def add_relationships_to_neo4j(graph: Neo4jGraph, relationships: list):
    """
    MERGE nodes và relationships vào Neo4j.
    Dùng graph._driver.session() để chạy Cypher trực tiếp.
    """
    with graph._driver.session() as session:
        for subject, relation, obj in relationships:
            cypher_query = f"""
            MERGE (a:Entity {{name: $subject}})
            MERGE (b:Entity {{name: $object}})
            MERGE (a)-[:`{relation}`]->(b)
            """
            session.run(cypher_query, subject=subject, object=obj)
    print("  → Relationships added to Neo4j.")


# ============================================================
# 6. VÒNG LẶP XỬ LÝ DỮ LIỆU CHÍNH
# ============================================================
def build_knowledge_graph(df: pd.DataFrame, num_rows: int = 30):
    """
    Duyệt qua các hàng trong DataFrame, trích xuất entities/relationships
    và đẩy vào Neo4j.
    """
    n = min(num_rows, len(df))
    print(f"\n{'='*50}")
    print(f"Bắt đầu xây dựng Knowledge Graph từ {n} dòng dữ liệu...")
    print(f"{'='*50}\n")

    for i in range(n):
        sample = df.iloc[i]["caption"]
        print(f"\n[{i+1}/{n}] Đang xử lý: {sample[:60]}...")
        try:
            result = extract_entities_and_relationships(sample)
            _, relationship_list = process_llm_out(result)
            if relationship_list:
                add_relationships_to_neo4j(kg, relationship_list)
            else:
                print("  → Không tìm thấy relationship nào.")
        except Exception as e:
            print(f"  → Lỗi khi xử lý dòng {i}: {e}")

    print(f"\n{'='*50}")
    print("Hoàn tất xây dựng Knowledge Graph!")
    print(f"{'='*50}\n")


# ============================================================
# 7. CYPHER GENERATION TEMPLATE (giữ nguyên từ hehe.py, bổ sung ví dụ)
# ============================================================
CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
- Analyze the question and extract relevant graph components dynamically. Use this to construct the Cypher query.
- Use only the relationship types and properties from the provided schema. Do not include any other relationship types or properties.
- The schema is based on a graph structure with nodes and relationships as follows:
{schema}
- Return only the generated Cypher query in your response. Do not include explanations, comments, or additional text.
- Ensure the Cypher query directly addresses the given question using the schema accurately.

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

# Máy CNC được sử dụng như thế nào trong video?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(b.name) CONTAINS 'cnc' OR toLower(a.name) CONTAINS 'cnc'
RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS may_moc

# Nhân viên QC kiểm tra gì trong video?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'qc' OR toLower(a.name) CONTAINS 'chất lượng'
RETURN a.name AS nhan_vien, type(r) AS hanh_dong, b.name AS doi_tuong

# Liệt kê tất cả các hoạt động liên quan đến dây chuyền lắp ráp?
MATCH (a:Entity)-[r]->(b:Entity)
    WHERE toLower(a.name) CONTAINS 'dây chuyền' OR toLower(b.name) CONTAINS 'lắp ráp'
RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS doi_tuong

The question is:
{question}
"""


# ============================================================
# 8. THIẾT LẬP LANGCHAIN QUERYING (giữ nguyên pattern hehe.py)
# ============================================================
def setup_cypher_chain() -> GraphCypherQAChain:
    """Khởi tạo GraphCypherQAChain với Gemini và Neo4jGraph."""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE,
    )

    # Dùng 2 model riêng: 1 để sinh Cypher (temperature=0), 1 để tổng hợp câu trả lời
    cypher_llm = ChatGoogleGenerativeAI(
        model="gemma-3-12b-it",
        temperature=0,
    )
    qa_llm = ChatGoogleGenerativeAI(
        model="gemma-3-1b-it",
        temperature=0.3,
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm=qa_llm,
        cypher_llm=cypher_llm,
        graph=kg,
        verbose=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        allow_dangerous_requests=True,
    )

    return cypher_chain


def prettyCypherChain(chain: GraphCypherQAChain, question: str) -> str:
    """Chạy chain và in kết quả gọn gàng."""
    print(f"\n{'─'*50}")
    print(f"Câu hỏi: {question}")
    print(f"{'─'*50}")
    try:
        response = chain.invoke({"query": question})
        answer = response.get("result", str(response))
        print("\nTrả lời:")
        print(textwrap.fill(answer, 60))
    except Exception as e:
        answer = f"Lỗi: {e}"
        print(answer)
    print(f"{'─'*50}\n")
    return answer


# ============================================================
# 9. MAIN - CHẠY TOÀN BỘ PIPELINE
# ============================================================
if __name__ == "__main__":

    # --- BƯỚC 1: Tải dữ liệu ---
    df = load_dataframe(CSV_PATH)

    # --- BƯỚC 2: Xây dựng Knowledge Graph ---
    # Đổi num_rows=len(df) để xử lý toàn bộ, hoặc để nhỏ để test nhanh
    build_knowledge_graph(df, num_rows=8)

    # Refresh schema sau khi insert (để LangChain biết schema mới)
    kg.refresh_schema()

    # --- BƯỚC 3: Thiết lập Chain truy vấn ---
    print("\nĐang khởi tạo chuỗi truy vấn GraphCypherQAChain...")
    cypher_chain = setup_cypher_chain()

    # --- BƯỚC 4: Chạy thử các câu hỏi ---

    prettyCypherChain(cypher_chain, input("Nhập câu hỏi"))

    # test_questions = [
    #     "Xe nâng xuất hiện trong video lúc nào và đang làm gì?",
    #     "Công nhân bắt đầu vào ca lúc mấy giờ và làm những gì?",
    #     "Đội bảo trì đã kiểm tra những thiết bị nào?",
    #     "Nhân viên QC thực hiện công việc gì trong video?",
    # ]


    # for question in test_questions:
    #     prettyCypherChain(cypher_chain, question)

    # # --- (Tuỳ chọn) Cypher thủ công để kiểm tra trực tiếp ---
    # print("\n=== Truy vấn Cypher thủ công ===")
    # results = kg.query(
    #     """MATCH (a:Entity)-[r]->(b:Entity)
    #     RETURN a.name AS chu_the, type(r) AS hanh_dong, b.name AS doi_tuong
    #     LIMIT 20
    #     """
    # )
    # if results:
    #     for row in results:
    #         print(row)
    # else:
    #     print("Không tìm thấy kết quả.")

    # # --- (Tuỳ chọn) Chế độ hỏi đáp tương tác ---
    # print("\n=== Chế độ hỏi đáp tương tác (gõ 'quit' để thoát) ===")
    # while True:
    #     user_input = input("Câu hỏi của bạn: ").strip()
    #     if not user_input or user_input.lower() in ("quit", "exit", "thoát"):
    #         break
    #     prettyCypherChain(cypher_chain, user_input)

    print("\nĐã hoàn tất. Tạm biệt!")
