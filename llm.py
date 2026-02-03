from google import genai
from config import Config

class GeminiManager:
    def __init__(self):
        self.client = genai.Client(
            api_key=Config.GEMINI_API_KEY,
            http_options={'api_version': 'v1'}
        )

    def generate_answer(self, query, context):
        prompt = f"""
            Bạn là một trợ lý AI thông minh cho hệ thống tìm kiếm video.
            Dưới đây là các thông tin trích xuất được từ video dựa trên câu hỏi của người dùng.

            Dữ liệu trích xuất:
            {context}

            Câu hỏi của người dùng: "{query}"

            Hãy trả lời câu hỏi trên một cách tự nhiên, lịch sự dựa trên dữ liệu trích xuất. 
            Nếu không thấy thông tin phù hợp, hãy báo là không tìm thấy.
            Nêu rõ mốc thời gian (timestamp).
            """
        response = self.client.models.generate_content(
            model=Config.LLM_MODEL_NAME,
            contents=prompt
        )
        return response.text