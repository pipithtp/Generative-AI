import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# Strong system prompt - explicit role, rules, format
STRONG_SYSTEM = """You are a senior Python engineer reviewing code for a production AI pipeline.

Your job:
- Identify bugs, security issues, and performance problems
- Suggest concrete improvements with code examples
- Explain WHY each issue matters

Rules:
- Be direct. Do not pad with compliments.
- If code is correct, say so briefly and move on.
- Always include the corrected code when suggesting a fix.

Format:
Return your review as a numbered list. Each item: Issue → Impact → Fix."""

# Code snippet yang akan direview
code_to_review = """def get_user(user_id):
    key = os.getenv('DB_KEY')
    result = requests.get(f'http://db/{user_id}?key={key}')
    return result.json()"""

# 2. Susun messages dengan System Prompt di role "system"
messages = [
    {"role": "system", "content": STRONG_SYSTEM},
    {"role": "user", "content": f"Review this function:\n{code_to_review}"}
]

# 3. Jalankan pemanggilan ke Ollama
print(f"--- Modul 7.2 System Prompts Demo (Model: {MODEL_NAME}) ---\n")
response = ollama.chat(
    model=MODEL_NAME,
    messages=messages
)

# 4. Tampilkan hasil review
print(response["message"]["content"])