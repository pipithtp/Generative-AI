import os
import re
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# System prompt yang mewajibkan format tag XML
SYSTEM_PROMPT = """Solve problems using this exact format:
<thinking>
Step-by-step reasoning here.
</thinking>
<answer>
The final answer only, no reasoning.
</answer>"""

user_query = "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 tokens. The model has a 4096 token limit for context. How many tokens remain for the response?"

print(f"--- Modul 7.4 Structured CoT Demo (Model: {MODEL_NAME}) ---\n")

# 2. Kirim request ke Ollama
response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

raw_text = response["message"]["content"]

# 3. Parsing XML Tag menggunakan Regular Expression (re)
thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_text, re.DOTALL)
answer_match = re.search(r"<answer>(.*?)</answer>", raw_text, re.DOTALL)

thinking_text = thinking_match.group(1).strip() if thinking_match else "not found"
answer_text = answer_match.group(1).strip() if answer_match else "not found"

# 4. Tampilkan hasil ekstraksi terpisah
print("--- Raw Output ---")
print(raw_text)

print("\n--- Parsed Results ---")
print("Reasoning :", thinking_text)
print("Answer    :", answer_text)