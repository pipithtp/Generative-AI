import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

print(f"--- Streaming Response (Model: {MODEL_NAME}) ---\n")

# 2. Panggil ollama.chat dengan parameter stream=True
stream = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "List 5 use cases for vector databases.",
        }
    ],
    stream=True,
)

# 3. Looping untuk mencetak potongan teks (chunk) secara real-time
prompt_tokens = 0
eval_tokens = 0

for chunk in stream:
    # Tampilkan teks per kata/karakter yang baru tiba
    print(chunk["message"]["content"], end="", flush=True)

    # Ambil statistik token dari chunk terakhir jika sudah selesai (done == True)
    if chunk.get("done"):
        prompt_tokens = chunk.get("prompt_eval_count", 0)
        eval_tokens = chunk.get("eval_count", 0)

print()  # Newline setelah streaming selesai

# 4. Tampilkan total penggunaan token
total_tokens = prompt_tokens + eval_tokens
print(f"\nTotal tokens: {total_tokens} (Input: {prompt_tokens}, Output: {eval_tokens})")