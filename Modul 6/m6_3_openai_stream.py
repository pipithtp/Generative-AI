import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. Inisialisasi OpenAI Client mengarah ke Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

print(f"--- Streaming OpenAI SDK (Model: {MODEL_NAME}) ---\n")

# 3. Request streaming chat completion
stream = client.chat.completions.create(
    model=MODEL_NAME,
    max_tokens=512,
    stream=True,
    messages=[
        {"role": "user", "content": "Explain embeddings in 3 bullet points."}
    ]
)

# 4. Looping chunk delta per chunk
for chunk in stream:
    # Memeriksa delta content pada pilihan pertama
    if chunk.choices and chunk.choices[0].delta.content:
        delta = chunk.choices[0].delta.content
        print(delta, end="", flush=True)

print()  # Newline setelah streaming selesai