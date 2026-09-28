import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. Inisialisasi OpenAI Client dengan mengarahkan ke Ollama lokal
# (api_key diisi string bebas/placeholder karena Ollama tidak butuh key)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 3. Pemanggilan API persis seperti struktur SDK OpenAI
response = client.chat.completions.create(
    model=MODEL_NAME,
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "What is the difference between RAG and fine-tuning?"}
    ]
)

# 4. Tampilkan hasil dan statistik penggunaan token
print("--- Response Text ---")
print(response.choices[0].message.content)

print("\n--- Token Usage ---")
if response.usage:
    print(f"Tokens used: {response.usage.total_tokens}")