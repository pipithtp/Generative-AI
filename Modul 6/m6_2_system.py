import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. System Prompt dimasukkan ke dalam daftar messages dengan role "system"
response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical writer. Answer in plain English, no jargon.",
        },
        {
            "role": "user",
            "content": "Explain what a vector database does.",
        },
    ],
)

# 3. Tampilkan hasil
print("--- Response dengan System Prompt ---")
print(response["message"]["content"])