import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. Basic Message Call (Pengganti client.messages.create)
response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "What is retrieval-augmented generation?",
        }
    ],
)

# 3. Tampilkan teks balasan (Di Ollama: response['message']['content'])
print("--- Response Text ---")
print(response["message"]["content"])

# 4. Statistik Penggunaan Token (Usage Stats)
print("\n--- Usage Stats ---")
print(f"Prompt tokens (Input): {response.get('prompt_eval_count', 'N/A')}")
print(f"Completion tokens (Output): {response.get('eval_count', 'N/A')}")