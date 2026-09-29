import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. Inisialisasi OpenAI Client yang mengarah ke Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

print(f"--- Modul 7.5 OpenAI JSON Mode Demo (Model: {MODEL_NAME}) ---\n")

# 3. Request ke API dengan response_format={"type": "json_object"}
response = client.chat.completions.create(
    model=MODEL_NAME,
    response_format={"type": "json_object"},  # Memaksa output JSON di tingkat API
    messages=[
        {
            "role": "system",
            "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}"""
        },
        {
            "role": "user",
            "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla."
        }
    ]
)

# 4. Parsing string balasan menjadi Dictionary Python
raw_content = response.choices[0].message.content
result = json.loads(raw_content)

print("Hasil Ekstraksi JSON:")
print(json.dumps(result, indent=2))