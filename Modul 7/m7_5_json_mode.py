import os
import json
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": "string",
  "founded": integer or null,
  "products": ["string"],
  "headquarters": "string or null",
  "is_public": boolean
}"""

texts = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San Francisco, it remains private despite a major Microsoft investment.",
]


def extract_company_info(text: str) -> dict:
    # Menggunakan parameter format="json" bawaan Ollama
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        format="json",  # Memaksa Ollama mengembalikan output JSON yang valid
    )

    raw = resp["message"]["content"].strip()

    # Membersihkan pagar markdown jika ada
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    return json.loads(raw)


if __name__ == "__main__":
    print(f"--- Modul 7.5 Prompt-enforced JSON Demo (Model: {MODEL_NAME}) ---\n")

    for text in texts:
        try:
            info = extract_company_info(text)
            print("Hasil JSON Parsed:")
            print(json.dumps(info, indent=2))
            print("-" * 40)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Gagal melakukan parsing JSON: {e}")