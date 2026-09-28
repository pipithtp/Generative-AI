import os
import json
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. Definisikan Fungsi Python Lokal
def get_model_info(model_name: str) -> dict:
    """Fungsi eksekusi lokal yang menyimpan database informasi model."""
    db = {
        "qwen2.5:3b": {"context_k": 32, "cost_input": 0.00, "cost_output": 0.00},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00, "cost_output": 15.00},
        "gpt-4o": {"context_k": 128, "cost_input": 2.50, "cost_output": 10.00},
        "gemini-1.5-pro": {"context_k": 1000, "cost_input": 1.25, "cost_output": 5.00},
    }
    return db.get(model_name.lower(), {"error": f"Unknown model: {model_name}"})

# 3. Definisikan Tool Schema (Format OpenAI/Ollama)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window size and cost per 1M tokens for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "The model identifier, e.g. 'qwen2.5:3b' or 'gpt-4o'.",
                    }
                },
                "required": ["model_name"],
            },
        },
    }
]

# 4. Request Pertama ke Ollama
messages = [
    {"role": "user", "content": "How large is the context window of qwen2.5:3b?"}
]

print("--- Step 1: Sending request with tools ---")
response = ollama.chat(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
)

# 5. Cek apakah model meminta pemanggilan tool (tool_calls)
if response["message"].get("tool_calls"):
    # Ambil pemanggilan tool pertama
    tool_call = response["message"]["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = tool_call["function"]["arguments"]

    print(f"\nModel meminta Tool: {tool_name}")
    print(f"Argumen dari Model : {tool_args}")

    # Eksekusi fungsi Python secara lokal
    if tool_name == "get_model_info":
        tool_result = get_model_info(**tool_args)
        print(f"Hasil Eksekusi Tool : {tool_result}\n")

        # 6. Masukkan balasan pesan assistant & hasil tool kembali ke daftar messages
        messages.append(response["message"])
        messages.append({
            "role": "tool",
            "content": json.dumps(tool_result),
        })

        # 7. Request Kedua ke Ollama untuk mendapatkan jawaban akhir
        final_response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
        )

        print("--- Final Answer ---")
        print(final_response["message"]["content"])
else:
    print("Model tidak memanggil tool, langsung menjawab:")
    print(response["message"]["content"])