import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# 2. Inisialisasi OpenAI client yang mengarah ke server Ollama lokal
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 3. Schema Tools (Format standar OpenAI Function Calling)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"]
            }
        }
    }
]

# 4. Fungsi eksekusi Python lokal
def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {"context_k": 128, "cost_input": 2.50},
        "qwen2.5:3b": {"context_k": 32, "cost_input": 0.00},
        "claude-sonnet-4-5": {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name.lower(), {"error": "unknown model"})

# 5. Percakapan Awal
messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]

# Request pertama ke LLM
response = client.chat.completions.create(
    model=MODEL_NAME,
    tools=tools,
    messages=messages
)

response_message = response.choices[0].message

# 6. Cek apakah finish_reason adalah "tool_calls" (atau jika ada tool_calls pada message)
if response.choices[0].finish_reason == "tool_calls" or response_message.tool_calls:
    tool_call = response_message.tool_calls[0]
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"Tool dipanggil : {name}")
    print(f"Argumen        : {args}")
    
    # Eksekusi fungsi lokal
    result = get_model_info(**args)
    print(f"Hasil Eksekusi : {result}\n")

    # Tambahkan pesan assistant dan hasil tool ke riwayat percakapan
    messages.append(response_message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Request kedua ke LLM untuk mendapatkan jawaban akhir
    final = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=messages
    )
    
    print("--- Final Answer ---")
    print(final.choices[0].message.content)
else:
    print("Model langsung menjawab tanpa memanggil tool:")
    print(response_message.content)