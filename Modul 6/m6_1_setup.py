import os
from dotenv import load_dotenv

# Load file .env
load_dotenv()

# Ambil variabel untuk Ollama (Aman dari KeyError)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

print("--- Modul 6.1 Setup ---")
print(f"Ollama Host  : {OLLAMA_HOST}")
print(f"Ollama Model : {OLLAMA_MODEL}")