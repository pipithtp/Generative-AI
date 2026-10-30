import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Memuat environment variables dari .env
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print("Menyiapkan LlamaIndex Settings...")
print(f"LLM       : {ollama_model} ({ollama_host})")
print("Embedding : BAAI/bge-small-en-v1.5 (FastEmbed - CPU/Lokal)\n")

# 1. Konfigurasi LLM global menggunakan Ollama
Settings.llm = Ollama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.1,
    request_timeout=120.0,
)

# 2. Konfigurasi Embedding global menggunakan FastEmbed (aman dan ringan)
Settings.embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# 3. Konfigurasi pemotongan token (Chunking)
Settings.chunk_size = 512       # tokens per chunk
Settings.chunk_overlap = 64     # token overlap antar chunk

# Verifikasi koneksi LLM
print("Memverifikasi koneksi LLM...")
response = Settings.llm.complete("In one sentence, what is LlamaIndex?")
print(f"Respon Uji Coba:\n{response.text}\n")
print(f"Pengaturan selesai. Chunk size: {Settings.chunk_size}, Chunk overlap: {Settings.chunk_overlap}")