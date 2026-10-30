import os
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(
    f"Menyiapkan Persistent Storage dengan model: {ollama_model}"
    f" ({ollama_host})\n"
)

# 1. Konfigurasi LLM dan Embedding global
Settings.llm = Ollama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.2,
    request_timeout=120.0,
)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Direktori penyimpanan lokal di Windows
PERSIST_DIR = "./storage"

# 2. Bangun dan simpan index jika folder penyimpanan belum ada
if not os.path.exists(PERSIST_DIR):
  print(f"Index belum ada. Membangun dan menyimpan ke '{PERSIST_DIR}'...")
  documents = [
      Document(
          text="LlamaIndex persists indexes to disk as JSON files."
      ),
      Document(
          text=(
              "Loading a persisted index avoids re-embedding on every run."
          )
      ),
  ]
  index = VectorStoreIndex.from_documents(documents)
  index.storage_context.persist(persist_dir=PERSIST_DIR)
  print(f"Index berhasil disimpan ke folder '{PERSIST_DIR}'.\n")
else:
  print(
      f"Folder '{PERSIST_DIR}' sudah ditemukan. Memuat index dari disk..."
  )

# 3. Muat kembali index dari disk (tanpa perlu re-embedding)
print("Memuat index dari storage_context...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
loaded_index = load_index_from_storage(storage_context)

# 4. Buat query engine dari index yang dimuat
query_engine = loaded_index.as_query_engine()

pertanyaan = "How does LlamaIndex handle persistence?"
print(f"Pertanyaan: {pertanyaan}\n")
response = query_engine.query(pertanyaan)

print("--- Jawaban (After Reload) ---")
print(response)