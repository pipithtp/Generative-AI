import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(
    f"Menyiapkan VectorStoreIndex dengan model: {ollama_model}"
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

# 2. Dokumen pengetahuan
documents = [
    Document(
        text=(
            "RAG combines retrieval with generation to answer questions from"
            " external knowledge."
        )
    ),
    Document(
        text=(
            "Fine-tuning trains a model on new data to improve performance on"
            " a specific task."
        )
    ),
    Document(
        text=(
            "Prompt engineering optimises the input to an LLM to get better"
            " outputs without any training."
        )
    ),
    Document(
        text=(
            "Vector databases store high-dimensional embeddings and support"
            " fast approximate nearest-neighbour search."
        )
    ),
    Document(
        text=(
            "LangChain provides abstractions for building chains, agents, and"
            " memory-enabled LLM applications."
        )
    ),
]

# 3. Build index - meng-embed semua dokumen secara otomatis
print("Membangun VectorStoreIndex...")
index = VectorStoreIndex.from_documents(documents)

# 4. Query engine - mencari k dokumen terdekat (top-k) lalu mensintesis jawaban
query_engine = index.as_query_engine(similarity_top_k=3)

pertanyaan = "What is the difference between RAG and fine-tuning?"
print(f"Pertanyaan: {pertanyaan}\n")
response = query_engine.query(pertanyaan)

print("--- Jawaban ---")
print(response)

print("\n--- Source Nodes (Konteks Terpilih) ---")
for node in response.source_nodes:
  print(f"Score {node.score:.4f}: {node.text[:80]}...")