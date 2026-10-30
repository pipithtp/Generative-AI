import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print("Menyiapkan VectorIndexRetriever...")
print("Embedding: BAAI/bge-small-en-v1.5 (FastEmbed - CPU/Lokal)\n")

# Konfigurasi LLM dan Embedding
Settings.llm = Ollama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.2,
    request_timeout=120.0,
)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Dokumen pengetahuan beserta metadata tag
documents = [
    Document(
        text=(
            "ChromaDB is an open-source embedding database with a simple Python"
            " API."
        ),
        metadata={"tag": "vector-db"},
    ),
    Document(
        text=(
            "Pinecone is a managed vector database designed for production"
            " workloads."
        ),
        metadata={"tag": "vector-db"},
    ),
    Document(
        text=(
            "FAISS is a library by Meta for efficient similarity search on"
            " dense vectors."
        ),
        metadata={"tag": "vector-search"},
    ),
    Document(
        text="pgvector adds vector similarity search to PostgreSQL.",
        metadata={"tag": "vector-db"},
    ),
    Document(
        text=(
            "Weaviate is an open-source vector database with a GraphQL API."
        ),
        metadata={"tag": "vector-db"},
    ),
]

# 1. Bangun index dari dokumen
index = VectorStoreIndex.from_documents(documents)

# 2. Inisialisasi Retriever mandiri (tanpa synthesis LLM)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

# 3. Retrieve node murni berdasarkan semantic similarity
query = "open source vector databases"
print(f"Query: '{query}'\n")
nodes = retriever.retrieve(query)

print("--- Hasil Retrieved Nodes (Top 3) ---")
for i, node in enumerate(nodes, 1):
  print(f"[{i}] Score: {node.score:.4f}")
  print(f"    Teks: {node.text}")
  print(f"    Tag : {node.metadata.get('tag')}\n")