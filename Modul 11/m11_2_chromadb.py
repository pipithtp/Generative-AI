import os
import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

# Menyiapkan model embedding lokal CPU-friendly (FastEmbed)
print("Memuat FastEmbed (BAAI/bge-small-en-v1.5)...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


# --- Embedding helper menggunakan FastEmbed lokal ---
def embed(texts: list[str]) -> list[list[float]]:
  embeddings = list(embed_model.embed(texts))
  return [e.tolist() for e in embeddings]


# --- Persistent client (disimpan ke folder lokal Windows) ---
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)

# Dapatkan atau buat collection baru dengan metrik Cosine
collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"},  # cosine similarity
)

# --- Dokumen untuk di-index ---
documents = [
    "RAG connects LLMs to external knowledge bases at inference time.",
    "Fine-tuning adapts model weights using task-specific training data.",
    "Prompt engineering guides LLM outputs without any model training.",
    "Vector databases index embeddings for fast similarity search.",
    "Agents use LLMs to plan and execute multi-step tasks with tools.",
]
ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [
    {"category": "technique", "index": i} for i in range(len(documents))
]

print("Menghitung embeddings...")
embeddings = embed(documents)

# Upsert dokumen ke koleksi
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
)
print(f"Collection size: {collection.count()}")

# --- Query Search ---
query = "how do I add knowledge to an LLM without retraining?"
print(f"\nQuery: '{query}'")
q_embed = embed([query])[0]

results = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    include=["documents", "distances", "metadatas"],
)

print("\n--- Top Results (Cosine Distance terkecil = paling mirip) ---")
for doc, dist, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0],
):
  print(f"Distance: [{dist:.4f}] -> {doc}")
  print(f"Metadata: category = {meta['category']}, index = {meta['index']}\n")