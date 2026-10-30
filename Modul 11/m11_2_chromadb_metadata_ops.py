import os
import chromadb
from fastembed import TextEmbedding

# Menyiapkan model embedding lokal CPU-friendly (FastEmbed)
print("Memuat FastEmbed (BAAI/bge-small-en-v1.5)...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed(texts: list[str]) -> list[list[float]]:
  embeddings = list(embed_model.embed(texts))
  return [e.tolist() for e in embeddings]


# Persistent client lokal Windows
persist_path = "./chroma_db"
client = chromadb.PersistentClient(path=persist_path)

collection = client.get_or_create_collection(
    name="ai_knowledge_base",
    metadata={"hnsw:space": "cosine"},
)

# Inisialisasi ulang/reset data awal agar operasi update dan delete berjalan konsisten
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

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embed(documents),
    metadatas=metadatas,
)
print(f"Jumlah awal koleksi: {collection.count()}\n")

# 1. Metadata Filtering (Pre-filter sebelum perangkingan similarity)
query = "how do I add knowledge to an LLM without retraining?"
q_embed = embed([query])[0]

print("=== 1. Metadata Filtering (where category == 'technique') ===")
results_filtered = collection.query(
    query_embeddings=[q_embed],
    n_results=3,
    where={"category": {"$eq": "technique"}},  # Metadata filter
    include=["documents", "distances", "metadatas"],
)

for doc, dist, meta in zip(
    results_filtered["documents"][0],
    results_filtered["distances"][0],
    results_filtered["metadatas"][0],
):
  print(f"Distance: [{dist:.4f}] | Metadata: {meta} | {doc}")

# 2. Update Dokumen (memperbarui teks dan vektor embedding doc_0)
print("\n=== 2. Update Dokumen (doc_0) ===")
updated_text = (
    "RAG retrieves external documents and injects them into the LLM context."
)
collection.update(
    ids=["doc_0"],
    documents=[updated_text],
    embeddings=embed([updated_text]),
)
updated_doc = collection.get(ids=["doc_0"])
print(f"Isi doc_0 setelah di-update:\n  {updated_doc['documents'][0]}")

# 3. Delete Dokumen (menghapus doc_4)
print("\n=== 3. Delete Dokumen (doc_4) ===")
collection.delete(ids=["doc_4"])
print(f"After delete, count: {collection.count()}")