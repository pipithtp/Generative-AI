import numpy as np
import ollama

# Model embedding lokal via Ollama (100% gratis & offline)
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMENSION = 768

print(f"Menyiapkan Embedding via Ollama: {EMBED_MODEL}")
ollama_client = ollama.Client(host="http://localhost:11434")


def embed(texts: list[str]) -> list[list[float]]:
  response = ollama_client.embed(model=EMBED_MODEL, input=texts)
  return response.embeddings


# --- Simulasi Pinecone Client Lokal (Tanpa Biaya / API Key) ---
class MockPineconeIndex:

  def __init__(self, name: str):
    self.name = name
    self.vectors = {}

  def upsert(self, vectors: list[dict]):
    for v in vectors:
      self.vectors[v["id"]] = {
          "values": np.array(v["values"], dtype=float),
          "metadata": v.get("metadata", {}),
      }

  def describe_index_stats(self):
    return {
        "total_vector_count": len(self.vectors),
        "dimension": EMBED_DIMENSION,
    }

  def query(
      self,
      vector: list[float],
      top_k: int = 3,
      include_metadata: bool = True,
      filter: dict = None,
  ):
    q_vec = np.array(vector, dtype=float)
    scores = []
    for vid, data in self.vectors.items():
      if filter:
        match = all(
            data["metadata"].get(k) == v.get("$eq", v)
            for k, v in filter.items()
        )
        if not match:
          continue
      dot = np.dot(q_vec, data["values"])
      norm = np.linalg.norm(q_vec) * np.linalg.norm(data["values"])
      cosine_score = float(dot / norm) if norm != 0 else 0.0
      scores.append({
          "id": vid,
          "score": cosine_score,
          "metadata": data["metadata"] if include_metadata else {},
      })
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"matches": scores[:top_k]}


class MockPinecone:

  def __init__(self):
    self._indexes = {}

  def list_indexes(self):
    class IndexList:

      def __init__(self, names):
        self._names = names

      def names(self):
        return self._names

    return IndexList(list(self._indexes.keys()))

  def create_index(self, name: str, **kwargs):
    self._indexes[name] = MockPineconeIndex(name)

  def Index(self, name: str):
    return self._indexes[name]


# Inisialisasi Mock Pinecone
pc = MockPinecone()
INDEX_NAME = "ai-knowledge-base-ollama"

pc.create_index(name=INDEX_NAME, dimension=EMBED_DIMENSION, metric="cosine")
index = pc.Index(INDEX_NAME)

# --- Upsert vectors ---
documents = [
    {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
    {
        "id": "d2",
        "text": "Fine-tuning updates model weights on task-specific data.",
    },
    {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
]

print("Menghitung embeddings dengan Ollama dan menyiapkan vektor...")
vectors = [
    {
        "id": doc["id"],
        "values": embed([doc["text"]])[0],
        "metadata": {"text": doc["text"], "source": "guide"},
    }
    for doc in documents
]

index.upsert(vectors=vectors)
print("\nStatistik Index Pinecone (Simulasi Lokal Gratis):")
print(index.describe_index_stats())

# --- Query Similarity Search ---
query_text = "how to add external knowledge to an LLM?"
print(f"\n--- Melakukan Query: '{query_text}' ---")
q_vector = embed([query_text])[0]

results = index.query(
    vector=q_vector,
    top_k=3,
    include_metadata=True,
)

print("Hasil Pencarian:")
for match in results["matches"]:
  print(f"[{match['score']:.4f}] {match['metadata']['text']}")

# --- Query dengan Metadata Filter ---
print("\n--- Query dengan Metadata Filter (source == 'guide') ---")
filtered = index.query(
    vector=q_vector,
    top_k=3,
    filter={"source": {"$eq": "guide"}},
    include_metadata=True,
)

for match in filtered["matches"]:
  print(f"[{match['score']:.4f}] {match['metadata']['text']}")