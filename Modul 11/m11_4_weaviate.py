import re
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


# --- Simulasi Weaviate v4 Python Client Lokal (Gratis, tanpa Docker / Cloud) ---
class MockWeaviateObject:

  def __init__(self, properties: dict, score: float = 0.0):
    self.properties = properties
    self.score = score


class MockWeaviateBatch:

  def __init__(self, collection):
    self.collection = collection

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def add_object(self, properties: dict, vector: list[float]):
    self.collection.objects.append({
        "properties": properties,
        "vector": np.array(vector, dtype=float),
    })


class MockWeaviateQuery:

  def __init__(self, collection):
    self.collection = collection

  def _cosine_similarity(self, v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm) if norm != 0 else 0.0

  def _bm25_simulated_score(self, query: str, text: str) -> float:
    q_tokens = set(re.findall(r"\w+", query.lower()))
    t_tokens = re.findall(r"\w+", text.lower())
    if not t_tokens or not q_tokens:
      return 0.0
    matches = sum(1 for token in t_tokens if token in q_tokens)
    return matches / len(t_tokens)

  def near_vector(
      self,
      near_vector: list[float],
      limit: int = 3,
      return_properties: list = None,
      filters=None,
  ):
    q_vec = np.array(near_vector, dtype=float)
    scored = []
    for obj in self.collection.objects:
      if filters and not filters(obj["properties"]):
        continue
      sim = self._cosine_similarity(q_vec, obj["vector"])
      props = (
          {k: obj["properties"][k] for k in return_properties}
          if return_properties
          else obj["properties"]
      )
      scored.append(MockWeaviateObject(properties=props, score=sim))
    scored.sort(key=lambda x: x.score, reverse=True)
    return MockWeaviateQueryResult(scored[:limit])

  def hybrid(
      self, query: str, vector: list[float], alpha: float = 0.5, limit: int = 3
  ):
    q_vec = np.array(vector, dtype=float)
    scored = []
    for obj in self.collection.objects:
      vec_score = self._cosine_similarity(q_vec, obj["vector"])
      bm25_score = self._bm25_simulated_score(
          query, obj["properties"].get("text", "")
      )
      # Rumus hybrid search Weaviate: (1 - alpha) * BM25 + alpha * Vector
      combined_score = ((1.0 - alpha) * bm25_score) + (alpha * vec_score)
      scored.append(
          MockWeaviateObject(
              properties=obj["properties"], score=combined_score
          )
      )
    scored.sort(key=lambda x: x.score, reverse=True)
    return MockWeaviateQueryResult(scored[:limit])


class MockWeaviateQueryResult:

  def __init__(self, objects: list):
    self.objects = objects


class MockWeaviateCollection:

  def __init__(self, name: str):
    self.name = name
    self.objects = []
    self.batch = MockWeaviateBatch(self)
    self.batch.dynamic = lambda: self.batch
    self.query = MockWeaviateQuery(self)


class MockWeaviateCollections:

  def __init__(self):
    self._collections = {}

  def create(self, name: str, **kwargs):
    self._collections[name] = MockWeaviateCollection(name)
    return self._collections[name]

  def get(self, name: str):
    return self._collections[name]


class MockWeaviateClient:

  def __init__(self):
    self.collections = MockWeaviateCollections()


# Filter class yang kompatibel dengan Weaviate v4
class Filter:

  class _FilterBuilder:

    def __init__(self, prop: str):
      self.prop = prop

    def equal(self, val):
      return lambda props: props.get(self.prop) == val

  @staticmethod
  def by_property(prop: str):
    return Filter._FilterBuilder(prop)


# =====================================================================
# EKSEKUSI WORKFLOW WEAVIATE 11.4
# =====================================================================
print(
    "--- 1. Inisialisasi Mock Weaviate Client & Buat Collection (AIDocument)"
    " ---"
)
client = MockWeaviateClient()
client.collections.create(name="AIDocument")
collection = client.collections.get("AIDocument")

documents = [
    {
        "text": (
            "RAG connects LLMs to external knowledge bases at inference time."
        ),
        "source": "guide",
        "category": "ai",
    },
    {
        "text": (
            "Fine-tuning adapts model weights using task-specific training"
            " data."
        ),
        "source": "paper",
        "category": "ml",
    },
    {
        "text": (
            "Prompt engineering guides LLM outputs without any training."
        ),
        "source": "guide",
        "category": "ai",
    },
    {
        "text": (
            "Vector databases like Weaviate index embeddings for fast hybrid"
            " search."
        ),
        "source": "docs",
        "category": "ai",
    },
]

print("--- 2. Batch Insert Dokumen dengan Ollama Embeddings ---")
with collection.batch.dynamic() as batch:
  for doc in documents:
    batch.add_object(
        properties={
            "text": doc["text"],
            "source": doc["source"],
            "category": doc["category"],
        },
        vector=embed([doc["text"]])[0],
    )
print(f"Berhasil memasukkan {len(collection.objects)} objek ke Weaviate.\n")

# --- 3. Vector Search (Near Vector) ---
print("--- 3. Vector Search (near_vector) ---")
q_vec = embed(["external knowledge for LLMs"])[0]
results_vector = collection.query.near_vector(
    near_vector=q_vec,
    limit=3,
    return_properties=["text", "source"],
)
for obj in results_vector.objects:
  print(
      f"Score [{obj.score:.4f}] | Source: {obj.properties['source']} | Text:"
      f" {obj.properties['text']}"
  )

# --- 4. Hybrid Search (BM25 Keyword + Vector) ---
print("\n--- 4. Hybrid Search (BM25 + Vector, alpha=0.5) ---")
results_hybrid = collection.query.hybrid(
    query="external knowledge LLM",
    vector=q_vec,
    alpha=0.5,  # 0 = pure BM25, 1 = pure vector
    limit=3,
)
for obj in results_hybrid.objects:
  print(
      f"Combined Score [{obj.score:.4f}] | Category:"
      f" {obj.properties['category']} | Text: {obj.properties['text']}"
  )

# --- 5. Metadata Filter ---
print("\n--- 5. Near Vector + Metadata Filter (category == 'ai') ---")
q_db_vec = embed(["vector database"])[0]
results_filtered = collection.query.near_vector(
    near_vector=q_db_vec,
    limit=3,
    filters=Filter.by_property("category").equal("ai"),
)
for obj in results_filtered.objects:
  print(
      f"Score [{obj.score:.4f}] | Category: {obj.properties['category']} |"
      f" Text: {obj.properties['text']}"
  )