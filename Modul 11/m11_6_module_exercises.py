import json
import os
import shutil
import time
import chromadb
from fastembed import TextEmbedding
import numpy as np

# Inisialisasi model embedding lokal (100% gratis, cepat, jalan di CPU)
print("Memuat FastEmbed (BAAI/bge-small-en-v1.5)...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
EMBED_DIM = 384


def embed(texts: list[str]) -> list[list[float]]:
  embeddings = list(embed_model.embed(texts))
  return [e.tolist() for e in embeddings]


# Path folder database pengujian
BENCH_DIR = "./chroma_bench_db"
if os.path.exists(BENCH_DIR):
  shutil.rmtree(BENCH_DIR)

client = chromadb.PersistentClient(path=BENCH_DIR)

# =====================================================================
# SOAL 1: VectorDBBenchmark
# =====================================================================
print("\n" + "=" * 65)
print("SOAL 1: VectorDBBenchmark (1.000 Dokumen, 100 Queries)")
print("=" * 65)


class VectorDBBenchmark:

  def __init__(self, client: chromadb.ClientAPI):
    self.client = client
    self.collection = self.client.get_or_create_collection(
        name="benchmark_coll", metadata={"hnsw:space": "cosine"}
    )

  def run_benchmark(self, num_docs: int = 1000, num_queries: int = 100):
    print(f"1. Menyiapkan {num_docs} dokumen sintetis...")
    # Menggunakan dummy normalized embedding acak untuk mengukur murni latensi & throughput DB
    np.random.seed(42)
    raw_vecs = np.random.randn(num_docs, EMBED_DIM).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norm_vecs = (raw_vecs / norms).tolist()

    docs = [
        f"This is synthetic benchmark document number {i}"
        for i in range(num_docs)
    ]
    ids = [f"bench_doc_{i}" for i in range(num_docs)]
    metadatas = [
        {"doc_index": i, "category": f"cat_{i % 5}"} for i in range(num_docs)
    ]

    # Chroma batch insert
    batch_size = 250
    print(f"2. Memasukkan dokumen ke ChromaDB dalam batch {batch_size}...")
    t0 = time.perf_counter()
    for i in range(0, num_docs, batch_size):
      self.collection.upsert(
          ids=ids[i : i + batch_size],
          documents=docs[i : i + batch_size],
          embeddings=norm_vecs[i : i + batch_size],
          metadatas=metadatas[i : i + batch_size],
      )
    insert_duration = time.perf_counter() - t0
    print(f"   Selesai insert dalam {insert_duration:.2f} detik.")

    # Menyiapkan 100 queries
    print(f"3. Menjalankan {num_queries} queries...")
    q_raw = np.random.randn(num_queries, EMBED_DIM).astype(np.float32)
    q_norms = np.linalg.norm(q_raw, axis=1, keepdims=True)
    q_vecs = (q_raw / q_norms).tolist()

    latencies = []
    total_results_retrieved = 0
    top_k = 5

    t_start_all = time.perf_counter()
    for q in q_vecs:
      t_q_start = time.perf_counter()
      res = self.collection.query(
          query_embeddings=[q],
          n_results=top_k,
          include=["documents", "distances"],
      )
      latencies.append((time.perf_counter() - t_q_start) * 1000)  # milidetik
      total_results_retrieved += len(res["documents"][0])
    total_time = time.perf_counter() - t_start_all

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    throughput_qps = num_queries / total_time
    throughput_rps = total_results_retrieved / total_time

    print("\n--- HASIL BENCHMARK ---")
    print(f"Average Query Latency  : {avg_lat:.2f} ms")
    print(f"95th Percentile Latency: {p95_lat:.2f} ms")
    print(f"Queries per Second     : {throughput_qps:.2f} QPS")
    print(f"Throughput Results/sec : {throughput_rps:.2f} results/s")


bench = VectorDBBenchmark(client)
bench.run_benchmark(num_docs=1000, num_queries=100)


# =====================================================================
# SOAL 2: MultiCollectionSearch
# =====================================================================
print("\n" + "=" * 65)
print("SOAL 2: MultiCollectionSearch across Multiple Named Collections")
print("=" * 65)


class MultiCollectionSearch:

  def __init__(self, client: chromadb.ClientAPI, collection_names: list[str]):
    self.client = client
    self.collections = [
        self.client.get_or_create_collection(name) for name in collection_names
    ]

  def search(self, query_text: str, top_k: int = 3) -> list[dict]:
    q_emb = embed([query_text])[0]
    merged_results = []

    for coll in self.collections:
      res = coll.query(
          query_embeddings=[q_emb],
          n_results=top_k,
          include=["documents", "distances", "metadatas"],
      )
      for doc, dist, meta, doc_id in zip(
          res["documents"][0],
          res["distances"][0],
          res["metadatas"][0] or [{}] * len(res["documents"][0]),
          res["ids"][0],
      ):
        merged_results.append({
            "id": doc_id,
            "collection": coll.name,
            "document": doc,
            "distance": dist,
            "metadata": meta,
        })

    # Urutkan berdasarkan distance terendah (paling mirip)
    merged_results.sort(key=lambda x: x["distance"])
    return merged_results[:top_k]


c_tech = client.get_or_create_collection(
    "coll_tech", metadata={"hnsw:space": "cosine"}
)
c_biz = client.get_or_create_collection(
    "coll_business", metadata={"hnsw:space": "cosine"}
)

tech_docs = [
    "Vector databases store high-dimensional embeddings.",
    "Transformers utilize self-attention mechanisms.",
]
biz_docs = [
    "Startups optimize operational costs and LLM inference budgets.",
    "Market adoption of GenAI accelerated rapidly.",
]

c_tech.upsert(
    ids=["t1", "t2"], documents=tech_docs, embeddings=embed(tech_docs)
)
c_biz.upsert(ids=["b1", "b2"], documents=biz_docs, embeddings=embed(biz_docs))

multi_searcher = MultiCollectionSearch(client, ["coll_tech", "coll_business"])
unified_results = multi_searcher.search(
    "How do embeddings work in technology?", top_k=3
)

for rank, item in enumerate(unified_results, 1):
  print(
      f"Rank {rank} | Coll: [{item['collection']}] | Dist:"
      f" {item['distance']:.4f}"
  )
  print(f"       Teks: {item['document']}")


# =====================================================================
# SOAL 3: Tenant-Isolated Vector Store
# =====================================================================
print("\n" + "=" * 65)
print("SOAL 3: Tenant-Isolated Vector Store (TenantVectorStore)")
print("=" * 65)


class TenantVectorStore:

  def __init__(self, client: chromadb.ClientAPI):
    self.client = client

  def _get_tenant_collection(self, tenant_id: str):
    safe_tenant_name = f"tenant_{tenant_id.replace('-', '_')}"
    return self.client.get_or_create_collection(
        name=safe_tenant_name,
        metadata={"hnsw:space": "cosine", "tenant_id": tenant_id},
    )

  def add_documents(
      self,
      tenant_id: str,
      ids: list[str],
      documents: list[str],
      metadatas: list[dict] = None,
  ):
    coll = self._get_tenant_collection(tenant_id)
    embeddings = embed(documents)
    coll.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(
        f"[{tenant_id}] Berhasil menyimpan {len(documents)} dokumen ke koleksi"
        f" '{coll.name}'."
    )

  def search(self, tenant_id: str, query: str, top_k: int = 2) -> list[dict]:
    coll = self._get_tenant_collection(tenant_id)
    q_emb = embed([query])[0]
    res = coll.query(
        query_embeddings=[q_emb], n_results=top_k, include=["documents", "distances"]
    )
    output = []
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
      output.append({"tenant": tenant_id, "document": doc, "distance": dist})
    return output


tenant_store = TenantVectorStore(client)

# Data Tenant Alpha
tenant_store.add_documents(
    tenant_id="client_alpha",
    ids=["a1"],
    documents=[
        "Alpha Corporation internal roadmap for Q4 focuses on enterprise"
        " security."
    ],
)

# Data Tenant Beta
tenant_store.add_documents(
    tenant_id="client_beta",
    ids=["b1"],
    documents=["Beta Healthcare confidential clinical trials on oncology."],
)

print("\nTesting Isolasi: Search Tenant Alpha:")
res_alpha = tenant_store.search("client_alpha", "internal roadmap", top_k=2)
for r in res_alpha:
  print(f"  -> Found in {r['tenant']}: {r['document']}")


# =====================================================================
# SOAL 4: Export to JSON and Import Migration
# =====================================================================
print("\n" + "=" * 65)
print("SOAL 4: Export to JSON and Import Migration")
print("=" * 65)


def export_collection_to_json(collection, export_filepath: str):
  data = collection.get(include=["documents", "embeddings", "metadatas"])
  export_records = []

  count = len(data["ids"])
  for i in range(count):
    emb = data["embeddings"][i]
    emb_list = emb.tolist() if hasattr(emb, "tolist") else emb

    export_records.append({
        "id": data["ids"][i],
        "text": data["documents"][i],
        "embedding": emb_list,
        "metadata": data["metadatas"][i] if data["metadatas"] else {},
    })

  with open(export_filepath, "w", encoding="utf-8") as f:
    json.dump(export_records, f, indent=2)
  print(
      f"Koleksi '{collection.name}' berhasil diekspor ke '{export_filepath}'"
      f" ({count} records)."
  )


def import_collection_from_json(
    client: chromadb.ClientAPI, new_collection_name: str, import_filepath: str
):
  with open(import_filepath, "r", encoding="utf-8") as f:
    records = json.load(f)

  ids = [r["id"] for r in records]
  documents = [r["text"] for r in records]
  embeddings = [r["embedding"] for r in records]
  metadatas = [r["metadata"] for r in records]

  new_coll = client.get_or_create_collection(
      name=new_collection_name, metadata={"hnsw:space": "cosine"}
  )
  new_coll.upsert(
      ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
  )
  print(
      f"Koleksi '{new_collection_name}' berhasil diimpor dari"
      f" '{import_filepath}' ({new_coll.count()} items)."
  )


JSON_BACKUP = "./chroma_backup.json"
export_collection_to_json(c_tech, JSON_BACKUP)
import_collection_from_json(client, "coll_tech_restored", JSON_BACKUP)