import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(
    f"Menyiapkan Custom Query Pipeline dengan model: {ollama_model}"
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

# 2. Dokumen pengetahuan tentang algoritma vector search
documents = [
    Document(
        text=(
            "HNSW (Hierarchical Navigable Small World) is the graph algorithm"
            " powering most modern vector search."
        )
    ),
    Document(
        text=(
            "IVF (Inverted File Index) partitions vectors into clusters for fast"
            " approximate search."
        )
    ),
    Document(
        text=(
            "PQ (Product Quantisation) compresses vectors to reduce memory"
            " usage at the cost of recall."
        )
    ),
    Document(
        text=(
            "Exact nearest-neighbour search guarantees correctness but is O(n)"
            " - unusable at scale."
        )
    ),
]

# 3. Bangun index dan definisikan retriever mandiri
index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)

# 4. Bangun Query Engine secara eksplisit dengan berbagai ResponseMode
query_text = "What algorithms are used in vector search?"
print(f"Query: '{query_text}'\n")

for mode in [ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE]:
  # Inisialisasi synthesiser sesuai response_mode
  synthesiser = get_response_synthesizer(response_mode=mode)

  # Gabungkan retriever dan synthesiser menjadi RetrieverQueryEngine
  engine = RetrieverQueryEngine(
      retriever=retriever,
      response_synthesizer=synthesiser,
  )

  resp = engine.query(query_text)
  print(f"=== Response Mode: [{mode.value}] ===")
  print(f"{str(resp)}\n")