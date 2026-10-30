from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# 1. Membuat dokumen dari teks beserta metadatanya
documents = [
    Document(
        text="""LlamaIndex is a data framework for LLM applications.
It provides tools for ingesting, structuring, and querying data
using large language models. The core abstraction is the Index,
which organises documents into a structure optimised for retrieval.""",
        metadata={"source": "llamaindex_overview", "category": "framework"},
    ),
    Document(
        text="""Vector store indexes store document embeddings in a vector database.
They support fast semantic search using approximate nearest-neighbour algorithms.
LlamaIndex integrates with ChromaDB, Pinecone, Weaviate, and many other backends.""",
        metadata={"source": "vector_store_guide", "category": "retrieval"},
    ),
    Document(
        text="""The query engine is the highest-level abstraction in LlamaIndex.
It takes a natural language question, retrieves relevant nodes,
and uses an LLM to synthesise a coherent answer from the retrieved context.""",
        metadata={"source": "query_engine_docs", "category": "querying"},
    ),
]

# 2. Parsing Dokumen menjadi Nodes (Chunks)
parser = SentenceSplitter(chunk_size=128, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)

print(f"Total Dokumen: {len(documents)}, Total Nodes: {len(nodes)}\n")

for i, n in enumerate(nodes, 1):
  print(f"--- Node #{i} (ID: {n.node_id[:8]}...) ---")
  print(f"Metadata : {n.metadata}")
  print(f"Panjang  : {len(n.text)} karakter")
  print(f"Konten   :\n{n.text}\n")