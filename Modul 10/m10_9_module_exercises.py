import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field

# 1. Konfigurasi Environment & Model Lokal
load_dotenv()
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menggunakan Model: {ollama_model} ({ollama_host})\n")

Settings.llm = Ollama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.2,
    request_timeout=120.0,
)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")


# =====================================================================
# LATIHAN 1: Folder Q&A System with ask(question) -> dict
# =====================================================================
print("=" * 65)
print("LATIHAN 1: Document Q&A System (Folder .txt -> Answer + Sources)")
print("=" * 65)

SAMPLE_DATA_DIR = "./sample_docs"
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
with open(
    os.path.join(SAMPLE_DATA_DIR, "intro_genai.txt"), "w", encoding="utf-8"
) as f:
  f.write(
      "Generative AI models create new synthetic content including text,"
      " images, and code using trained probabilistic neural representations."
  )
with open(
    os.path.join(SAMPLE_DATA_DIR, "rag_architecture.txt"), "w", encoding="utf-8"
) as f:
  f.write(
      "RAG combines neural dense retrieval with generative text synthesis to"
      " provide dynamic factual updates to LLM context windows."
  )

docs_l1 = SimpleDirectoryReader(
    input_dir=SAMPLE_DATA_DIR, required_exts=[".txt"]
).load_data()
index_l1 = VectorStoreIndex.from_documents(docs_l1)
engine_l1 = index_l1.as_query_engine(similarity_top_k=2)


def ask(question: str) -> Dict[str, Any]:
  response = engine_l1.query(question)
  sources = list({
      node.metadata.get("file_name", "unknown")
      for node in response.source_nodes
  })
  return {
      "question": question,
      "answer": str(response).strip(),
      "sources": sources,
  }


l1_result = ask("What does RAG combine?")
print(f"Question : {l1_result['question']}")
print(f"Answer   : {l1_result['answer']}")
print(f"Sources  : {l1_result['sources']}\n")


# =====================================================================
# LATIHAN 2: VectorStoreIndex vs SummaryIndex Comparison
# =====================================================================
print("=" * 65)
print("LATIHAN 2: VectorStoreIndex vs SummaryIndex Comparison")
print("=" * 65)

ten_paragraphs = [
    Document(
        text=(
            "Paragraph 1: Convolutional Neural Networks (CNN) excel in 2D"
            " spatial grid processing for image vision."
        )
    ),
    Document(
        text=(
            "Paragraph 2: Recurrent Neural Networks (RNN) and LSTMs capture"
            " temporal sequences but suffer from vanishing gradients."
        )
    ),
    Document(
        text=(
            "Paragraph 3: Transformers replaced recurrence with self-attention"
            " mechanisms in 2017."
        )
    ),
    Document(
        text=(
            "Paragraph 4: BERT introduced bidirectional transformer encoders"
            " for masked language modeling tasks."
        )
    ),
    Document(
        text=(
            "Paragraph 5: GPT introduced autoregressive decoder-only"
            " pretraining optimized for text completion."
        )
    ),
    Document(
        text=(
            "Paragraph 6: Modern LLMs require prompt engineering, fine-tuning,"
            " or retrieval augmentation for domain specialization."
        )
    ),
    Document(
        text=(
            "Paragraph 7: ChromaDB and FAISS serve high-speed approximate"
            " nearest-neighbor vector lookups."
        )
    ),
    Document(
        text=(
            "Paragraph 8: LoRA and QLoRA enable low-rank adaptation with"
            " dramatically reduced GPU memory footprint."
        )
    ),
    Document(
        text=(
            "Paragraph 9: Hallucinations occur when generative models"
            " synthesize plausible but factually incorrect assertions."
        )
    ),
    Document(
        text=(
            "Paragraph 10: Conclusion: The AI paradigm has transitioned from"
            " task-specific discriminative models to versatile foundation"
            " models."
        )
    ),
]

vec_index = VectorStoreIndex.from_documents(ten_paragraphs)
sum_index = SummaryIndex.from_documents(ten_paragraphs)

vec_engine = vec_index.as_query_engine(similarity_top_k=2)
sum_engine = sum_index.as_query_engine()

questions_5 = [
    "When were Transformers introduced?",  # Specific Fact
    "What technique reduces GPU memory for fine-tuning?",  # Specific Fact
    "What is the role of ChromaDB?",  # Specific Fact
    (
        "Summarise the progression of model architectures across all"
        " paragraphs."
    ),  # Document-wide Synthesis
    "Provide a unified takeaway of all 10 paragraphs.",  # Document-wide Synthesis
]

for idx, q in enumerate(questions_5, 1):
  q_type = "Specific Fact" if idx <= 3 else "Document-wide Synthesis"
  print(f"\n[Q{idx} - {q_type}]: {q}")
  v_ans = vec_engine.query(q)
  s_ans = sum_engine.query(q)
  print(f"  > VectorStoreIndex : {str(v_ans).strip()[:110]}...")
  print(f"  > SummaryIndex     : {str(s_ans).strip()[:110]}...")


# =====================================================================
# LATIHAN 3: Metadata-Filtered Retriever (topic & year)
# =====================================================================
print("\n" + "=" * 65)
print("LATIHAN 3: Metadata-Filtered Retriever (Year & Topic)")
print("=" * 65)

tagged_corpus = [
    Document(
        text="Modern dense embeddings in 2024 leverage contrastive loss.",
        metadata={"year": 2024, "topic": "embeddings"},
    ),
    Document(
        text="Sparse keyword retrieval remains strong in hybrid search setups.",
        metadata={"year": 2024, "topic": "search"},
    ),
    Document(
        text="Word2Vec in 2013 pioneered word vector representations.",
        metadata={"year": 2013, "topic": "embeddings"},
    ),
    Document(
        text=(
            "ColBERT introduces late-interaction token embeddings in 2024."
        ),
        metadata={"year": 2024, "topic": "embeddings"},
    ),
    Document(
        text="Quantisation schemes reduce embedding cache size in 2023.",
        metadata={"year": 2023, "topic": "embeddings"},
    ),
]

tagged_index = VectorStoreIndex.from_documents(tagged_corpus)


def retrieve_filtered(
    query: str, target_year: int, target_topic: str
) -> List[Any]:
  filters = MetadataFilters(
      filters=[
          MetadataFilter(
              key="year", value=target_year, operator=FilterOperator.EQ
          ),
          MetadataFilter(
              key="topic", value=target_topic, operator=FilterOperator.EQ
          ),
      ]
  )
  filtered_retriever = tagged_index.as_retriever(
      similarity_top_k=5, filters=filters
  )
  return filtered_retriever.retrieve(query)


filtered_nodes = retrieve_filtered(
    "embedding advances", target_year=2024, target_topic="embeddings"
)
print(
    f"Total nodes matching (year=2024, topic='embeddings'):"
    f" {len(filtered_nodes)}"
)
for n in filtered_nodes:
  print(f"  - Metadata: {n.metadata} | Text: {n.text}")


# =====================================================================
# LATIHAN 4: Two-Stage Pipeline (Top-10 Retrieval -> LLM Re-ranking Top-3)
# =====================================================================
print("\n" + "=" * 65)
print("LATIHAN 4: Two-Stage Pipeline (Top-10 Retrieve -> LLM Re-rank Top-3)")
print("=" * 65)

pool_docs = [
    Document(text=f"Doc {i}: " + txt)
    for i, txt in enumerate(
        [
            (
                "HNSW builds multi-layer geometric graphs for approximate"
                " nearest-neighbor search."
            ),
            (
                "Postgres pgvector enables SQL relational databases to perform"
                " vector queries."
            ),
            (
                "Qwen2.5 is an open weights LLM series with high instruction"
                " and coding abilities."
            ),
            (
                "Inverted file indexes cluster feature vectors to narrow"
                " distance computation batches."
            ),
            (
                "Quantization compresses 32-bit floating point vectors down to"
                " 8-bit or 4-bit representations."
            ),
            (
                "Cosine similarity measures the angle between normalized vector"
                " directions."
            ),
            (
                "Dot product equivalence holds when vector embeddings are L2"
                " unit normalized."
            ),
            (
                "Cross-encoders jointly process queries and passages for"
                " higher reranking accuracy."
            ),
            (
                "Reciprocal Rank Fusion merges ranked result lists without"
                " needing calibrated scores."
            ),
            (
                "Text chunk overlap prevents context fragmentation across"
                " document boundaries."
            ),
            (
                "ChromaDB stores persistent local collections using SQLite and"
                " Parquet files."
            ),
            (
                "LangChain and LlamaIndex provide abstractions for RAG and"
                " autonomous agents."
            ),
        ],
        1,
    )
]

pool_index = VectorStoreIndex.from_documents(pool_docs)
stage1_retriever = VectorIndexRetriever(index=pool_index, similarity_top_k=10)


class NodeRelevanceScore(BaseModel):
  score: int = Field(
      description=(
          "Relevance score from 0 to 10 evaluating how relevant the text is to"
          " the query"
      )
  )


def two_stage_retrieve_rerank(query: str) -> List[tuple]:
  # Stage 1: Ambil Top-10 menggunakan VectorIndexRetriever
  retrieved_nodes = stage1_retriever.retrieve(query)
  print(f"Stage 1 Retrieved: {len(retrieved_nodes)} candidate nodes.")

  scored_nodes = []
  # Stage 2: Skoring tiap node menggunakan LLM (0-10)
  for node in retrieved_nodes:
    prompt = (
        f"Question: {query}\n\nDocument snippet:\n{node.text}\n\nEvaluate how"
        " directly and accurately this document answers or relates to the"
        " question. Rate its relevance strictly on a scale of 0 to 10 (10 ="
        " directly answers, 0 = completely irrelevant)."
    )
    try:
      eval_res: NodeRelevanceScore = Settings.llm.as_structured_llm(
          NodeRelevanceScore
      ).complete(prompt).raw
      score = eval_res.score
    except Exception:
      resp_text = Settings.llm.complete(
          prompt + "\nReturn only an integer number between 0 and 10."
      ).text.strip()
      digits = "".join([c for c in resp_text if c.isdigit()])
      score = int(digits[:2]) if digits else 5

    scored_nodes.append((score, node))

  # Urutkan berdasarkan skor LLM tertinggi dan ambil Top-3
  scored_nodes.sort(key=lambda x: x[0], reverse=True)
  return scored_nodes[:3]


test_query = "How do vector indexing algorithms optimize similarity searches?"
top3_reranked = two_stage_retrieve_rerank(test_query)

print(f"\nTop-3 Re-ranked Results for query: '{test_query}'")
for rank, (score, node) in enumerate(top3_reranked, 1):
  print(f"Rank {rank} | LLM Relevance Score: {score}/10")
  print(f"       Text: {node.text}\n")