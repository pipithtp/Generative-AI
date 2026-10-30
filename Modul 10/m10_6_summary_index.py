import os
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SummaryIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menyiapkan SummaryIndex dengan model: {ollama_model} ({ollama_host})\n")

# Konfigurasi LLM dan Embedding global
Settings.llm = Ollama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.2,
    request_timeout=120.0,
)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Simulasi dokumen panjang yang terbagi menjadi beberapa bagian
sections = [
    Document(
        text=(
            "Introduction: This report analyses the performance of five LLMs"
            " on coding, reasoning, and language tasks."
        )
    ),
    Document(
        text=(
            "Section 1 - Coding: Claude Sonnet 4.5 scored highest on HumanEval"
            " with 94.2%. GPT-4o followed at 90.1%."
        )
    ),
    Document(
        text=(
            "Section 2 - Reasoning: GPT-4o led on GSM8K with 97.1%. Claude"
            " Sonnet 4.5 achieved 96.8%."
        )
    ),
    Document(
        text=(
            "Section 3 - Language: Both models performed similarly on MMLU,"
            " within 1% of each other."
        )
    ),
    Document(
        text=(
            "Conclusion: For coding tasks, prefer Claude Sonnet 4.5. For"
            " balanced performance, GPT-4o is the safer choice."
        )
    ),
]

# Bangun SummaryIndex
# Berbeda dengan VectorStoreIndex, SummaryIndex membaca dan mensintesis seluruh nodes
print("Membangun SummaryIndex...")
index = SummaryIndex.from_documents(sections)

# Inisialisasi query engine
query_engine = index.as_query_engine()

pertanyaan = "Which model should I use for a coding project?"
print(f"Pertanyaan: {pertanyaan}\n")
response = query_engine.query(pertanyaan)

print("--- Jawaban ---")
print(response)