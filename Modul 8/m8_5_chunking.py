import re
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """
    Memotong teks menjadi chunk yang menghormati batas kalimat (sentence boundaries).
    Menambahkan overlap agar konteks di batas chunk tidak hilang.
    """
    # Split berdasarkan akhir kalimat (. ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[Chunk] = []
    current = ""
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > max_chars and current:
            # Simpan chunk saat ini
            end = char_offset + len(current)
            chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))
            chunk_idx += 1

            # Mulai chunk baru dengan overlap dari bagian akhir chunk sebelumnya
            overlap_start = max(0, len(current) - overlap_chars)
            overlap_text = current[overlap_start:]
            current = (overlap_text + " " + sentence).strip()
            char_offset = end - len(overlap_text)
        else:
            current = candidate

    # Simpan chunk terakhir
    if current.strip():
        end = char_offset + len(current)
        chunks.append(Chunk(doc_id, chunk_idx, current.strip(), char_offset, end))

    return chunks


if __name__ == "__main__":
    print(f"--- Modul 8.5 Chunking Text for Embedding Demo ---\n")

    document = """
Large language models (LLMs) are neural networks trained on vast amounts of text data.
They learn to predict the next token in a sequence, which gives them broad language understanding.
Models like GPT-4 and Claude are examples of LLMs used in production today.
Retrieval-Augmented Generation, or RAG, extends LLMs by connecting them to external knowledge bases.
Instead of relying solely on knowledge encoded during training, a RAG system retrieves relevant documents at inference time.
This allows the model to answer questions about recent events or private data it was never trained on.
The retrieval step in RAG typically uses embedding-based semantic search.
A query is embedded into a vector, and the nearest document vectors are retrieved from a database.
These documents are then injected into the LLM's context window alongside the query.
"""

    chunks = chunk_by_sentences(document.strip(), doc_id="intro_to_llms", max_chars=300, overlap_chars=50)

    print(f"Total chunk yang dihasilkan: {len(chunks)}\n")
    for c in chunks:
        print(f"Chunk {c.chunk_index} ({c.char_start}-{c.char_end}):")
        print(f"  Teks: {c.text}")
        print("-" * 50)

    # Contoh lanjutan: Menghasilkan embedding untuk setiap chunk di Ollama
    chunk_texts = [f"search_document: {c.text}" for c in chunks]
    response = ollama.embed(model=EMBED_MODEL, input=chunk_texts)
    print(f"\n[Success] Berhasil meng-embed {len(response['embeddings'])} chunks menggunakan '{EMBED_MODEL}'!")