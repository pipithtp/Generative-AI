import os
import numpy as np
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def embed_voyage_style(
    texts: list[str], 
    input_type: str = "document", 
    model: str = EMBED_MODEL
) -> np.ndarray:
    """
    Mengadaptasi gaya Voyage AI (input_type: 'document' vs 'query') 
    ke model Ollama nomic-embed-text menggunakan prefix khusus.
    """
    # Menambahkan prefix sesuai spesifikasi nomic-embed-text
    prefix = "search_document: " if input_type == "document" else "search_query: "
    prefixed_texts = [f"{prefix}{text}" for text in texts]

    response = ollama.embed(model=model, input=prefixed_texts)
    
    # Ambil token usage jika tersedia di response Ollama
    prompt_tokens = response.get("prompt_eval_count", "N/A")
    print(f"[Info] Input Type: '{input_type}' | Estimated Tokens: {prompt_tokens}")

    return np.array(response["embeddings"], dtype=np.float32)


if __name__ == "__main__":
    print(f"--- Modul 8.2 Voyage AI-Style Embeddings (Model: {EMBED_MODEL}) ---\n")

    # 1. Embed Korpus Dokumen
    documents = [
        "What is RAG?",
        "Explain vector databases."
    ]
    doc_embeddings = embed_voyage_style(documents, input_type="document")
    
    print(f"Shape            : {doc_embeddings.shape}")  # (2, 768) untuk nomic-embed-text
    print(f"Sample 3 Values  : {doc_embeddings[0][:3]}\n")

    # 2. Embed Kueri Pencarian
    queries = ["How does RAG work?"]
    query_embeddings = embed_voyage_style(queries, input_type="query")
    print(f"Query Shape      : {query_embeddings.shape}")