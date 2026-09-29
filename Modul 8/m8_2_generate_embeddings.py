import os
import numpy as np
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Mengubah daftar teks menjadi NumPy Array berisi vektor embedding."""
    # Pemanggilan API Ollama Embeddings
    response = ollama.embed(model=model, input=texts)
    
    # Ambil list vektor hasil ekstraksi
    vectors = response["embeddings"]
    return np.array(vectors, dtype=np.float32)


# Data teks sampel
texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]

if __name__ == "__main__":
    print(f"--- Modul 8.2 Generating Embeddings (Model: {EMBED_MODEL}) ---\n")

    embeddings = embed(texts)

    # 1. Tampilkan dimensi array (Jumlah Teks, Dimensi Vektor)
    print(f"Shape/Dimensi Vektor : {embeddings.shape}")

    # 2. Hitung panjang/magnitude vektor pertama
    first_vector_norm = np.linalg.norm(embeddings[0])
    print(f"Norm (Panjang) Vektor Pertama : {first_vector_norm:.4f}")

    # 3. Cetak sampel 5 angka pertama dari vektor pertama
    print(f"Sample 5 Nilai Vektor Pertama : {embeddings[0][:5]}")