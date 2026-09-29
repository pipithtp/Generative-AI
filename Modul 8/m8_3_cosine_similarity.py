import os
import numpy as np
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# --- 1. Cosine Similarity untuk Dua Vektor 1D ---
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity antara dua vektor 1-D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# --- 2. Pairwise Similarity Matriks (Semua Pasangan Teks) ---
def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Menghitung pairwise cosine similarity untuk seluruh baris dalam matriks.
    Mengembalikan matriks bernilai (n, n).
    """
    # Normalisasi semua baris ke unit length (panjang = 1.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = matrix / norms

    # Dot product dari vektor yang telah dinormalisasi = Cosine Similarity
    return (normed @ normed.T).astype(np.float32)


if __name__ == "__main__":
    print(f"--- Modul 8.3 Cosine Similarity Demo (Model: {EMBED_MODEL}) ---\n")

    # 1. Matriks Vektor Acak Sesuai Contoh Modul
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((4, 8)).astype(np.float32)
    sim_matrix = pairwise_similarity(vecs)

    print("Pairwise similarities (Vektor Acak):")
    for i in range(4):
        for j in range(i + 1, 4):
            print(f"  vec[{i}] vs vec[{j}] : {sim_matrix[i, j]:.4f}")

    print(f"Diagonal (Identik)  : {np.diag(sim_matrix)}\n")
    print("=" * 60 + "\n")

    # 2. Pengujian Nyata Menggunakan Embedding Ollama
    sample_texts = [
        "Retrieval-Augmented Generation combines search with LLMs.",
        "RAG retrieves documents then generates an answer from them.",
        "The Eiffel Tower is in Paris.",
        "Python is a popular programming language."
    ]

    print("Pengujian dengan Embedding Kalimat Nyata:")
    response = ollama.embed(model=EMBED_MODEL, input=sample_texts)
    real_embeddings = np.array(response["embeddings"], dtype=np.float32)

    real_sim_matrix = pairwise_similarity(real_embeddings)

    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            print(f"Text [{i}] vs Text [{j}]: {real_sim_matrix[i, j]:.4f}")
            print(f"  - Teks {i}: '{sample_texts[i]}'")
            print(f"  - Teks {j}: '{sample_texts[j]}'\n")