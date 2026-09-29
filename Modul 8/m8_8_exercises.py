import os
import sqlite3
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# =====================================================================
# Exercise 1: DuplicateDetector Class (Near-Duplicate Detection)
# =====================================================================
class DuplicateDetector:
    def __init__(self, threshold: float = 0.90, model: str = EMBED_MODEL):
        self.threshold = threshold
        self.model = model

    def find_duplicates(self, texts: list[str]) -> list[tuple[int, int, float]]:
        """Mengembalikan pasangan indeks (i, j) dan skor similarity yang melebihi threshold."""
        prefixed = [f"search_document: {t}" for t in texts]
        resp = ollama.embed(model=self.model, input=prefixed)
        vecs = np.array(resp["embeddings"], dtype=np.float32)

        # Normalisasi ke unit length
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.where(norms == 0, 1, norms)

        # Matriks Cosine Similarity (n, n)
        sim_matrix = normed @ normed.T

        duplicates = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= self.threshold:
                    duplicates.append((i, j, round(score, 4)))
        return duplicates


# =====================================================================
# Exercise 2: HybridSearch Class (Semantic + Keyword Search)
# =====================================================================
class HybridSearch:
    def __init__(self, corpus: list[str], model: str = EMBED_MODEL):
        self.corpus = corpus
        self.model = model
        
        # Pre-compute semantic embeddings
        prefixed = [f"search_document: {t}" for t in corpus]
        resp = ollama.embed(model=self.model, input=prefixed)
        vecs = np.array(resp["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.matrix = vecs / np.where(norms == 0, 1, norms)

    def _keyword_score(self, query: str, doc: str) -> float:
        """Skor BM25 sederhana berbasis rasio kemunculan kata kunci (TF/Keyword Overlap)."""
        q_words = set(query.lower().split())
        doc_words = doc.lower().split()
        if not q_words or not doc_words:
            return 0.0
        matches = sum(1 for w in doc_words if w in q_words)
        return matches / len(q_words)

    def search(self, query: str, alpha: float = 0.7, top_k: int = 3) -> list[tuple[str, float]]:
        """
        GABUNGAN SKOR: final_score = alpha * semantic + (1 - alpha) * keyword
        """
        # 1. Semantic Scores
        q_resp = ollama.embed(model=self.model, input=[f"search_query: {query}"])
        q_vec = np.array(q_resp["embeddings"][0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        semantic_scores = self.matrix @ q_vec

        # 2. Keyword Scores
        keyword_scores = np.array([self._keyword_score(query, doc) for doc in self.corpus], dtype=np.float32)

        # 3. Hybrid Blending
        final_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

        top_idx = np.argsort(final_scores)[::-1][:top_k]
        return [(self.corpus[i], float(final_scores[i])) for i in top_idx]


# =====================================================================
# Exercise 3: VectorStore dengan Dukungan Delete & Update Operations
# =====================================================================
@dataclass
class Document:
    id: str
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


class DynamicVectorStore:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model
        self.docs: list[Document] = []
        self.matrix: Optional[np.ndarray] = None

    def _embed_text(self, text: str) -> np.ndarray:
        resp = ollama.embed(model=self.model, input=[f"search_document: {text}"])
        vec = np.array(resp["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _rebuild_matrix(self):
        if not self.docs:
            self.matrix = None
        else:
            self.matrix = np.array([d.embedding for d in self.docs], dtype=np.float32)

    def add(self, doc_id: str, text: str):
        vec = self._embed_text(text)
        self.docs.append(Document(doc_id, text, vec))
        self._rebuild_matrix()

    def delete(self, doc_id: str) -> bool:
        """Menghapus dokumen berdasarkan doc_id dan merekonstruksi matriks internal."""
        initial_count = len(self.docs)
        self.docs = [d for d in self.docs if d.id != doc_id]
        if len(self.docs) < initial_count:
            self._rebuild_matrix()
            print(f"[VectorStore] Document '{doc_id}' deleted.")
            return True
        print(f"[VectorStore] Document '{doc_id}' not found.")
        return False

    def update(self, doc_id: str, new_text: str) -> bool:
        """Memperbarui teks dan embedding dokumen, lalu menyinkronkan matriks."""
        for d in self.docs:
            if d.id == doc_id:
                d.text = new_text
                d.embedding = self._embed_text(new_text)
                self._rebuild_matrix()
                print(f"[VectorStore] Document '{doc_id}' updated.")
                return True
        print(f"[VectorStore] Document '{doc_id}' not found.")
        return False


# =====================================================================
# Exercise 4: Local SQLite Caching untuk Embeddings (embed_with_cache)
# =====================================================================
class SQLiteEmbeddingCache:
    def __init__(self, db_path: str = "Modul 8/embeddings_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """)

    def _generate_key(self, text: str, model: str) -> str:
        content = f"{model}:{text}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def get_embedding(self, text: str, model: str = EMBED_MODEL) -> np.ndarray:
        key = self._generate_key(text, model)

        # 1. Cek Cache
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                print(f"[CACHE HIT] Retrieving embedding for: '{text[:30]}...'")
                return np.frombuffer(row[0], dtype=np.float32)

        # 2. Jika Cache MISS -> Panggil Ollama
        print(f"[CACHE MISS] Requesting embedding from Ollama for: '{text[:30]}...'")
        resp = ollama.embed(model=model, input=[f"search_document: {text}"])
        vec = np.array(resp["embeddings"][0], dtype=np.float32)

        # 3. Simpan ke SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO cache (key, embedding) VALUES (?, ?)", (key, vec.tobytes()))

        return vec


# =====================================================================
# MAIN EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    print("=== 8.8 Module 08 Exercises Demo ===\n")

    # --- Test Exercise 1 ---
    print("--- 1. Testing DuplicateDetector ---")
    corpus_ex1 = [
        "Retrieval-Augmented Generation is a technique combining search with LLMs.",
        "Retrieval-Augmented Generation combines external search with LLMs.",  # Near Duplicate
        "Python is a high-level programming language.",
    ]
    detector = DuplicateDetector(threshold=0.88)
    dups = detector.find_duplicates(corpus_ex1)
    for i, j, score in dups:
        print(f"Found Duplicate pair (score={score:.4f}):\n  Text 1: '{corpus_ex1[i]}'\n  Text 2: '{corpus_ex1[j]}'\n")

    # --- Test Exercise 2 ---
    print("--- 2. Testing HybridSearch ---")
    hybrid = HybridSearch(corpus_ex1)
    results = hybrid.search(query="Python programming language", alpha=0.5, top_k=2)
    for doc, score in results:
        print(f"  [Score: {score:.4f}] {doc}")
    print()

    # --- Test Exercise 3 ---
    print("--- 3. Testing DynamicVectorStore (Delete/Update) ---")
    store = DynamicVectorStore()
    store.add("d1", "The quick brown fox jumps over the lazy dog.")
    store.add("d2", "Artificial Intelligence is transforming industries.")
    print("Initial docs count:", len(store.docs))

    store.update("d1", "A fast dark-colored fox jumps over a sleepy hound.")
    store.delete("d2")
    print("Final docs count:", len(store.docs), "\n")

    # --- Test Exercise 4 ---
    print("--- 4. Testing SQLiteEmbeddingCache ---")
    cache = SQLiteEmbeddingCache()
    sample_phrase = "Caching embeddings saves time and API costs."

    # Call 1 (Cache Miss)
    vec1 = cache.get_embedding(sample_phrase)
    # Call 2 (Cache Hit)
    vec2 = cache.get_embedding(sample_phrase)
    print("Are vectors identical?", np.array_equal(vec1, vec2))