import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def embed_texts(texts: list[str], input_type: str = "document") -> np.ndarray:
    """Mengubah daftar teks menjadi vektor ter-normalisasi menggunakan Ollama."""
    prefix = "search_document: " if input_type == "document" else "search_query: "
    prefixed_texts = [f"{prefix}{t}" for t in texts]

    resp = ollama.embed(model=EMBED_MODEL, input=prefixed_texts)
    vecs = np.array(resp["embeddings"], dtype=np.float32)
    
    # Normalisasi vektor ke unit length
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        """Embed dan simpan dokumen ke memori."""
        embeddings = embed_texts([d.text for d in docs], input_type="document")
        for doc, emb in zip(docs, embeddings):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Lakukan pre-filtering metadata sebelum ranking kemiripan vektor."""
        # 1. Apply pre-filter
        candidates = self._docs if filter_fn is None else [d for d in self._docs if filter_fn(d)]
        if not candidates:
            return []

        # 2. Embed kueri
        q_vec = embed_texts([query], input_type="query")[0]

        # 3. Hitung Cosine Similarity via dot product
        matrix = np.array([d.embedding for d in candidates], dtype=np.float32)
        scores = matrix @ q_vec

        # 4. Ambil Top-k
        k = min(k, len(candidates))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(candidates[i], float(scores[i])) for i in top_idx]


# ── DEMO EXECUTION ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = [
        FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
        FilteredDocument("a2", "Claude 3.5 Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
        FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
        FilteredDocument("a4", "Claude Opus 4 is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
        FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
    ]

    print(f"--- Modul 8.7 Metadata Filtering Demo (Model: {EMBED_MODEL}) ---\n")

    fstore = FilteredVectorStore()
    fstore.add(docs)

    # 1. Search tanpa filter (Semua dokumen)
    print("=== All docs ===")
    results = fstore.search("which model is good at coding?", k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc.id} ({doc.metadata['category']}): {doc.text}")

    # 2. Search hanya dokumen Anthropic
    print("\n=== Anthropic only ===")
    results_anthropic = fstore.search(
        "which model is good at coding?",
        k=3,
        filter_fn=lambda d: d.metadata["category"] == "anthropic"
    )
    for doc, score in results_anthropic:
        print(f"  [{score:.4f}] {doc.id} ({doc.metadata['category']}): {doc.text}")

    # 3. Search hanya dokumen tahun >= 2024
    print("\n=== Year >= 2024 ===")
    results_recent = fstore.search(
        "which model is good at coding?",
        k=3,
        filter_fn=lambda d: d.metadata["year"] >= 2024
    )
    for doc, score in results_recent:
        print(f"  [{score:.4f}] {doc.id} ({doc.metadata['year']}): {doc.text}")