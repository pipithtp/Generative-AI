import numpy as np
# ARRAYS AND DTYPES

# Creating arrays
scores = np.array(
    [0.91, 0.76, 0.88, 0.65, 0.95],
    dtype=np.float32
)

print(scores.dtype, scores.shape)
# float32 (5,)


# Zeros, ones, ranges
zeros = np.zeros((3, 4))
print("\nZeros:")
print(zeros)

rng_vals = np.arange(0, 1.0, 0.1)
print("\nRange values:")
print(rng_vals)

linspace = np.linspace(0, 1, 5)
print("\nLinspace:")
print(linspace)


# Random - seeded Generator for reproducibility
rng = np.random.default_rng(seed=42)

mock_embedding = rng.standard_normal(1536)

print(
    f"\nEmbedding shape: {mock_embedding.shape}, "
    f"mean: {mock_embedding.mean():.4f}"
)


# SHAPE, RESHAPE, INDEXING

# Simulate 4 document embeddings of dimension 8
rng = np.random.default_rng(42)

embeddings = rng.standard_normal((4, 8))

print("\nShape:", embeddings.shape)
print("First embedding:", embeddings[0])

print(
    "First 3 dims of all docs:\n",
    embeddings[:, :3]
)


# Reshape
flat = embeddings.flatten()

print("\nFlat shape:", flat.shape)

back = flat.reshape(4, 8)

print("Reshaped shape:", back.shape)


# Boolean indexing
similarity_scores = np.array(
    [0.91, 0.43, 0.78, 0.55]
)

above_threshold = embeddings[
    similarity_scores > 0.7
]

print(
    f"Docs above 0.7 similarity: "
    f"{above_threshold.shape[0]}"
)


# COSINE SIMILARITY

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """Compute cosine similarity between two vectors."""

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


def top_k_similar(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return indices and scores of the k most similar vectors."""

    # Normalise corpus rows
    norms = np.linalg.norm(
        corpus,
        axis=1,
        keepdims=True
    )

    norms = np.where(
        norms == 0,
        1,
        norms
    )

    normed = corpus / norms

    # Normalise query
    q_norm = np.linalg.norm(query)

    q_normed = query / (
        q_norm if q_norm > 0 else 1
    )

    # Matrix-vector multiplication
    # gives cosine similarities
    sims = normed @ q_normed

    top_idx = np.argsort(sims)[::-1][:k]

    return [
        (int(i), float(sims[i]))
        for i in top_idx
    ]


# TEST TOP-K SIMILARITY

rng = np.random.default_rng(42)

corpus = rng.standard_normal((10, 8))

query = rng.standard_normal(8)

results = top_k_similar(
    query,
    corpus,
    k=3
)

print("\nTop 3 similar documents:")

for idx, score in results:
    print(
        f"Doc {idx}: similarity = {score:.4f}"
    )