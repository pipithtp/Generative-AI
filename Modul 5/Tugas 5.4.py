import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class EvalResult:
    prompt: str
    model: str
    response: str
    latency_ms: float
    score: float = field(default=0.0)

def mock_llm_call(prompt: str, model: str) -> tuple[str, float]:
    """Simulasi panggilan LLM. Mengembalikan (response, latency_ms)."""
    rng = np.random.default_rng(abs(hash(prompt + model)) % 2**31)
    latency = rng.uniform(300, 700)
    response = f"[{model}] Answer regarding {prompt[:20]} with attention and model details."
    return response, latency

def score_response(response: str, expected_keywords: list[str]) -> float:
    """Penilaian sederhana berbasis keyword (0.0 - 1.0)."""
    found = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
    return found / len(expected_keywords) if expected_keywords else 0.0

# Menjalankan evaluasi
prompts = [
    ("What is a transformer?", ["attention", "model"]),
    ("Define RAG?", ["retrieval", "generation"]),
    ("What is fine-tuning?", ["training", "weights"]),
]
models = ["claude-sonnet-4-5", "gpt-4o"]

results: list[EvalResult] = []
for prompt, keywords in prompts:
    for model in models:
        response, latency = mock_llm_call(prompt, model)
        score = score_response(response, keywords)
        results.append(EvalResult(prompt, model, response, latency, score))

# Analisis dengan Pandas
df = pd.DataFrame([vars(r) for r in results])
summary = df.groupby("model").agg(
    avg_score   = ("score", "mean"),
    avg_latency = ("latency_ms", "mean"),
).round(3)

print("Hasil Evaluasi Model:\n", summary)

# Simpan ke CSV
df.to_csv("eval_results.csv", index=False)
print("\nBerhasil disimpan ke 'eval_results.csv'")
