import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# --- 1. Token Counting via Ollama ---
def count_prompt_tokens(prompt: str, system_prompt: str = "") -> int:
    """Estimasi jumlah input token menggunakan respons Ollama."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Melakukan request singkat untuk mendapatkan prompt_eval_count (input tokens)
    response = ollama.chat(model=MODEL_NAME, messages=messages)
    return response.get("prompt_eval_count", 0)


# --- 2. Cost Estimator (Per 1 Million Tokens) ---
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5": {"input": 15.00, "output": 75.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "qwen2.5:3b (Ollama Local)": {"input": 0.00, "output": 0.00},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


# --- 3. Context Window Limits Check ---
CONTEXT_LIMITS = {
    "claude-sonnet-4-5": 200_000,
    "claude-opus-4-5": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
    "qwen2.5:3b": 32_768,
}

def fits_in_context(model: str, token_count: int, reserve_for_output: int = 2048) -> bool:
    """Memeriksa apakah jumlah token muat dalam context window model."""
    limit = CONTEXT_LIMITS.get(model, 32_768)
    return token_count + reserve_for_output <= limit


# --- MAIN TEST EXECUTION ---
if __name__ == "__main__":
    system_text = "You are a concise assistant."
    user_prompt = "Explain the transformer architecture."

    print(f"--- 6.6 Token Management Demo (Model: {MODEL_NAME}) ---\n")

    # 1. Tes Menghitung Token Input
    input_tokens = count_prompt_tokens(user_prompt, system_text)
    print(f"Estimated input tokens: {input_tokens}")

    # 2. Tes Simulasi Estimasi Biaya (Jika pakai model berbayar vs Ollama)
    cost_claude = estimate_cost("claude-sonnet-4-5", input_tokens=500, output_tokens=300)
    cost_qwen = estimate_cost("qwen2.5:3b (Ollama Local)", input_tokens=500, output_tokens=300)

    print(f"Estimated cost (Claude Sonnet 4.5): ${cost_claude:.6f}")
    print(f"Estimated cost (Qwen Local)        : ${cost_qwen:.6f}")

    # 3. Tes Context Window Check
    is_safe = fits_in_context(MODEL_NAME, token_count=input_tokens, reserve_for_output=2048)
    print(f"Apakah prompt muat di context window {MODEL_NAME}? -> {is_safe}")