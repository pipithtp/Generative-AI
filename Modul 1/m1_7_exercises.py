import time
from functools import wraps
from typing import Callable, Any

# =====================================================================
# Exercise 1: token_cost Function with Known Model Lookup & Validation
# =====================================================================
def token_cost(tokens: int, model: str) -> float:
    """Menghitung perkiraan biaya pemanggilan API berdasarkan jumlah token dan nama model.
    
    Harga per 1.000 (1K) token disimpan dalam dictionary internal.
    """
    costs_per_1k = {
        "gpt-4o": 0.005,
        "claude-sonnet-4-5": 0.003,
        "qwen2.5:3b": 0.0,  # Lokal via Ollama (Gratis)
    }

    if model not in costs_per_1k:
        raise ValueError(f"Unknown model '{model}'. Supported models: {list(costs_per_1k.keys())}")

    return (tokens / 1000.0) * costs_per_1k[model]


# =====================================================================
# Exercise 2: Retry Decorator (using functools.wraps)
# =====================================================================
def retry(max_attempts: int = 3, delay: float = 0.1) -> Callable:
    """Decorator yang mencoba ulang eksekusi fungsi hingga max_attempts kali jika terjadi Exception."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        print(f"[Retry Decorator] Attempt {attempt}/{max_attempts} failed. Max retries reached.")
                        raise e
                    print(f"[Retry Decorator] Attempt {attempt}/{max_attempts} failed ({e}). Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


# Fungsi Uji Coba untuk Exercise 2 (Gagal 2 kali pertama)
_attempts_counter = 0

@retry(max_attempts=4, delay=0.01)
def unreliable_api_call() -> str:
    global _attempts_counter
    _attempts_counter += 1
    if _attempts_counter < 3:
        raise ConnectionError(f"Network error on attempt {_attempts_counter}")
    return "API response received successfully!"


# =====================================================================
# Exercise 3: temperature_label Function
# =====================================================================
def temperature_label(t: float) -> str:
    """Memetakan nilai temperature ke label deskriptif."""
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Temperature {t} is outside valid range [0.0, 1.0]")

    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"


# =====================================================================
# Exercise 4: Parse String Without Regex
# =====================================================================
def parse_token_info(text: str) -> tuple[int, float]:
    """Menguraikan string '128000 tokens, 0.005 USD per 1K' tanpa RegEx.
    
    Mengembalikan tuple: (token_count: int, cost_per_1k: float)
    """
    # Contoh teks: "128000 tokens, 0.005 USD per 1K"
    parts = text.split(",")  # ["128000 tokens", " 0.005 USD per 1K"]
    
    # 1. Ekstraksi token count dari bagian pertama
    token_part = parts[0].strip()  # "128000 tokens"
    tokens_str = token_part.split()[0]  # "128000"
    tokens = int(tokens_str)

    # 2. Ekstraksi cost dari bagian kedua
    cost_part = parts[1].strip()  # "0.005 USD per 1K"
    cost_str = cost_part.split()[0]  # "0.005"
    cost = float(cost_str)

    return tokens, cost


# =====================================================================
# MAIN EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    print("=== Exercise 1: token_cost ===")
    print(f"Cost gpt-4o (10k tokens)        : ${token_cost(10000, 'gpt-4o'):.4f}")
    print(f"Cost qwen2.5:3b (10k tokens)     : ${token_cost(10000, 'qwen2.5:3b'):.4f}")
    try:
        token_cost(1000, "unknown-model")
    except ValueError as e:
        print(f"Caught expected error           : {e}\n")

    print("=== Exercise 2: Retry Decorator ===")
    result = unreliable_api_call()
    print(f"Result                          : {result}\n")

    print("=== Exercise 3: temperature_label ===")
    print(f"Temp 0.2  : {temperature_label(0.2)}")
    print(f"Temp 0.5  : {temperature_label(0.5)}")
    print(f"Temp 0.9  : {temperature_label(0.9)}")
    try:
        temperature_label(1.5)
    except ValueError as e:
        print(f"Caught expected error           : {e}\n")

    print("=== Exercise 4: Parse String Without Regex ===")
    sample_raw_string = "128000 tokens, 0.005 USD per 1K"
    parsed_tokens, parsed_cost = parse_token_info(sample_raw_string)
    print(f"Input string                    : '{sample_raw_string}'")
    print(f"Parsed Tokens (int)             : {parsed_tokens} (Type: {type(parsed_tokens).__name__})")
    print(f"Parsed Cost (float)             : {parsed_cost} (Type: {type(parsed_cost).__name__})")