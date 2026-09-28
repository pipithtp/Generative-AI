import os
import time
import asyncio
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


# =====================================================================
# Exercise 1: retry_on_rate_limit with Exponential Backoff
# =====================================================================
def retry_on_rate_limit(call_func, max_retries: int = 5, initial_delay: float = 1.0):
    """
    Menangkap RateLimitError (atau Exception umum) dan melakukan retry 
    secara otomatis dengan jeda waktu eksponensial (1s, 2s, 4s, 8s, dst).
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return call_func()
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Mencapai retries maksimum ({max_retries}). Gagal!")
                raise e
            print(f"[WARNING] Terkena limit/error ({e}). Retry dalam {delay}s... (Attempt {attempt}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # Exponential backoff


# =====================================================================
# Exercise 2: TokenBudgetManager Class & Custom Exception
# =====================================================================
class BudgetExceededException(Exception):
    """Custom exception ketika penggunaan token melebihi anggaran."""
    pass


class TokenBudgetManager:
    def __init__(self, max_token_budget: int):
        self.max_token_budget = max_token_budget
        self.total_tokens_used = 0

    def add_usage(self, input_tokens: int, output_tokens: int):
        tokens_in_call = input_tokens + output_tokens
        if self.total_tokens_used + tokens_in_call > self.max_token_budget:
            raise BudgetExceededException(
                f"Anggaran token terlampaui! Digunakan: {self.total_tokens_used}, "
                f"Percobaan: +{tokens_in_call}, Batas Maksimum: {self.max_token_budget}"
            )
        self.total_tokens_used += tokens_in_call
        print(f"[Budget] +{tokens_in_call} tokens. Total Penggunaan: {self.total_tokens_used}/{self.max_token_budget}")


# =====================================================================
# Exercise 3: Concurrent Model Comparison with asyncio & pandas
# =====================================================================
async def fetch_model_response(model: str, prompt: str) -> dict:
    start_time = time.perf_counter()

    # Memanggil Ollama secara asinkronus menggunakan asyncio
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "model": model,
        "response_text": response["message"]["content"],
        "input_tokens": response.get("prompt_eval_count", 0),
        "output_tokens": response.get("eval_count", 0),
        "latency_ms": round(latency_ms, 2)
    }


async def compare_models_async(prompt: str, models: list[str]) -> pd.DataFrame:
    """Memanggil beberapa model secara concurrent via asyncio.gather dan mengembalikan DataFrame."""
    tasks = [fetch_model_response(model, prompt) for model in models]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


# =====================================================================
# Exercise 4: Real-time stream_to_file Function
# =====================================================================
def stream_to_file(prompt: str, output_path: str, model: str = MODEL_NAME):
    """Menerima token streaming dan menulisnya langsung ke file secara real-time."""
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print(f"\nWriting stream to '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in stream:
            text = chunk["message"]["content"]
            f.write(text)
            f.flush()  # Paksa penulisan buffer langsung ke disk
            print(text, end="", flush=True)

    print(f"\n[Selesai] Output tersimpan di: {output_path}")


# =====================================================================
# DEMO EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("=== 6.8 Module 06 Exercises Demo ===\n")

    # --- Test Exercise 1 ---
    print("--- 1. Testing retry_on_rate_limit ---")
    def sample_call():
        return ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": "Hi!"}])

    res = retry_on_rate_limit(sample_call)
    print("Success response:", res["message"]["content"], "\n")

    # --- Test Exercise 2 ---
    print("--- 2. Testing TokenBudgetManager ---")
    budget_mgr = TokenBudgetManager(max_token_budget=100)
    try:
        budget_mgr.add_usage(input_tokens=20, output_tokens=30)  # Total 50
        budget_mgr.add_usage(input_tokens=30, output_tokens=30)  # Total 110 (Akan melebihi limit)
    except BudgetExceededException as e:
        print("Caught Exception:", e, "\n")

    # --- Test Exercise 3 ---
    print("--- 3. Testing compare_models (asyncio) ---")
    # Membandingkan model qwen2.5:3b (bisa ditambahkan 'llava' jika ada)
    df_result = asyncio.run(compare_models_async(
        prompt="Explain vector search in 1 sentence.",
        models=[MODEL_NAME]
    ))
    print(df_result[["model", "input_tokens", "output_tokens", "latency_ms"]], "\n")

    # --- Test Exercise 4 ---
    print("--- 4. Testing stream_to_file ---")
    output_file = Path("Modul 6/exercise_output.txt")
    stream_to_file("Explain what an embedding is in 2 sentences.", str(output_file))