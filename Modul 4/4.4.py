# --- LATIHAN 1: Simpan dan Muat Riwayat Percakapan (JSON) ---
import json
import pathlib

def save_conversation(history: list[dict], path: str) -> None:
    pathlib.Path(path).write_text(json.dumps(history, indent=2), encoding="utf-8")

def load_conversation(path: str) -> list[dict]:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

print("--- Hasil Latihan 1 ---")
sample_history = [{"role": "user", "content": "Halo AI!"}, {"role": "assistant", "content": "Halo! Ada yang bisa dibantu?"}]
save_conversation(sample_history, "chat_history.json")
loaded_history = load_conversation("chat_history.json")
print("Berhasil dimuat dari file:", loaded_history)


# --- LATIHAN 2: Async Endpoints Fetcher dengan Httpx ---
import asyncio
import httpx

async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, float]]:
    """Mengambil beberapa URL secara konkuren dan mengembalikan (url, status_code, response_time_ms)"""
    async def fetch_one(client, url):
        start_time = asyncio.get_event_loop().time()
        try:
            response = await client.get(url, timeout=5.0)
            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
            return (url, response.status_code, elapsed)
        except Exception:
            elapsed = (asyncio.get_event_loop().time() - start_time) * 1000
            return (url, 0, elapsed) # 0 menandakan gagal/timeout

    async with httpx.AsyncClient() as client:
        tasks = [fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)

print("\n--- Hasil Latihan 2 ---")
test_urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/status/200",
    "https://httpbin.org/json"
]
# Jalankan di Colab menggunakan await
results_latency = await compare_endpoints(test_urls)
for res in results_latency:
    print(f"URL: {res[0]} | Status: {res[1]} | Waktu: {res[2]:.2f} ms")


# --- LATIHAN 3: Config Loader dengan Env Overrides ---
def load_config_with_env(json_path: str) -> dict:
    path = pathlib.Path(json_path)
    # 1. Baca dari file JSON jika ada, jika tidak buat default
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        config = {"model": "gpt-4o", "temperature": 0.7}

    # 2. Timpa dengan Environment Variable jika tersedia (Env mengambil prioritas utama)
    if os.getenv("AI_MODEL"):
        config["model"] = os.getenv("AI_MODEL")
    if os.getenv("AI_TEMPERATURE"):
        config["temperature"] = float(os.getenv("AI_TEMPERATURE"))

    return config

print("\n--- Hasil Latihan 3 ---")
pathlib.Path("config.json").write_text(json.dumps({"model": "claude-3-opus", "temperature": 0.5}), encoding="utf-8")
print("Config awal:", load_config_with_env("config.json"))


# --- LATIHAN 4: Thread-Safe CSV Log Writer ---
import threading
from datetime import datetime

class CSVLogWriter:
    def __init__(self, filepath: str):
        self.filepath = pathlib.Path(filepath)
        self.lock = threading.Lock()

        # Buat header jika file belum ada
        if not self.filepath.exists():
            with self.filepath.open("w", encoding="utf-8") as f:
                f.write("timestamp,model,input_tokens,output_tokens,latency_ms\n")

    def log(self, model: str, input_tokens: int, output_tokens: int, latency_ms: float):
        # Menggunakan thread lock agar aman saat diakses banyak thread sekaligus
        with self.lock:
            timestamp = datetime.utcnow().isoformat()
            line = f"{timestamp},{model},{input_tokens},{output_tokens},{latency_ms}\n"
            with self.filepath.open("a", encoding="utf-8") as f:
                f.write(line)

print("\n--- Hasil Latihan 4 ---")
logger = CSVLogWriter("llm_audit.csv")
logger.log("gpt-4o", 120, 45, 320.4)
logger.log("claude-sonnet-4-5", 200, 80, 410.1)
print("Log berhasil ditulis dengan aman ke 'llm_audit.csv'.")
print("Isi file log:", pathlib.Path("llm_audit.csv").read_text(encoding="utf-8"))
