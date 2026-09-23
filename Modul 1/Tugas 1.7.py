#Latihan 1: Fungsi Estimasi Biaya Token#

def token_cost(tokens: int, model: str) -> float:
    # Dictionary simulasi harga USD per 1000 token
    costs_per_1k = {
        "gpt-4o": 0.005,
        "claude-sonnet-4-5": 0.003,
        "gemini-1.5-pro": 0.002
    }

    # Mengecek apakah model ada di dalam dictionary
    if model not in costs_per_1k:
        raise ValueError(f"Model '{model}' is unknown.")

    # Menghitung total biaya
    return (tokens / 1000) * costs_per_1k[model]

# Tes fungsinya
print(f"Biaya 2500 token untuk GPT-4o: ${token_cost(2500, 'gpt-4o')}") 

#Latihan 2: Decorator Retry untuk Fungsi API#

import functools
import time

def retry(n: int, sleep_time: float):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Melakukan perulangan hingga n kali
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == n:
                        print(f"Percobaan ke-{attempt} gagal. Berhenti mencoba.")
                        raise e # Lempar error asli jika sudah batas maksimal

                    print(f"Percobaan ke-{attempt} gagal ({e}). Retrying dalam {sleep_time} detik...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

# Fungsi simulasi yang sengaja akan gagal di 2 panggilan pertama
@retry(n=3, sleep_time=1.0)
def flaky_api_call():
    if not hasattr(flaky_api_call, "calls"):
        flaky_api_call.calls = 0
    flaky_api_call.calls += 1

    if flaky_api_call.calls <= 2:
        raise ConnectionError("Jaringan tidak stabil")
    return "Berhasil memanggil API di percobaan terakhir!"

# Tes fungsinya
print(flaky_api_call())

#Latihan 3: Pengkategorian Suhu (Temperature) Model#

def temperature_label(t: float) -> str:
    # Memastikan suhu ada di rentang yang valid (0.0 sampai 1.0)
    if not (0.0 <= t <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"

# Tes fungsinya
print(f"Karakter suhu 0.2: {temperature_label(0.2)}")
print(f"Karakter suhu 0.5: {temperature_label(0.5)}")
print(f"Karakter suhu 0.9: {temperature_label(0.9)}")

# Kamu bisa hapus tanda '#' di bawah ini untuk melihat error jika suhu 1.5
# print(temperature_label(1.5))

#Latihan 4: Parsing String Data (Tanpa Regex)#

text = "128000 tokens, 0.005 USD per 1K"

# 1. Pisahkan string menjadi dua bagian berdasarkan tanda koma dan spasi
parts = text.split(", ")

# 2. Ambil bagian pertama ("128000 tokens"), lalu pisah spasi dan ambil kata pertama
tokens_str = parts[0].split(" ")[0]

# 3. Ambil bagian kedua ("0.005 USD per 1K"), lalu pisah spasi dan ambil angka pertama
cost_str = parts[1].split(" ")[0]

# 4. Konversi tipe datanya
token_count = int(tokens_str)
cost = float(cost_str)

# Tes hasilnya
print(f"Jumlah token : {token_count} (Tipe data: {type(token_count)})")
print(f"Harga        : {cost} (Tipe data: {type(cost)})")

