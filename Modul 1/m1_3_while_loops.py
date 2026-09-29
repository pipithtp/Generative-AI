import time

MAX_RETRIES: int = 3

# --- Skenario 1: Berhasil di tengah iterasi (Trigger break) ---
print("--- Skenario 1: Berhasil di Attempt ke-2 ---")
attempt: int = 0

while attempt < MAX_RETRIES:
    attempt += 1
    print(f"Attempt {attempt}...")
    
    # Simulasi berhasil di percobaaan ke-2
    if attempt == 2:
        print("Success! Stopping retries.\n")
        break
    
    time.sleep(0.1)
else:
    # Blok ini TIDAK akan berjalan karena loop di-break pada attempt 2
    print("All retries failed.")


# --- Skenario 2: Gagal hingga percobaan habis (Tanpa break) ---
print("--- Skenario 2: Gagal Seluruh Retries ---")
attempt = 0

while attempt < MAX_RETRIES:
    attempt += 1
    print(f"Attempt {attempt}...")
    time.sleep(0.1)
    # Tidak ada break di sini
else:
    # Blok ini AKAN berjalan karena loop selesai hingga attempt < MAX_RETRIES bernilai False
    print("All retries failed. Triggering fallback logic!\n")