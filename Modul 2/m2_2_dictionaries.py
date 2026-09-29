# 1. Membangun LLM API Payload (Nested Dictionary & List)
payload: dict = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "temperature": 0.7,
    "messages": [
        {"role": "user", "content": "What is attention in transformers?"}
    ],
}

# 2. Akses Nilai
print("--- 1. Dictionary Access ---")
print("Model            :", payload["model"])
# get() mengembalikan nilai default (1.0) jika key "top_p" tidak ada
print("top_p (default)  :", payload.get("top_p", 1.0))
print()

# 3. Update Nilai
payload["temperature"] = 0.3
payload.update({"stream": True, "top_k": 40})

print("--- 2. Iteration (Non-list values) ---")
for key, value in payload.items():
    if not isinstance(value, list):
        print(f"{key:<15} : {value}")
print()

# 4. Pengecekan Keberadaan Key (Membership Check)
print("--- 3. Membership Check ---")
print("Is 'stream' in payload? :", "stream" in payload)  # True
print("Is 'top_p' in payload?  :", "top_p" in payload)   # False