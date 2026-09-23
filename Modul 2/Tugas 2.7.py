# --- LATIHAN 1: Filter & Sort API Responses ---
api_responses = [
    {"tokens": 150, "latency_ms": 450},
    {"tokens": 300, "latency_ms": 600},
    {"tokens": 50,  "latency_ms": 200},
    {"tokens": 500, "latency_ms": 480}
]
# Mengambil response di bawah 500ms, diurutkan berdasarkan jumlah token
fast_responses = sorted([r for r in api_responses if r["latency_ms"] < 500], key=lambda x: x["tokens"])
print("Hasil Latihan 1:", fast_responses)


# --- LATIHAN 2: Fungsi Statistik Percakapan ---
def conversation_stats(messages: list[dict]) -> dict:
    total = len(messages)
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

    total_words = sum(len(m.get("content", "").split()) for m in messages)
    avg_words = total_words / total if total > 0 else 0

    return {
        "total_messages": total,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": avg_words
    }

msgs = [
    {"role": "user", "content": "Tolong jelaskan RAG"},
    {"role": "assistant", "content": "RAG adalah Retrieval-Augmented Generation"}
]
print("\nHasil Latihan 2:", conversation_stats(msgs))


# --- LATIHAN 3: Generator Batch Items ---
def batch_items(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

data_dokumen = [1, 2, 3, 4, 5, 6, 7, 8]
print("\nHasil Latihan 3:", list(batch_items(data_dokumen, 3)))


# --- LATIHAN 4: Set Operations untuk Model AI ---
fast_models  = ["gpt-4o-mini", "gemini-1.5-flash", "claude-haiku"]
cheap_models = ["claude-haiku", "gpt-4o-mini", "llama-3-8b"]

# Mencari irisan (model yang cepat DAN murah)
fast_and_cheap = set(fast_models) & set(cheap_models)
print("\nHasil Latihan 4:", fast_and_cheap)
