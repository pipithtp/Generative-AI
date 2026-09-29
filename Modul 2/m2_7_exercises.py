from typing import Generator, Any, Iterable


# =====================================================================
# Exercise 1: Comprehension & Sorting for Response Dicts
# =====================================================================
def filter_and_sort_responses(responses: list[dict]) -> list[dict]:
    """Mengembalikan respon dengan latency < 500ms, diurutkan berdasarkan token terkecil."""
    # List comprehension untuk memfilter, lalu diurutkan menggunakan sorted() dan lambda
    filtered = [r for r in responses if r.get("latency_ms", 0) < 500]
    return sorted(filtered, key=lambda r: r["tokens"])


# =====================================================================
# Exercise 2: conversation_stats Function
# =====================================================================
def conversation_stats(messages: list[dict]) -> dict[str, Any]:
    """Menganalisis riwayat pesan dan mengembalikan statistik agregat."""
    if not messages:
        return {
            "total_messages": 0,
            "user_turns": 0,
            "assistant_turns": 0,
            "avg_words_per_message": 0.0,
        }

    total_messages = len(messages)
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

    # Menghitung total kata pada setiap pesan
    total_words = sum(len(m.get("content", "").split()) for m in messages)
    avg_words = total_words / total_messages

    return {
        "total_messages": total_messages,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": round(avg_words, 2),
    }


# =====================================================================
# Exercise 3: Batch Items Generator
# =====================================================================
def batch_items(items: Iterable[Any], batch_size: int) -> Generator[list[Any], None, None]:
    """Yields list berukuran batch_size dari iterable apapun, termasuk partial batch terakhir."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# =====================================================================
# Exercise 4: Set Operations (Fast & Cheap Models)
# =====================================================================
def find_fast_and_cheap(fast_models: list[str], cheap_models: list[str]) -> set[str]:
    """Mencari model yang berada di kedua list menggunakan set intersection (&)."""
    return set(fast_models) & set(cheap_models)


# =====================================================================
# MAIN EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    print("=== Exercise 1: Filter & Sort Responses ===")
    mock_responses = [
        {"id": 1, "tokens": 450, "latency_ms": 320},
        {"id": 2, "tokens": 120, "latency_ms": 610},  # Ditolak (> 500ms)
        {"id": 3, "tokens": 200, "latency_ms": 150},
        {"id": 4, "tokens": 80, "latency_ms": 480},
    ]
    result_ex1 = filter_and_sort_responses(mock_responses)
    for r in result_ex1:
        print(f"ID: {r['id']} | Tokens: {r['tokens']} | Latency: {r['latency_ms']}ms")
    print()

    print("=== Exercise 2: Conversation Stats ===")
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am an AI assistant, ready to help you."},
        {"role": "user", "content": "Explain vector embeddings briefly."},
    ]
    stats = conversation_stats(conversation)
    for k, v in stats.items():
        print(f"{k:<22} : {v}")
    print()

    print("=== Exercise 3: Batch Generator ===")
    sample_items = [f"doc_{i}" for i in range(1, 10)]  # 9 items
    for idx, batch in enumerate(batch_items(sample_items, batch_size=4), start=1):
        print(f"Batch {idx} (size {len(batch)}): {batch}")
    print()

    print("=== Exercise 4: Fast & Cheap Models ===")
    fast = ["gpt-4o-mini", "claude-haiku", "qwen2.5:3b", "gemini-flash"]
    cheap = ["qwen2.5:3b", "gemini-flash", "deepseek-r1", "gpt-4o-mini"]
    fast_and_cheap = find_fast_and_cheap(fast, cheap)
    print("Models that are both Fast & Cheap:", fast_and_cheap)