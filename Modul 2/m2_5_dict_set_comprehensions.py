models: list[str] = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]
context_windows: list[int] = [128_000, 200_000, 1_000_000]

# 1. Dict Comprehension menggunakan zip()
model_context: dict[str, int] = {m: c for m, c in zip(models, context_windows)}

print("--- 1. Model Context Dictionary ---")
print(model_context)
print()

# 2. Filtering pada Dict Comprehension (> 150K context)
large_context: dict[str, int] = {
    m: c for m, c in model_context.items() if c > 150_000
}

print("--- 2. Models with > 150K Context ---")
print(large_context)
print()

# 3. Set Comprehension untuk Mencari Panjang Kata Unik
sentence: str = "the quick brown fox jumps over the lazy dog"
unique_lengths: set[int] = {len(w) for w in sentence.split()}

print("--- 3. Unique Word Lengths (Set Comprehension) ---")
print(f"Set result    : {unique_lengths}")
print(f"Sorted result : {sorted(unique_lengths)}")