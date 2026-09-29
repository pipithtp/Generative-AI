messages: list[dict[str, str]] = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    {"role": "user", "content": "Give an example."},
]

# 1. Menyarikan hanya pesan dari user
user_messages: list[str] = [m["content"] for m in messages if m["role"] == "user"]

print("--- 1. User Messages Only ---")
print(user_messages)
print()

# 2. Menghitung jumlah kata tiap pesan
word_counts: list[int] = [len(m["content"].split()) for m in messages]

print("--- 2. Word Counts Per Message ---")
print(word_counts)  # [3, 5, 3]
print()

# 3. Meratakan (Flatten) List Bersarang
keywords: list[list[str]] = [["RAG", "retrieval"], ["LLM", "embedding"], ["vector"]]
flat_keywords: list[str] = [kw for group in keywords for kw in group]

print("--- 3. Flattened Keywords ---")
print(flat_keywords)