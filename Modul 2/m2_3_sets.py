# 1. Deduplikasi Dokumen yang Ditemukan (Retrieval Deduplication)
retrieved_doc_ids: list[str] = ["doc_3", "doc_1", "doc_3", "doc_7", "doc_1"]
unique_ids: set[str] = set(retrieved_doc_ids)

print("--- 1. Deduplication ---")
print("Original List :", retrieved_doc_ids)
print("Unique Set    :", unique_ids)
print()

# 2. Operasi Set untuk Analisis Topik Model
gpt4_topics: set[str] = {"coding", "math", "reasoning", "vision"}
claude_topics: set[str] = {"coding", "writing", "reasoning", "safety"}

both: set[str] = gpt4_topics & claude_topics      # Intersection
either: set[str] = gpt4_topics | claude_topics    # Union
gpt_only: set[str] = gpt4_topics - claude_topics  # Difference

print("--- 2. Topic Analysis Operations ---")
print("Both (Intersection) :", both)
print("Either (Union)      :", either)
print("GPT only (Diff)     :", gpt_only)