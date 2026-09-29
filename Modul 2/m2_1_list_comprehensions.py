# Sample data hasil retrieval dokumen beserta skor relevansinya
search_results: list[dict] = [
    {"doc_id": "d01", "score": 0.92, "text": " RAG connects LLMs with external data. "},
    {"doc_id": "d02", "score": 0.65, "text": " Fine-tuning adapts model weights. "},
    {"doc_id": "d03", "score": 0.88, "text": " Vector databases store embeddings. "},
    {"doc_id": "d04", "score": 0.45, "text": " Prompt engineering optimizes input. "},
]

# 1. Mapping: Mengambil daftar teks dan membersihkan spasi tambahan (.strip())
clean_texts: list[str] = [res["text"].strip() for res in search_results]

print("--- 1. Cleaned Texts ---")
for text in clean_texts:
    print(f"- {text}")
print()

# 2. Filtering: Hanya ambil dokumen dengan skor similarity >= 0.70
high_score_docs: list[dict] = [
    res for res in search_results if res["score"] >= 0.70
]

print("--- 2. High Score Documents (score >= 0.70) ---")
for doc in high_score_docs:
    print(f"[{doc['doc_id']}] Score: {doc['score']:.2f} | {doc['text'].strip()}")
print()

# 3. Combined Mapping & Filtering: Ambil ID dokumen yang skornya < 0.70
low_score_ids: list[str] = [
    res["doc_id"] for res in search_results if res["score"] < 0.70
]
print(f"Low score doc IDs: {low_score_ids}")