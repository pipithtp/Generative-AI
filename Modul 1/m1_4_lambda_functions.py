responses: list[dict] = [
    {"model": "gpt-4o", "tokens": 540},
    {"model": "claude-sonnet-4-5", "tokens": 310},
    {"model": "gemini-1.5-pro", "tokens": 820},
]

# Mengurutkan list dictionary berdasarkan jumlah token (Ascending)
sorted_responses = sorted(responses, key=lambda r: r["tokens"])

print("--- Data Terurut Berdasarkan Jumlah Token (Ascending) ---")
for r in sorted_responses:
    print(f"{r['model']:<20} : {r['tokens']} tokens")