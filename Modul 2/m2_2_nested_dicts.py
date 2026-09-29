from collections import defaultdict

# Inisialisasi defaultdict bersarang untuk mencatat token input & output per model
usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

# Menambahkan penggunaan token secara otomatis tanpa inisialisasi kunci manual
usage["claude-sonnet-4-5"]["input"] += 350
usage["claude-sonnet-4-5"]["output"] += 210

usage["gpt-4o"]["input"] += 420
usage["gpt-4o"]["output"] += 180

# Penambahan akumulatif kedua untuk model yang sama
usage["claude-sonnet-4-5"]["input"] += 150

print("--- Token Usage Tracker ---")
for model, counts in usage.items():
    total = counts["input"] + counts["output"]
    print(f"{model:<20} : {total:>5} total tokens (Input: {counts['input']}, Output: {counts['output']})")