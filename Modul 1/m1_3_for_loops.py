models: list[str] = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]

# 1. Basic Iteration
print("--- 1. Basic Iteration ---")
for model in models:
    print(f"Checking: {model}")
print()

# 2. Iterasi dengan Indeks menggunakan enumerate()
print("--- 2. Enumerate (Index & Value) ---")
for i, model in enumerate(models, start=1):
    print(f"{i}. {model}")
print()

# 3. Iterasi Key-Value Pair pada Dictionary
print("--- 3. Dictionary Items Iteration ---")
token_limits: dict[str, int] = {
    "gpt-4o": 128000,
    "claude-sonnet-4-5": 200000,
    "gemini-1.5-pro": 1000000,
}

for model, limit in token_limits.items():
    print(f"{model:<20} : {limit:,} tokens")