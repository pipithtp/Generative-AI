import math

# 1. Underscores untuk angka besar (Readability)
MAX_CONTEXT_WINDOW: int = 1_000_000  # Sama dengan 1000000

# 2. Integer & Float Arithmetic
tokens_used: int = 450
tokens_limit: int = 1024

remaining_tokens: int = tokens_limit - tokens_used  # 574
cost_per_token: float = 0.000003
total_cost: float = tokens_used * cost_per_token

print("--- Token & Cost Calculation ---")
print(f"Remaining Tokens : {remaining_tokens}")
print(f"Total Cost       : ${total_cost:.6f}")
print()

# 3. Integer Division (//) & Modulo (%)
batch_size: int = 100
batches: int = tokens_used // batch_size  # 4 batch penuh
leftover: int = tokens_used % batch_size   # 50 token tersisa

print("--- Batch Calculations ---")
print(f"Full Batches     : {batches}")
print(f"Leftover Tokens  : {leftover}")
print()

# 4. Built-in Math Module
print("--- Math Module Operations ---")
print(f"Log2 of 512      : {math.log2(512)}")  # 9.0
print(f"Ceil of 3.1      : {math.ceil(3.1)}")   # 4