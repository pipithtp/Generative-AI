import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# Prompt 1: Direct Prompt (Tanpa instruksi berfikir step-by-step)
DIRECT_PROMPT = "If a model costs $3.00 per million input tokens and $15.00 per million output tokens, and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?"

# Prompt 2: Explicit CoT Prompt
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens, what is the total cost in USD?
Think through this step by step before giving the final answer."""

# Prompt 3: Zero-shot CoT (Menambahkan tag instruksi eksplisit)
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX

Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

prompts = [
    ("Direct Prompt", DIRECT_PROMPT),
    ("CoT Prompt", COT_PROMPT),
    ("Zero-shot CoT", ZERO_SHOT_COT),
]

print(f"--- Modul 7.4 Chain-of-Thought Prompting Demo (Model: {MODEL_NAME}) ---\n")

for label, prompt in prompts:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    
    print(f"=== {label} ===")
    print(response["message"]["content"])
    print("-" * 50 + "\n")