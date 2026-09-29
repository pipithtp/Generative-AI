import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

# Few-shot System Prompt dengan contoh eksplisit
FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI benchmark result string, extract: model name, task, and score as a JSON object.

Examples:
Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Return ONLY the JSON object. No explanation."""

test_inputs = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]

print(f"--- Modul 7.3 Few-Shot Prompting Demo (Model: {MODEL_NAME}) ---\n")

# Iterasi pengujian untuk setiap teks input
for text in test_inputs:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": FEW_SHOT_SYSTEM},
            {"role": "user", "content": text},
        ],
    )
    
    print(f"Input : {text}")
    print(f"Output: {response['message']['content']}\n")