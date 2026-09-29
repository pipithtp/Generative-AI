import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]  # Minimal salah satu kata kunci harus ada di respons
    must_be_json: bool = False


def evaluate_prompt(system: str, cases: list[EvalCase]) -> dict:
    """Menjalankan prompt ke sekumpulan test cases dan mengembalikan pass rate + detail pengujian."""
    results = []
    
    for case in cases:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case.input_text},
            ],
            # Jika kasus membutuhkan JSON valid, aktifkan format="json" di Ollama
            format="json" if case.must_be_json else "",
        )
        
        text = response["message"]["content"].strip()

        # 1. Cek kecocokan kata kunci (Keyword Hit)
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)

        # 2. Cek validitas JSON jika diminta (JSON validity)
        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False

        passed = keyword_hit and json_valid
        
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })

    pass_rate = sum(r["passed"] for r in results) / len(results) if results else 0.0
    return {"pass_rate": pass_rate, "results": results}


# --- MAIN TEST EXECUTION ---
if __name__ == "__main__":
    print(f"--- Modul 7.7 Prompt Evaluation Demo (Model: {MODEL_NAME}) ---\n")

    CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

    test_cases = [
        EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
        EvalCase("Write a product description for headphones.", ["GENERATION"]),
        EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
        EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
        EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
    ]

    report = evaluate_prompt(CLASSIFY_SYSTEM, test_cases)

    print(f"Pass rate: {report['pass_rate']:.0%}\n")
    for r in report["results"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['input']!r} → {r['response_preview']!r}")