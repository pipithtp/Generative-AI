import os
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from string import Formatter
from typing import Any
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


# =====================================================================
# Exercise 1: System Prompt Evaluation (Basic, Intermediate, Expert)
# =====================================================================
BASIC_PROMPT = "You are a code reviewer. Review the following code."

INTERMEDIATE_PROMPT = """You are an experienced Python developer. Review the code for bugs, readability, and performance.
Provide constructive feedback in bullet points."""

EXPERT_PROMPT = """You are a Principal Security & Systems Architect reviewing critical production code.
Identify security vulnerabilities, architectural flaws, and performance bottlenecks.
Format: Numbered list with Issue -> Impact -> Corrected Code."""

CODE_SNIPPETS = [
    "def add(a, b): return a + b",
    "def get_user(id): return db.query(f'SELECT * FROM users WHERE id={id}')",
    "def read_file(path): f = open(path); return f.read()",
    "def calculate(items): return [x * 1.1 for x in items if x > 0]",
    "import os; secret = os.getenv('API_KEY', 'default_secret_key_123')"
]

def run_exercise_1():
    print("=== Exercise 1: System Prompt Comparison ===")
    prompts = [("Basic", BASIC_PROMPT), ("Intermediate", INTERMEDIATE_PROMPT), ("Expert", EXPERT_PROMPT)]
    
    for name, prompt in prompts:
        print(f"\n--- Testing System Prompt: {name} ---")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Review this snippet:\n{CODE_SNIPPETS[1]}"}  # Tes snippet SQL injection
            ]
        )
        print(response["message"]["content"][:300] + "...\n")


# =====================================================================
# Exercise 2: PromptLibrary with JSON Save/Load & Version Tracking
# =====================================================================
@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str = "1.0"
    last_eval_pass_rate: float = 0.0

    def render(self, **kwargs: Any) -> tuple[str, str]:
        return self.system.format(**kwargs), self.user.format(**kwargs)


class PromptLibrary:
    def __init__(self):
        self.templates: dict[str, PromptTemplate] = {}

    def add_template(self, template: PromptTemplate):
        self.templates[template.name] = template

    def save_to_json(self, file_path: str):
        data = {k: asdict(v) for k, v in self.templates.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Library] Saved {len(data)} templates to {file_path}")

    def load_from_json(self, file_path: str):
        if not Path(file_path).exists():
            print(f"[ERROR] File {file_path} not found.")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.templates = {k: PromptTemplate(**v) for k, v in data.items()}
        print(f"[Library] Loaded {len(self.templates)} templates from {file_path}")


# =====================================================================
# Exercise 3: Automatic JSON Repair (safe_json_parse)
# =====================================================================
def safe_json_parse(text: str, model: str = MODEL_NAME) -> dict:
    """Mencoba parsing JSON, membersihkan markdown fences, dan meminta model memperbaiki jika masih invalid."""
    # 1. Coba json.loads biasa
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Bersihkan markdown fences (```json ... ```)
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Minta model memperbaiki JSON secara otomatis
    print("[safe_json_parse] Output bukan JSON valid. Meminta Ollama melakukan perbaikan...")
    fix_prompt = f"Fix the following broken string so that it becomes valid raw JSON only. Do not add explanation:\n\n{text}"
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": fix_prompt}],
        format="json"
    )
    
    fixed_text = response["message"]["content"].strip()
    return json.loads(fixed_text)


# =====================================================================
# Exercise 4: CoT Prompt for LLM Score Ranking & Recommendation
# =====================================================================
COT_RANKING_PROMPT = """Analyze the provided LLM evaluation scores across tasks. 
Follow this exact CoT format:

<thinking>
1. Calculate average or weighted score per model across tasks.
2. Rank the models from highest to lowest based on calculations.
3. Formulate key strengths/weaknesses for recommendation.
</thinking>

<recommendation>
Rankings: 1. ModelA, 2. ModelB, ...
Write exactly 2 sentences of actionable recommendation.
</recommendation>"""

TEST_INPUT_DATA = [
    "Model Alpha: MMLU 80, GSM8K 70, HumanEval 60. Model Beta: MMLU 75, GSM8K 85, HumanEval 80.",
    "Model X: MMLU 90, GSM8K 92, HumanEval 88. Model Y: MMLU 95, GSM8K 60, HumanEval 70.",
    "Qwen-3B: MMLU 65, GSM8K 60, HumanEval 55. Llama-8B: MMLU 72, GSM8K 78, HumanEval 70."
]

def run_exercise_4():
    print("=== Exercise 4: CoT Ranking & Recommendation ===")
    for idx, data in enumerate(TEST_INPUT_DATA, 1):
        print(f"\n--- Test Input {idx} ---")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": COT_RANKING_PROMPT},
                {"role": "user", "content": f"Scores:\n{data}"}
            ]
        )
        print(response["message"]["content"])


# =====================================================================
# MAIN EXECUTION DEMO
# =====================================================================
if __name__ == "__main__":
    # --- Demo Ex 1 ---
    run_exercise_1()

    # --- Demo Ex 2 ---
    print("=== Exercise 2: Testing PromptLibrary ===")
    lib = PromptLibrary()
    lib.add_template(PromptTemplate("code_review", EXPERT_PROMPT, "Snippet: {code}", version="2.0", last_eval_pass_rate=0.95))
    json_path = "Modul 7/prompt_library.json"
    lib.save_to_json(json_path)
    lib.load_from_json(json_path)
    print("Loaded template version:", lib.templates["code_review"].version, "\n")

    # --- Demo Ex 3 ---
    print("=== Exercise 3: Testing safe_json_parse ===")
    broken_json = "```json\n{'model': 'qwen', 'score': 95,}```"  # Malformed JSON (quotes & trailing comma)
    parsed = safe_json_parse(broken_json)
    print("Successfully parsed repaired JSON:", parsed, "\n")

    # --- Demo Ex 4 ---
    run_exercise_4()