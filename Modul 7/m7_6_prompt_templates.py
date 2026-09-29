from dataclasses import dataclass, field
from string import Formatter
from typing import Any
import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


@dataclass
class PromptTemplate:
    """Reusable & versioned prompt template manager."""
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Otomatis mendeteksi variabel yang dibutuhkan dari string template
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        """Mengembalikan tuple (rendered_system, rendered_user)."""
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


# 2. Definisikan Template sebagai Konstanta Terpusat
QA_TEMPLATE = PromptTemplate(
    name="question_answering",
    version="1.2",
    system="""You are a {domain} expert. Answer questions accurately and concisely.
If you are unsure, say so.""",
    user="Question: {question}\n\nContext:\n{context}",
)

SUMMARY_TEMPLATE = PromptTemplate(
    name="document_summary",
    version="1.0",
    system="You are a technical writer. Summarise documents clearly for a {audience} audience.",
    user="Summarise the following in {max_sentences} sentences or fewer:\n\n{document}",
)


# 3. Main Execution
if __name__ == "__main__":
    print(f"--- Modul 7.6 Prompt Templates Demo (Model: {MODEL_NAME}) ---\n")

    # Render template QA
    system_rendered, user_rendered = QA_TEMPLATE.render(
        domain="machine learning",
        question="What is the vanishing gradient problem?",
        context="Gradients in deep neural networks are computed via backpropagation by chain rule. Small values multiplied together repeatedly cause gradients to vanish.",
    )

    print("Template Meta    :", f"{QA_TEMPLATE.name} (v{QA_TEMPLATE.version})")
    print("Required Vars    :", QA_TEMPLATE.required_vars)
    print("Rendered System  :", system_rendered)
    print("Rendered User    :", user_rendered)
    print("-" * 50)

    # Eksekusi dengan Ollama menggunakan hasil render template
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_rendered},
            {"role": "user", "content": user_rendered},
        ],
    )

    print("\n--- Response dari Ollama ---")
    print(response["message"]["content"])