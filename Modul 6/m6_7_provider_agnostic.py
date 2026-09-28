from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import ollama
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


# --- Data Structures ---
@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


# --- Abstract Base Class ---
class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        pass


# --- 1. Implementasi Ollama Client (Native Ollama) ---
class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        resp = ollama.chat(
            model=self.model,
            messages=api_messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )

        return ChatResponse(
            text=resp["message"]["content"],
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
            model=self.model,
        )


# --- 2. Implementasi OpenAI Client (Mengarah ke Ollama / OpenAI) ---
class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = MODEL_NAME, base_url: str = "http://localhost:11434/v1"):
        self.model = model
        self._client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "ollama")
        )

    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )

        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return ChatResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- 6.7 Building a Provider-Agnostic Client ---\n")

    # KITA BISA BERGANTI CLIENT HANYA DENGAN MENGUBAH BARIS INI:
    client: BaseLLMClient = OllamaClient(model=MODEL_NAME)
    # client: BaseLLMClient = OpenAIClient(model=MODEL_NAME)

    # Kode aplikasi di bawah ini tetap sama tanpa peduli backend mana yang digunakan!
    msgs = [ChatMessage(role="user", content="What is a vector database?")]
    result = client.chat(msgs, system="Be concise and clear in 2 sentences.")

    print(f"Backend Model : {result.model}")
    print(f"Response      : {result.text}")
    print(f"Token Usage   : {result.input_tokens} in, {result.output_tokens} out")