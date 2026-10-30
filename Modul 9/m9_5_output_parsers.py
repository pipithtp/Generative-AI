import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menggunakan Model: {ollama_model} ({ollama_host})\n")

# Inisialisasi model Ollama lokal
llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.7,
)


# Skema Pydantic untuk penegakan format output terstruktur
class ModelComparison(BaseModel):
  models: List[str] = Field(description="Model names compared")
  best_for_cost: str = Field(description="Most cost-effective model")
  best_for_quality: str = Field(description="Highest quality model")
  recommendation: str = Field(description="Overall recommendation")


# with_structured_output didukung langsung oleh ChatOllama
structured_llm = llm.with_structured_output(ModelComparison)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI model comparison expert."),
    ("human", "Compare these models for a startup on a budget: {models}"),
])

chain = prompt | structured_llm

print("--- Memproses Perbandingan Model ---")
result: ModelComparison = chain.invoke(
    {"models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"}
)

print(f"Models Evaluated : {result.models}")
print(f"Best for cost    : {result.best_for_cost}")
print(f"Best for quality : {result.best_for_quality}")
print(f"Recommendation   : {result.recommendation}")