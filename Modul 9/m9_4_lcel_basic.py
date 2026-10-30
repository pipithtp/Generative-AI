import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

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

# Basic chain: prompt | model | parser
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarise the following text in exactly {sentences} sentences.",
        ),
        ("human", "{text}"),
    ]
)

# LCEL Pipeline: prompt -> LLM -> string parser
chain = prompt | llm | StrOutputParser()

# Eksekusi chain dengan invoke
result = chain.invoke(
    {
        "sentences": 2,
        "text": """LangChain is a framework for developing applications powered by large language models.
It provides abstractions for prompts, models, memory, and retrieval so developers can
compose complex AI workflows from simple building blocks without writing boilerplate.""",
    }
)

print("--- Hasil Ringkasan LCEL ---")
print(result)