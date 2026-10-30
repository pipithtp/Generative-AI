import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
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

# Template 1: Meringkas teks
summarise_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Summarise this in one sentence."),
        ("human", "{text}"),
    ]
)

# Template 2: Ekstraksi kata kunci
keywords_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract 3 keywords from this text. Return them comma-separated.",
        ),
        ("human", "{text}"),
    ]
)

parser = StrOutputParser()

# Menjalankan dua cabang chain secara paralel menggunakan RunnableParallel
parallel_chain = RunnableParallel(
    summary=summarise_prompt | llm | parser,
    keywords=keywords_prompt | llm | parser,
)

# Eksekusi parallel chain
result = parallel_chain.invoke(
    {
        "text": "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents at inference time."
    }
)

print("--- Hasil Eksekusi Paralel ---")
print("Summary :", result["summary"])
print("Keywords:", result["keywords"])