import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

load_dotenv()

# Inisialisasi model Ollama lokal (menggunakan konfigurasi yang sudah jalan)
model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
host_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

llm_local = ChatOllama(
    model=model_name,
    base_url=host_url,
    temperature=0.7,
)

print(f"Menggunakan Model: {model_name}\n")

# ==========================================
# 1. Simple Template
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in {language}."),
    ("human", "{question}"),
])

# Inspeksi struktur pesan yang dihasilkan
msgs = prompt.format_messages(
    role="Python tutor",
    language="Indonesian",
    question="Apa itu generator dalam Python?",
)

print("--- 1. Struktur Objek Pesan ---")
for m in msgs:
    print(f"[{m.type.upper()}]: {m.content}")

# Menjalankan langsung ke LLM menggunakan LCEL (Pipe '|')
chain = prompt | llm_local | StrOutputParser()
print("\n--- Output LLM (Simple Template) ---")
response = chain.invoke({
    "role": "Python tutor",
    "language": "Indonesian",
    "question": "Apa itu generator dalam Python secara singkat?"
})
print(response)


# ==========================================
# 2. Template dengan Riwayat Percakapan (MessagesPlaceholder)
# ==========================================
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today's context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

formatted = chat_prompt.format_messages(
    context="User is learning about RAG",
    history=[
        HumanMessage(content="What is retrieval?"),
    ],
    input="How is it different from fine-tuning?",
)

print("\n" + "="*50)
print("--- 2. Template dengan History ---")
print(f"Total pesan di dalam prompt: {len(formatted)}")
for m in formatted:
    print(f"[{m.type.upper()}]: {m.content}")

# Menjalankan ke LLM dengan riwayat percakapan
chat_chain = chat_prompt | llm_local | StrOutputParser()
print("\n--- Output LLM (Dengan Context & History) ---")
chat_response = chat_chain.invoke({
    "context": "User is learning about RAG",
    "history": [
        HumanMessage(content="What is retrieval?"),
    ],
    "input": "How is it different from fine-tuning?"
})
print(chat_response)