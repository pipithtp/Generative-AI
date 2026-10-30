import os
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menggunakan Model: {ollama_model} ({ollama_host})\n")

# Inisialisasi model ChatOllama lokal
llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.7,
)

parser = StrOutputParser()

# Prompt dengan MessagesPlaceholder untuk menyisipkan riwayat percakapan
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI tutor. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm | parser

# Session store - menyimpan histori chat berdasarkan session_id
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]


# Bungkus chain dengan RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user-123"}}

# Turn 1
print("=== Turn 1 ===")
r1 = chain_with_history.invoke({"input": "What is RAG?"}, config=config)
print("Turn 1 Answer:\n", r1)

# Turn 2 - Model otomatis mengingat konteks dari Turn 1
print("\n" + "=" * 50)
print("=== Turn 2 ===")
r2 = chain_with_history.invoke(
    {"input": "What are the main components I just asked about?"}, config=config
)
print("Turn 2 Answer:\n", r2)

# Inspeksi isi riwayat percakapan yang tersimpan di memori
print("\n" + "=" * 50)
history = get_session_history("user-123")
print(f"Messages in history: {len(history.messages)}")
for i, msg in enumerate(history.messages, 1):
  print(f"  {i}. [{msg.type.upper()}]: {msg.content[:80]}...")