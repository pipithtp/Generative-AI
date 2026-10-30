import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

# Otomatis mengambil dari .env atau default ke qwen2.5:3b
model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
host_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

llm_local = ChatOllama(
    model=model_name,
    base_url=host_url,
    temperature=0.7,
)

messages = [
    SystemMessage(content="You are a concise technical assistant."),
    HumanMessage(content="What is LangChain Expression Language?"),
]

# 1. Invoke
print("--- 1. Standard Invoke ---")
response = llm_local.invoke(messages)
print(response.content)

# 2. Stream
print("\n--- 2. Streaming Output ---")
for chunk in llm_local.stream(messages):
    print(chunk.content, end="", flush=True)
print()

# 3. Batch
print("\n--- 3. Batch Execution ---")
responses = llm_local.batch(
    [
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ]
)

for i, r in enumerate(responses, 1):
    print(f"Jawaban {i}: {r.content[:100]}...")