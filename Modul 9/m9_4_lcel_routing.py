import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menggunakan Model: {ollama_model} ({ollama_host})\n")

# Inisialisasi ChatOllama lokal
llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.7,
)

parser = StrOutputParser()

# Chain 1: Khusus untuk menghasilkan kode Python
code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Python expert. Provide a complete, working code"
            " example.",
        ),
        ("human", "{input}"),
    ]
)

# Chain 2: Khusus untuk penjelasan konseptual/prosa
prose_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a clear technical writer. Explain conceptually."),
        ("human", "{input}"),
    ]
)

code_chain = code_prompt | llm | parser
prose_chain = prose_prompt | llm | parser


def route(input_dict: dict) -> str:
  """Route ke chain code atau prose berdasarkan kata kunci pada pertanyaan."""
  q = input_dict["input"].lower()
  if any(
      w in q for w in ["write", "code", "implement", "example", "function"]
  ):
    return "code"
  return "prose"


def run_routed(input_dict: dict) -> str:
  choice = route(input_dict)
  print(f"-> Routing terdeteksi: [{choice.upper()}]")
  if choice == "code":
    return code_chain.invoke(input_dict)
  return prose_chain.invoke(input_dict)


router_chain = RunnableLambda(run_routed)

# Uji coba 1: Permintaan kode (akan diarahkan ke code_chain)
print("=== Uji Coba 1 ===")
print(
    router_chain.invoke({"input": "Write a function to chunk text with overlap."})
)

# Uji coba 2: Pertanyaan konsep (akan diarahkan ke prose_chain)
print("\n" + "=" * 50)
print("=== Uji Coba 2 ===")
print(router_chain.invoke({"input": "What is attention in transformers?"}))