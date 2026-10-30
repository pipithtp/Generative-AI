import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Memuat konfigurasi environment
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menggunakan Model: {ollama_model} ({ollama_host})\n")

# 1. Dokumen sumber pengetahuan
docs = [
    Document(
        page_content=(
            "RAG stands for Retrieval-Augmented Generation. It connects LLMs to"
            " external knowledge bases."
        )
    ),
    Document(
        page_content=(
            "The retrieval step uses embeddings and vector search to find"
            " relevant documents for a given query."
        )
    ),
    Document(
        page_content=(
            "The generation step takes the retrieved context and the user"
            " question to produce a grounded answer."
        )
    ),
    Document(
        page_content=(
            "Vector databases like ChromaDB, Pinecone, and Weaviate store"
            " embeddings for fast similarity search."
        )
    ),
    Document(
        page_content=(
            "Fine-tuning bakes knowledge into model weights. RAG retrieves"
            " knowledge at inference time. Both have trade-offs."
        )
    ),
]

# 2. Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# 3. Embedding & In-Memory Vector Store (Chroma)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # Gunakan model khusus embedding
    base_url=ollama_host,
)

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. RAG Prompt Template
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain the answer, say "I don't have that information."
Context:
{context}""",
    ),
    ("human", "{question}"),
])

# 5. LLM & LCEL RAG Chain
llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.2,  # Rendah agar jawaban strictly grounded ke context
)
parser = StrOutputParser()


def format_docs(docs: list[Document]) -> str:
  return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | parser
)

# 6. Pengujian Pertanyaan
questions = [
    "What does RAG stand for?",
    "How is RAG different from fine-tuning?",
    (
        "What is the capital of France?"
    ),  # Tidak ada di context -> model harus menjawab sesuai instruksi sistem
]

for q in questions:
  print(f"Q: {q}")
  print(f"A: {rag_chain.invoke(q)}\n")