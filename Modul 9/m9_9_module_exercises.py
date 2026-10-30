import hashlib
import os
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

print(f"Menggunakan Model: {ollama_model} ({ollama_host})\n")

llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=0.3,
)
parser = StrOutputParser()
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# =====================================================================
# LATIHAN 1: LCEL 2-Step Chain (Facts Extraction -> Quiz Questions -> List)
# =====================================================================
print("=" * 60)
print("LATIHAN 1: Facts to Quiz Chain (Output: Python List)")
print("=" * 60)


class KeyFacts(BaseModel):
  facts: List[str] = Field(
      description=(
          "List of exactly 3 distinct key facts extracted from the text"
      )
  )


class QuizList(BaseModel):
  questions: List[str] = Field(
      description="A list of quiz questions generated for each fact"
  )


facts_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an analytical assistant. Extract exactly 3 key facts from the"
        " text.",
    ),
    ("human", "{text}"),
])
facts_chain = facts_prompt | llm.with_structured_output(KeyFacts)

quiz_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an educator. Generate 1 quiz question for each fact provided.",
    ),
    ("human", "Facts:\n{facts}\n\nGenerate quiz questions for these facts."),
])
quiz_chain = quiz_prompt | llm.with_structured_output(QuizList)


def extract_and_generate_quiz(inputs: dict) -> List[str]:
  facts_obj: KeyFacts = facts_chain.invoke(inputs)
  formatted_facts = "\n".join(f"- {f}" for f in facts_obj.facts)
  quiz_obj: QuizList = quiz_chain.invoke({"facts": formatted_facts})
  return quiz_obj.questions


quiz_pipeline = RunnableLambda(extract_and_generate_quiz)

sample_text = """
The Python programming language was conceived in the late 1980s by Guido van Rossum. 
Python 3.0 was released in 2008 and introduced backward-incompatible design changes. 
Memory management in Python is handled automatically through reference counting and a cyclic garbage collector.
"""

questions_result = quiz_pipeline.invoke({"text": sample_text})
print(f"Hasil Tipe: {type(questions_result)}")
for idx, q in enumerate(questions_result, 1):
  print(f"{idx}. {q}")


# =====================================================================
# LATIHAN 2: Manual MultiQueryRetriever (Deduplicate by Content Hash)
# =====================================================================
print("\n" + "=" * 60)
print("LATIHAN 2: Manual MultiQueryRetriever")
print("=" * 60)

knowledge_docs = [
    Document(
        page_content=(
            "RAG combines dense vector retrieval with large language models to"
            " ground generation in facts."
        ),
        metadata={"id": 1},
    ),
    Document(
        page_content=(
            "Vector embeddings represent semantic textual meaning as"
            " high-dimensional coordinates."
        ),
        metadata={"id": 2},
    ),
    Document(
        page_content=(
            "ChromaDB and FAISS are popular vector search indexes optimized for"
            " similarity queries."
        ),
        metadata={"id": 3},
    ),
    Document(
        page_content=(
            "Prompt templates in LangChain enable standardizing instructions"
            " across different model backends."
        ),
        metadata={"id": 4},
    ),
]
vectorstore = Chroma.from_documents(knowledge_docs, embeddings)


class ParaphrasedQueries(BaseModel):
  queries: List[str] = Field(
      description="3 distinct paraphrased versions of the original query"
  )


multi_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Generate 3 different search variations or paraphrases for the given"
        " query to retrieve technical documents.",
    ),
    ("human", "{query}"),
])
paraphrase_chain = multi_query_prompt | llm.with_structured_output(
    ParaphrasedQueries
)


def manual_multiquery_retriever(query: str, k: int = 2) -> List[Document]:
  variations = paraphrase_chain.invoke({"query": query}).queries
  all_queries = [query] + variations
  print(f"Query & Variasi ({len(all_queries)}): {all_queries}")

  seen_hashes = set()
  unique_docs: List[Document] = []

  for q in all_queries:
    docs = vectorstore.similarity_search(q, k=k)
    for doc in docs:
      content_hash = hashlib.sha256(
          doc.page_content.strip().encode("utf-8")
      ).hexdigest()
      if content_hash not in seen_hashes:
        seen_hashes.add(content_hash)
        unique_docs.append(doc)

  return unique_docs


retrieved_union = manual_multiquery_retriever(
    "How does semantic vector retrieval work?", k=2
)
print(f"\nTotal Dokumen Unik Ditemukan: {len(retrieved_union)}")
for i, doc in enumerate(retrieved_union, 1):
  print(f"[{i}] {doc.page_content}")


# =====================================================================
# LATIHAN 3: Map-Reduce Long Document Summarisation
# =====================================================================
print("\n" + "=" * 60)
print("LATIHAN 3: Map-Reduce Summarisation")
print("=" * 60)

map_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise this text chunk in 1 concise sentence:"),
    ("human", "{chunk}"),
])
map_chain = map_prompt | llm | parser

reduce_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Synthesise the following chunk summaries into one coherent, unified"
        " executive summary:",
    ),
    ("human", "{summaries}"),
])
reduce_chain = reduce_prompt | llm | parser


def map_reduce_summarise(long_text: str) -> str:
  splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
  chunk_docs = splitter.create_documents([long_text])
  print(f"Jumlah chunks hasil splitting: {len(chunk_docs)}")

  # 1. Map step (ringkas masing-masing chunk)
  chunk_summaries = []
  for c in chunk_docs:
    summary = map_chain.invoke({"chunk": c.page_content})
    chunk_summaries.append(summary)

  combined_summaries = "\n".join(f"- {s}" for s in chunk_summaries)

  # 2. Reduce step (gabungkan ringkasan menjadi satu)
  final_summary = reduce_chain.invoke({"summaries": combined_summaries})
  return final_summary


long_sample_text = """
Artificial intelligence has witnessed exponential evolution over the last decade, primarily catalyzed by deep learning and neural network architectures.
The transformer architecture, introduced in 2017, completely transformed natural language processing by replacing recurrent structures with self-attention mechanisms.
Modern large language models like GPT, LLaMA, and Qwen leverage billions of parameters trained on internet-scale textual corpora.
However, fundamental challenges persist, notably hallucinations, high computational costs during training and inference, and data privacy vulnerabilities.
Techniques like Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning (PEFT) have emerged as primary paradigms to enhance factual accuracy and lower operational costs.
"""

summary_res = map_reduce_summarise(long_sample_text)
print("\nHasil Ringkasan Akhir (Reduce):")
print(summary_res)


# =====================================================================
# LATIHAN 4: RAG with Groundedness Quality Filter
# =====================================================================
print("\n" + "=" * 60)
print("LATIHAN 4: RAG dengan Response Quality Filter (Groundedness Score)")
print("=" * 60)


class EvaluatedAnswer(BaseModel):
  answer: str = Field(description="The generated answer to the question")
  groundedness_score: int = Field(
      description=(
          "Groundedness score from 0 to 10 evaluating how factual the answer is"
          " based on the context"
      )
  )
  reasoning: str = Field(description="Short reason explaining the score")


eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict grading evaluator.
Given the Context and an Answer, rate the groundedness of the Answer strictly on a scale of 0 to 10:
- 10: Entirely supported by context.
- 0: Completely fabricated or contradicts context.
Return the answer, the numeric score (0-10), and a brief justification.""",
    ),
    ("human", "Context:\n{context}\n\nGenerated Answer:\n{answer}"),
])
eval_chain = eval_prompt | llm.with_structured_output(EvaluatedAnswer)

rag_gen_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using ONLY the provided context. If unknown, say"
        " 'I don't have that information.'\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])
rag_gen_chain = rag_gen_prompt | llm | parser


def format_docs(docs: List[Document]) -> str:
  return "\n\n".join(d.page_content for d in docs)


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def full_evaluated_rag(question: str) -> dict:
  retrieved_docs = retriever.invoke(question)
  context_str = format_docs(retrieved_docs)

  answer = rag_gen_chain.invoke(
      {"context": context_str, "question": question}
  )
  evaluation: EvaluatedAnswer = eval_chain.invoke(
      {"context": context_str, "answer": answer}
  )

  return {
      "question": question,
      "answer": evaluation.answer,
      "score": evaluation.groundedness_score,
      "reasoning": evaluation.reasoning,
  }


eval_result = full_evaluated_rag("What is ChromaDB used for?")
print(f"Pertanyaan : {eval_result['question']}")
print(f"Jawaban    : {eval_result['answer']}")
print(f"Skor (0-10): {eval_result['score']}/10")
print(f"Alasan     : {eval_result['reasoning']}")