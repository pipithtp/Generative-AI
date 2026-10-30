import os
import time
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# =====================================================================
# SOAL 1: ModelBenchmark (Bandingkan 2 Pipeline Text-Generation)
# =====================================================================
print("=" * 75)
print("SOAL 1: ModelBenchmark (10 Prompts pada 2 Model Berbeda)")
print("=" * 75)


class ModelBenchmark:

  def __init__(self, model_name_1: str, model_name_2: str):
    print(f"Menginisialisasi Model 1: {model_name_1}...")
    self.pipe1 = pipeline("text-generation", model=model_name_1, device=-1)
    print(f"Menginisialisasi Model 2: {model_name_2}...")
    self.pipe2 = pipeline("text-generation", model=model_name_2, device=-1)
    self.m1 = model_name_1
    self.m2 = model_name_2

  def benchmark(self, prompts: list[str], max_new_tokens: int = 20):
    records = []
    for idx, p in enumerate(prompts, 1):
      # Model 1
      t0 = time.perf_counter()
      out1 = self.pipe1(
          p,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          pad_token_id=50256,
      )
      lat1 = (time.perf_counter() - t0) * 1000
      len1 = len(out1[0]["generated_text"])

      # Model 2
      t0 = time.perf_counter()
      out2 = self.pipe2(
          p,
          max_new_tokens=max_new_tokens,
          do_sample=False,
          pad_token_id=50256,
      )
      lat2 = (time.perf_counter() - t0) * 1000
      len2 = len(out2[0]["generated_text"])

      records.append({
          "prompt": p[:32] + "...",
          "lat1": lat1,
          "len1": len1,
          "lat2": lat2,
          "len2": len2,
      })

    # Cetak tabel perbandingan
    print(f"\n{'-'*75}")
    print(f"{'Prompt':<36} | {self.m1:^16} | {self.m2:^16}")
    print(
        f"{'':<36} | {'Lat(ms)':<7} {'Chars':<7} | {'Lat(ms)':<7} {'Chars':<7}"
    )
    print(f"{'-'*75}")
    for r in records:
      print(
          f"{r['prompt']:<36} | {r['lat1']:<7.1f} {r['len1']:<7} |"
          f" {r['lat2']:<7.1f} {r['len2']:<7}"
      )
    print(f"{'-'*75}\n")


ten_prompts = [
    "Artificial intelligence is",
    "Machine learning models require",
    "Natural language processing helps",
    "Deep neural networks consist of",
    "Large language models can",
    "Vector databases are designed for",
    "Python is widely used in",
    "Transformers rely on the concept of",
    "Prompt engineering is the process of",
    "Reinforcement learning learns via",
]

# Menggunakan distilgpt2 dan gpt2 (keduanya ringan dijalankan pada CPU)
benchmarker = ModelBenchmark("distilgpt2", "gpt2")
benchmarker.benchmark(ten_prompts, max_new_tokens=15)


# =====================================================================
# SOAL 2: TokenBudgetChecker (Filter Teks Berdasarkan Token Limit)
# =====================================================================
print("=" * 75)
print("SOAL 2: TokenBudgetChecker")
print("=" * 75)


class TokenBudgetChecker:

  def __init__(self, model_name: str = "gpt2"):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def filter_texts(self, texts: list[str], max_tokens: int) -> list[dict]:
    passed_texts = []
    for text in texts:
      token_count = len(self.tokenizer.encode(text))
      if token_count <= max_tokens:
        passed_texts.append({"text": text, "token_count": token_count})
      else:
        print(
            f"[EXCEEDED] {token_count} tokens > max {max_tokens} :"
            f" '{text[:50]}...'"
        )
    return passed_texts


sample_texts = [
    "Short query.",
    (
        "This is a medium length sentence describing retrieval augmented"
        " generation workflows."
    ),
    (
        "The transformer architecture relies on stacked self-attention and"
        " point-wise feed-forward layers."
    ),
    "A very long sentence repeated "
    + ("with lots of repetitive words and phrases " * 15),
]

checker = TokenBudgetChecker(model_name="gpt2")
approved = checker.filter_texts(sample_texts, max_tokens=25)
print("\n--- Approved Texts within Token Budget ---")
for item in approved:
  print(f"[{item['token_count']} tokens] {item['text']}")


# =====================================================================
# SOAL 3: Extract Unique Entities from Corpus (PERSON, ORG, LOC)
# =====================================================================
print("\n" + "=" * 75)
print("SOAL 3: extract_entities_from_corpus (PERSON, ORG, LOC)")
print("=" * 75)

ner_pipeline = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1,
)


def extract_entities_from_corpus(texts: list[str]) -> dict:
  entity_map = {"PERSON": set(), "ORG": set(), "LOC": set()}
  group_aliases = {
      "PER": "PERSON",
      "PERSON": "PERSON",
      "ORG": "ORG",
      "LOC": "LOC",
  }

  for text in texts:
    results = ner_pipeline(text)
    for ent in results:
      target_key = group_aliases.get(ent["entity_group"])
      if target_key:
        entity_map[target_key].add(ent["word"].strip())

  return {k: sorted(list(v)) for k, v in entity_map.items()}


corpus = [
    "Satya Nadella spoke about cloud advancements at Microsoft in Redmond.",
    "Sam Altman and Greg Brockman founded OpenAI in San Francisco, California.",
    "Sundar Pichai oversees Google and Alphabet operations from Mountain View.",
    (
        "Jensen Huang presented new accelerated computing hardware at NVIDIA"
        " in Santa Clara."
    ),
]

extracted = extract_entities_from_corpus(corpus)
print("Unique Extracted Entities by Type:")
for cat, values in extracted.items():
  print(f"  {cat:<8} ({len(values)}): {values}")


# =====================================================================
# SOAL 4: Full LoRA Fine-Tuning Loop on Custom 20 Q&A Dataset
# =====================================================================
print("\n" + "=" * 75)
print("SOAL 4: LoRA Fine-Tuning Loop (20 Custom Q&A Dataset)")
print("=" * 75)

custom_qa_dataset = [
    (
        "Question: What is RAG? Answer: RAG combines external document"
        " retrieval with LLM generation."
    ),
    (
        "Question: What is ChromaDB? Answer: ChromaDB is an open-source"
        " lightweight vector database."
    ),
    (
        "Question: What is Pinecone? Answer: Pinecone is a cloud-managed"
        " scalable vector database."
    ),
    (
        "Question: What is Weaviate? Answer: Weaviate is a vector database"
        " supporting native hybrid search."
    ),
    (
        "Question: What is an embedding? Answer: An embedding is a dense"
        " mathematical vector representing semantic meaning."
    ),
    (
        "Question: What is cosine similarity? Answer: Cosine similarity"
        " measures the cosine of the angle between two vectors."
    ),
    (
        "Question: What is prompt engineering? Answer: Prompt engineering"
        " optimizes model instructions without weight updates."
    ),
    (
        "Question: What is fine-tuning? Answer: Fine-tuning updates pre-trained"
        " neural network parameters on specific tasks."
    ),
    (
        "Question: What is LoRA? Answer: LoRA injects trainable low-rank"
        " decomposition matrices into model layers."
    ),
    (
        "Question: What is PEFT? Answer: PEFT stands for Parameter-Efficient"
        " Fine-Tuning."
    ),
    (
        "Question: What is LangChain? Answer: LangChain is a framework for"
        " building composable LLM chains and agents."
    ),
    (
        "Question: What is LlamaIndex? Answer: LlamaIndex is a data framework"
        " specialized for indexing and querying private data."
    ),
    (
        "Question: What is a VectorStoreIndex? Answer: VectorStoreIndex builds"
        " semantic vector indices from document nodes."
    ),
    (
        "Question: What is a SummaryIndex? Answer: SummaryIndex synthesizes"
        " across all document nodes holistically."
    ),
    (
        "Question: What is BM25? Answer: BM25 is a probabilistic keyword"
        " ranking algorithm for sparse search."
    ),
    (
        "Question: What is HNSW? Answer: HNSW is a graph algorithm used for"
        " fast approximate nearest-neighbor vector search."
    ),
    (
        "Question: What is quantization? Answer: Quantization compresses"
        " floating-point weights into lower precision integers."
    ),
    (
        "Question: What is an LLM agent? Answer: An LLM agent plans actions and"
        " uses tools to accomplish multi-step objectives."
    ),
    (
        "Question: What is FastEmbed? Answer: FastEmbed is a lightweight and"
        " CPU-optimized embedding library."
    ),
    (
        "Question: What is Ollama? Answer: Ollama allows running large language"
        " models locally on consumer hardware."
    ),
]

lora_model_name = "distilgpt2"
qa_tokenizer = AutoTokenizer.from_pretrained(lora_model_name)
qa_tokenizer.pad_token = qa_tokenizer.eos_token

qa_base_model = AutoModelForCausalLM.from_pretrained(lora_model_name)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
)
peft_qa_model = get_peft_model(qa_base_model, lora_config)


def tokenize_qa(examples):
  return qa_tokenizer(
      examples["text"], truncation=True, max_length=64, padding="max_length"
  )


raw_ds = Dataset.from_dict({"text": custom_qa_dataset})
tokenized_qa = raw_ds.map(tokenize_qa, batched=True)
tokenized_qa = tokenized_qa.map(lambda x: {"labels": x["input_ids"]})

LORA_OUT = "./lora_qa_output"
LORA_ADAPTER = "./lora_qa_adapter"

train_args = TrainingArguments(
    output_dir=LORA_OUT,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    fp16=False,
    use_cpu=True,
    logging_steps=2,
    save_strategy="no",
    report_to="none",
)

print("Memulai proses fine-tuning LoRA pada 20 custom Q&A...")
trainer = Trainer(
    model=peft_qa_model, args=train_args, train_dataset=tokenized_qa
)
trainer.train()

peft_qa_model.save_pretrained(LORA_ADAPTER)
qa_tokenizer.save_pretrained(LORA_ADAPTER)
print(f"LoRA Adapter berhasil disimpan ke '{LORA_ADAPTER}'.")

# Verifikasi Inferensi
print("\n--- Verifikasi Inferensi Model Fine-Tuned ---")
base_eval = AutoModelForCausalLM.from_pretrained(lora_model_name)
eval_model = PeftModel.from_pretrained(base_eval, LORA_ADAPTER)
eval_model.eval()

test_prompts = [
    "Question: What is RAG? Answer:",
    "Question: What is LoRA? Answer:",
    "Question: What is ChromaDB? Answer:",
]

for q in test_prompts:
  inputs = qa_tokenizer(q, return_tensors="pt")
  with torch.no_grad():
    out = eval_model.generate(
        **inputs,
        max_new_tokens=20,
        pad_token_id=qa_tokenizer.eos_token_id,
        do_sample=False,
    )
  res = qa_tokenizer.decode(out[0], skip_special_tokens=True)
  print(f"Input : {q}")
  print(f"Output: {res}\n")