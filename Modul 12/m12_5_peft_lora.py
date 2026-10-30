import os
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── 1. Load base model ───────────────────────────────────────────────────────
model_name = "distilgpt2"

print("Memuat base model dan tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# ── 2. Configure LoRA ────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,  # Scaling factor
    target_modules=["c_attn"],  # Layer proyeksi perhatian untuk keluarga GPT-2
    lora_dropout=0.1,
    bias="none",
)

peft_model = get_peft_model(model, lora_config)
print("\nStatistik Parameter Trainable:")
peft_model.print_trainable_parameters()

# ── 3. Prepare dataset ───────────────────────────────────────────────────────
training_examples = [
    (
        "Question: What is RAG? Answer: RAG stands for Retrieval-Augmented"
        " Generation."
    ),
    (
        "Question: What is fine-tuning? Answer: Fine-tuning trains a"
        " pre-trained model on new data."
    ),
    (
        "Question: What is an embedding? Answer: An embedding is a dense vector"
        " representing text meaning."
    ),
    (
        "Question: What is a vector database? Answer: A vector database stores"
        " embeddings for similarity search."
    ),
    (
        "Question: What is LangChain? Answer: LangChain is a framework for"
        " building LLM applications."
    ),
]


def tokenize(examples):
  return tokenizer(
      examples["text"],
      truncation=True,
      max_length=128,
      padding="max_length",
  )


dataset = Dataset.from_dict({"text": training_examples})
tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

# ── 4. Training arguments (Windows & CPU friendly) ───────────────────────────
OUTPUT_DIR = "./lora_output"
ADAPTER_DIR = "./lora_adapter"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    fp16=False,
    use_cpu=True,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

# ── 5. Train ─────────────────────────────────────────────────────────────────
print("\nMemulai training LoRA adapter...")
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized,
)
trainer.train()

# ── 6. Save adapter only ─────────────────────────────────────────────────────
print(f"\nMenyimpan adapter ke '{ADAPTER_DIR}'...")
peft_model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("LoRA adapter berhasil disimpan.")

# ── 7. Load adapter for inference ────────────────────────────────────────────
print("\nMemuat ulang base model dan memasang LoRA adapter...")
base = AutoModelForCausalLM.from_pretrained(model_name)
merged = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged.eval()

eval_prompt = "Question: What is RAG? Answer:"
inputs = tokenizer(eval_prompt, return_tensors="pt")

print(f'\nPrompt: "{eval_prompt}"')
with torch.no_grad():
  out = merged.generate(
      **inputs,
      max_new_tokens=30,
      pad_token_id=tokenizer.eos_token_id,
      do_sample=False,
  )

result_text = tokenizer.decode(out[0], skip_special_tokens=True)
print("\nHasil Inferensi setelah Fine-tuning:")
print(result_text)