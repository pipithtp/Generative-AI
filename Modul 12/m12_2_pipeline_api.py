from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

print("=== 1. Text Classification (Sentiment Analysis) ===")
classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
)
texts = [
    "This LLM API is incredibly fast and easy to use.",
    "The documentation is confusing and the latency is terrible.",
    "The model works fine for most tasks.",
]
results = classifier(texts)
for text, result in zip(texts, results):
  print(f"[{result['label']} | {result['score']:.3f}] {text[:60]}")

print("\n=== 2. Named Entity Recognition (NER) ===")
ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1,
)
text_ner = (
    "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."
)
entities = ner(text_ner)
for ent in entities:
  print(f"{ent['word']!r:25s} -> {ent['entity_group']} ({ent['score']:.3f})")

print("\n=== 3. Text Generation (DistilGPT-2) ===")
generator = pipeline(
    task="text-generation",
    model="distilgpt2",
    device=-1,
)
output = generator(
    "The future of AI agents is",
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256,
)
for i, seq in enumerate(output, 1):
  print(f"Sequence {i}:\n{seq['generated_text']}\n")

print("=== 4. Summarisation (BART-large-CNN) ===")
article = """
Large language models have fundamentally changed how software is built.
Developers now reach for AI APIs as naturally as they reach for database clients.
The shift from rule-based systems to neural language models has accelerated
since the release of GPT-3 in 2020, and the pace has only increased with
instruction-tuned models like ChatGPT, Claude, and Gemini. These models can
write code, explain concepts, extract information, and reason through complex
problems - tasks that previously required specialised rule-based systems or
significant human effort.
"""

# Menggunakan AutoModel & AutoTokenizer langsung (kebal dari perubahan pipeline Transformers v5+)
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer(
    article, return_tensors="pt", max_length=1024, truncation=True
)
summary_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens=60,
    min_new_tokens=20,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Hasil Ringkasan:")
print(summary_text)