from transformers import AutoTokenizer

print("=== 1. Memuat Tokenizer (GPT-2) ===")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set pad_token ke eos_token untuk mencegah warning padding pada GPT-2
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token

text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

# Encode teks menjadi token IDs
tokens = tokenizer.encode(text)
print(f"Token count   : {len(tokens)}")
print(f"Token IDs     : {tokens}")

# Decode kembali token IDs ke teks asli
decoded = tokenizer.decode(tokens)
print(f"Decoded       : {decoded}")

# Melihat token individual dalam bentuk string
token_strings = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens        : {token_strings}")

print("\n=== 2. Batch Encoding dengan Padding & Truncation ===")
# Batch encoding untuk input model (PyTorch tensors)
batch = ["What is RAG?", "Explain vector databases in detail."]
encoded = tokenizer(
    batch,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",  # Mengembalikan PyTorch tensor
)

print(f"Batch input_ids shape     : {encoded['input_ids'].shape}")
print(f"Attention mask shape      : {encoded['attention_mask'].shape}")
print(f"Isi input_ids:\n{encoded['input_ids']}")
print(f"Isi attention_mask:\n{encoded['attention_mask']}")

print("\n=== 3. Helper Menghitung Token Sebelum Inferensi ===")


def count_tokens(text: str, model: str = "gpt2") -> int:
  tok = AutoTokenizer.from_pretrained(model)
  return len(tok.encode(text))


prompt = "Summarise the following document: " + "word " * 500
print(f"Prompt token count: {count_tokens(prompt)}")