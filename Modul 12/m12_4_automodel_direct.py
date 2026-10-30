import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=== 12.4 AutoModel - Loading Models Directly (DistilGPT-2) ===")

model_name = "distilgpt2"

# 1. Muat Tokenizer dan Model CausalLM secara eksplisit
print(f"Memuat model dan tokenizer '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=(
        torch.float32
    ),  # float32 untuk CPU; float16 jika menggunakan GPU
)

# Set model ke evaluation mode
model.eval()

# 2. Siapkan prompt & tokenisasi input
prompt = "The key advantage of local LLMs is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f'\nPrompt: "{prompt}"')
print(f"Input IDs shape: {inputs['input_ids'].shape}")

# 3. Text Generation langsung melalui model (tanpa Pipeline wrapper)
print("\nMenghasilkan teks (inference)...")
with torch.no_grad():
  output_ids = model.generate(
      inputs["input_ids"],
      attention_mask=inputs.get("attention_mask"),
      max_new_tokens=40,
      do_sample=True,
      temperature=0.8,
      top_p=0.9,
      pad_token_id=tokenizer.eos_token_id,
  )

# 4. Decode token IDs kembali menjadi teks
generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n--- Output Hasil Generasi ---")
print(generated)