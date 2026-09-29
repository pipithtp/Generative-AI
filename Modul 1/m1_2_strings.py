# 1. Deklarasi String & f-string untuk Pembuatan Prompt
user_input: str = "Explain transformers in simple terms"
system_prompt: str = "You are a helpful AI tutor."

# f-strings - Cara standar menyusun prompt secara dinamis
full_prompt: str = f"System: {system_prompt}\nUser: {user_input}"
print("--- Full Prompt ---")
print(full_prompt)
print()

# 2. String Methods yang Sering Digunakan
print("--- String Methods ---")
print("Upper       :", user_input.upper())
print("Split Words :", user_input.split())
print("Replace     :", user_input.replace("simple", "plain"))
print("Length      :", len(user_input))
print()

# 3. Multi-line Strings untuk System Prompt Panjang
long_prompt: str = """
You are an expert data scientist.
Answer concisely in bullet points.
"""

# strip() menghapus spasi dan newline kosong di awal/akhir
clean_prompt: str = long_prompt.strip()

print("--- Cleaned Multi-line Prompt ---")
print(clean_prompt)