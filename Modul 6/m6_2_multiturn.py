import os
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def chat(system: str) -> None:
    """Interactive multi-turn chat loop menggunakan Ollama."""
    # Menyiapkan riwayat percakapan dengan system prompt di posisi pertama
    history = [{"role": "system", "content": system}]

    print(f"--- Chat Session Started (Model: {MODEL_NAME}) ---")
    print("Ketik 'exit' atau 'quit' untuk mengakhiri percakapan.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Chat ended.")
            break

        if not user_input:
            continue

        # 1. Tambahkan pesan user ke riwayat
        history.append({"role": "user", "content": user_input})

        # 2. Kirim seluruh riwayat percakapan ke Ollama
        response = ollama.chat(
            model=MODEL_NAME,
            messages=history,
        )

        assistant_text = response["message"]["content"]

        # 3. Tambahkan balasan assistant ke riwayat
        history.append({"role": "assistant", "content": assistant_text})

        print(f"\nQwen: {assistant_text}\n")


# Jalankan fungsi chat dengan system prompt khusus
if __name__ == "__main__":
    chat(system="You are a helpful Python tutor.")