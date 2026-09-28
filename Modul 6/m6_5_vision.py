import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import ollama

# 1. Load environment variables
load_dotenv()
# Catatan: Gunakan model vision seperti 'llava' atau 'llama3.2-vision'
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")


# Option A: Mengirim File Gambar Lokal secara langsung via Path (Ollama Native)
def describe_image_file_path(image_path: str) -> str:
    """Mengirim gambar lokal ke Ollama menggunakan file path."""
    if not Path(image_path).exists():
        return f"[ERROR] File {image_path} tidak ditemukan!"

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Describe what you see in this image in detail.",
                "images": [image_path],  # Path ke file gambar (misal: 'sample.jpg')
            }
        ],
    )
    return response["message"]["content"]


# Option B: Mengirim Gambar dengan Base64 Encoding
def describe_image_base64(image_path: str) -> str:
    """Membaca file lokal, mengonversi ke Base64, dan mengunduh deskripsinya."""
    if not Path(image_path).exists():
        return f"[ERROR] File {image_path} tidak ditemukan!"

    # Baca file binary & encode ke base64
    file_bytes = Path(image_path).read_bytes()
    b64_data = base64.b64encode(file_bytes).decode("utf-8")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is in this image?",
                "images": [b64_data],  # Data base64 dikirim langsung
            }
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print(f"--- 6.5 Vision Demo (Model: {VISION_MODEL}) ---")
    print("Catatan: Pastikan sudah pernah 'ollama pull llava' di terminal.\n")

    # Contoh pengujian jika kamu punya file gambar 'sample.jpg' di folder yang sama
    sample_img = "Modul 6/sample.jpg"

    if Path(sample_img).exists():
        print(f"Menganalisis gambar: {sample_img}...")
        description = describe_image_file_path(sample_img)
        print("\nDeskripsi Gambar:")
        print(description)
    else:
        print(f"File '{sample_img}' tidak ditemukan di folder Modul 6.")
        print("Silakan letakkan satu gambar berformat .jpg/.png di folder Modul 6 dan beri nama 'sample.jpg' untuk mengetesnya!")