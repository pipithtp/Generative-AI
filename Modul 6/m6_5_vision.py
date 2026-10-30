import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import requests
import ollama

# 1. Load environment variables
load_dotenv()
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llava")


def get_image_base64(source: str) -> str:
    """Mengambil byte gambar dari file lokal ATAU URL, lalu diubah ke base64."""
    # Jika input berupa URL (http:// atau https://)
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=15)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    # Jika input berupa path file lokal
    file_path = Path(source)
    if not file_path.exists():
        raise FileNotFoundError(f"File lokal '{source}' tidak ditemukan!")

    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def describe_image(source: str, prompt: str = "Describe what you see in this image in detail.") -> str:
    """Menganalisis gambar baik dari URL maupun file lokal."""
    try:
        b64_image = get_image_base64(source)
        
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_image],
                }
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[ERROR] Gagal memproses gambar: {e}"


if __name__ == "__main__":
    print(f"--- 6.5 Vision Demo (Model: {VISION_MODEL}) ---")
    print("Pastikan model vision (misal: llava atau llama3.2-vision) sudah di-pull di Ollama.\n")

    # 1. Contoh Gambar dari URL
    url_img = "https://selma.co.id/files/uploads/inspirationarticle/thumb_image/2022/Nov/08/636a230669ffb/hewanpeliharasetia-cover-770x770-resize.jpg?token=ec5f5b176b5c0035df496ac9ad4d203e"
    print(f"1. Menganalisis gambar dari URL: {url_img}")
    desc_url = describe_image(url_img, prompt="What logo or text is visible in this image?")
    print("\nHasil URL:")
    print(desc_url)
    print("-" * 50)

    # 2. Contoh Gambar dari File Lokal
    local_img = "Modul 6/sample.jpg"
    print(f"\n2. Menganalisis file lokal: {local_img}")
    if Path(local_img).exists():
        desc_local = describe_image(local_img)
        print("\nHasil Lokal:")
        print(desc_local)
    else:
        print(f"File '{local_img}' belum ada. Letakkan gambar di folder tersebut untuk tes lokal.")