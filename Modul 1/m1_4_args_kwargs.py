# 1. *args - Variable positional arguments (dikumpulkan sebagai Tuple)
def log_messages(*messages: str) -> None:
    """Menerima jumlah pesan teks yang tidak terbatas dan mencetaknya."""
    for msg in messages:
        print(f"[LOG] {msg}")


# 2. **kwargs - Variable keyword arguments (dikumpulkan sebagai Dictionary)
def create_api_payload(model: str, **kwargs) -> dict:
    """Membuat dictionary payload API secara dinamis dari opsi keyword kwargs."""
    payload = {"model": model}
    payload.update(kwargs)
    return payload


if __name__ == "__main__":
    print("--- 1. Testing *args ---")
    log_messages("Starting process", "Loading model parameters", "Ready for inference")
    print()

    print("--- 2. Testing **kwargs ---")
    payload = create_api_payload(
        "qwen2.5:3b",
        max_tokens=1024,
        temperature=0.3,
        stream=True,
    )
    
    print("Hasil Payload Dictionary:")
    print(payload)