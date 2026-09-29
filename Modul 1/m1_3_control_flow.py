def classify_response_length(token_count: int) -> str:
    """Mengklasifikasikan panjang respons LLM berdasarkan jumlah token."""
    if token_count < 100:
        return "short"
    elif token_count < 500:
        return "medium"
    elif token_count < 2000:
        return "long"
    else:
        return "very long"


if __name__ == "__main__":
    print("--- 1.3 Control Flow: IF / ELIF / ELSE Demo ---\n")

    # Pengujian fungsi klasifikasi
    test_counts = [80, 350, 1200, 3000]

    for count in test_counts:
        category = classify_response_length(count)
        print(f"Token Count: {count:<4} -> Category: '{category}'")