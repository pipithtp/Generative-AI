from typing import Generator


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Generator[str, None, None]:
    """Yield overlapping text chunks for embedding/RAG pipelines."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        start += chunk_size - overlap


if __name__ == "__main__":
    # Simulasi dokumen panjang
    document = "Python is a versatile language. " * 30

    print("--- 1. Iterating Over Generator Chunks ---")
    chunk_count = 0
    for chunk in chunk_text(document, chunk_size=100, overlap=20):
        chunk_count += 1
        # Di kode nyata: embed chunk dan simpan ke vector store
    
    print(f"Processed total {chunk_count} chunks efficiently.\n")

    print("--- 2. Generator Expression Demo ---")
    # Generator expression (lazy evaluation)
    sizes = (len(chunk) for chunk in chunk_text(document, chunk_size=100, overlap=20))
    print(f"Max chunk size: {max(sizes)} characters")