from typing import NamedTuple


# 1. Multiple Return Values & Tuple Unpacking
def parse_model_string(model_id: str) -> tuple[str, str, str]:
    """Parse 'provider/model-name:version' menjadi komponen terpisah."""
    provider, rest = model_id.split("/")
    if ":" in rest:
        name, version = rest.split(":")
    else:
        name, version = rest, "latest"
    return provider, name, version


# 2. Self-documenting Tuples dengan NamedTuple
class EmbeddingResult(NamedTuple):
    text: str
    vector: list[float]
    model: str


if __name__ == "__main__":
    print("--- 1. Tuple Unpacking Demo ---")
    provider, name, version = parse_model_string("anthropic/claude-sonnet-4-5:20241022")
    print(f"Provider : {provider}")
    print(f"Model    : {name}")
    print(f"Version  : {version}\n")

    print("--- 2. NamedTuple Demo ---")
    result = EmbeddingResult(
        text="Hello world",
        vector=[0.12, -0.34, 0.89],
        model="nomic-embed-text",
    )

    # Akses atribut berbasis nama
    print("Text     :", result.text)
    print("Model    :", result.model)
    print("Vector   :", result.vector)