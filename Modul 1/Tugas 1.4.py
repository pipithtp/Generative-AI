# Function with default argument
def build_prompt(system: str, user: str, temperature: float = 0.7) -> str:
    """Assemble a prompt string for an LLM API call.
    
    Args:
        system: The system instruction.
        user: The user message.
        temperature: Sampling temperature (0.0 – 2.0).
        
    Returns:
        Formatted prompt string.
    """
    return f"[System]\n{system}\n\n[User]\n{user}"


result = build_prompt(
    system="You are a concise AI assistant.",
    user="What is backpropagation?",
)

print(result)


# *args - variable positional arguments
def log_messages(*messages: str) -> None:
    for msg in messages:
        print(f"[LOG] {msg}")


log_messages("Starting", "Loading model", "Done")


# **kwargs - variable keyword arguments
def create_api_payload(model: str, **kwargs) -> dict:
    payload = {"model": model}
    payload.update(kwargs)
    return payload


payload = create_api_payload(
    "claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0.3,
    stream=True,
)

print(payload)


# Lambda Functions
responses = [
    {"model": "gpt-4o", "tokens": 540},
    {"model": "claude-sonnet-4-5", "tokens": 310},
    {"model": "gemini-1.5-pro", "tokens": 820},
]


# Sort by token count ascending
sorted_responses = sorted(
    responses,
    key=lambda r: r["tokens"]
)

for r in sorted_responses:
    print(f"{r['model']}: {r['tokens']} tokens")