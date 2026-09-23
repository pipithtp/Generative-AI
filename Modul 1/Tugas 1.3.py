# Function with conditional statements
def classify_response_length(token_count: int) -> str:
    if token_count < 100:
        return "short"
    elif token_count < 500:
        return "medium"
    elif token_count < 2000:
        return "long"
    else:
        return "very long"


print(classify_response_length(80))    # short
print(classify_response_length(350))   # medium
print(classify_response_length(3000))  # very long


# Lists
models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]

# Basic iteration
for model in models:
    print(f"Checking: {model}")


# With index - use enumerate, not range(len(...))
for i, model in enumerate(models):
    print(f"{i + 1}. {model}")


# Iterate over key-value pairs in a dict
token_limits = {
    "gpt-4o": 128000,
    "claude-sonnet-4-5": 20000,
}

for model, limit in token_limits.items():
    print(f"{model}: {limit:,} tokens")


# While Loops
import time

MAX_RETRIES = 3
attempt = 0

while attempt < MAX_RETRIES:
    attempt += 1
    print(f"Attempt {attempt}")

    if attempt == 2:
        print("Success!")
        break

    time.sleep(0.1)

else:
    # Runs only if loop exhausted without break
    print("All retries failed")