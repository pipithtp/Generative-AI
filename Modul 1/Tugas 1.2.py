# Core scalar types
model_name: str = "claude-sonnet-4-5"
temperature: float = 0.7
max_tokens: int = 1024
is_streaming: bool = True

# Check types at runtime
print(type(model_name))
print(type(temperature))

# None - the absence of a value
response = None
print(response is None)

user_input = "Explain transformers in simple terms"
system_prompt = "You are a helpful AI tutor."

# f-strings - the preferred way to assemble prompts
full_prompt = f"System: {system_prompt}\nUser: {user_input}"
print(full_prompt)

# Common string methods
print(user_input.upper())
print(user_input.split())
print(user_input.replace("simple", "plain"))
print(len(user_input))

# Multi-line strings - useful for long system prompts
prompt = """
You are an expert data scientist.
Answer concisely in bullet points.
"""
print(prompt.strip())

# Integer arithmetic
tokens_used = 450
tokens_limit = 1024

remaining = tokens_limit - tokens_used
print(remaining)

# Float arithmetic
cost_per_token = 0.000003
total_cost = tokens_used * cost_per_token
print(f"Cost: ${total_cost:.6f}")

# Integer division and modulo
batches = tokens_used // 100
leftover = tokens_used % 100

print(batches)
print(leftover)

# Built-in math
import math

print(math.log2(512))
print(math.ceil(3.1))