# Core scalar types dengan Type Hints
model_name: str = "claude-sonnet-4-5"
temperature: float = 0.7
max_tokens: int = 1024
is_streaming: bool = True

# Pengecekan tipe data secara runtime
print(f"model_name type   : {type(model_name)}")   # <class 'str'>
print(f"temperature type  : {type(temperature)}")  # <class 'float'>
print(f"max_tokens type   : {type(max_tokens)}")   # <class 'int'>
print(f"is_streaming type : {type(is_streaming)}") # <class 'bool'>

# NoneType - merepresentasikan ketiadaan nilai (absence of a value)
response = None
print(f"Is response None? : {response is None}")    # True