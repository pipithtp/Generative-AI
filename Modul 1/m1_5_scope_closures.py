# 1. Global Scope (Module level)
API_KEY: str = "sk-test-xxx"


def get_client() -> str:
    # Local Scope
    base_url: str = "https://api.anthropic.com"
    # Mengakses base_url (local) dan API_KEY (global via LEGB)
    return f"Client({base_url}, key={API_KEY[:6]}...)"


# 2. Closure - Mengingat Enclosing Scope
def make_counter(start: int = 0):
    """Membuat fungsi increment yang merekam hitungan token secara tertutup (encapsulated state)."""
    count = [start]  # Mutable container

    def increment() -> int:
        count[0] += 1
        return count[0]

    return increment


if __name__ == "__main__":
    print("--- 1. Testing Scope (LEGB) ---")
    print(get_client())
    print()

    print("--- 2. Testing Closure (Token Counter) ---")
    token_counter = make_counter(start=0)

    print("Call 1:", token_counter())  # 1
    print("Call 2:", token_counter())  # 2
    print("Call 3:", token_counter())  # 3