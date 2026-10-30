import os
# Pastikan library python-dotenv sudah terinstall (biasanya sudah ada, atau jalankan !pip install python-dotenv)
from dotenv import load_dotenv

# Memuat file .env ke dalam sistem (jika ada)
load_dotenv()

def get_api_key(provider: str) -> str:
    """Retrieve an API key from the environment."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }

    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(
            f"{env_var} is not set. Add it to your .env file."
        )
    return key

# Contoh penggunaan (ini akan memicu error jika variabel environment belum diset)
try:
    anthropic_key = get_api_key("anthropic")
except Exception as e:
    print(f"Catatan Keamanan: {e}")
