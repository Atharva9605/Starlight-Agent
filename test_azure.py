"""Local Azure connectivity check — uses env vars only (no hardcoded secrets)."""
import os
from openai import AzureOpenAI

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/") + "/"
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

if not ENDPOINT.strip("/") or not API_KEY:
    raise SystemExit(
        "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in the environment / .env"
    )

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=VERSION,
)


def test_chat():
    print("Testing Chat Completion…")
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=10,
        )
        print(f"Chat Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Chat Failed: {e}")


def test_embed():
    print("\nTesting Embedding…")
    try:
        response = client.embeddings.create(input="hello world", model=EMBED_MODEL)
        print(f"Embed Success: dim={len(response.data[0].embedding)}")
    except Exception as e:
        print(f"Embed Failed: {e}")


if __name__ == "__main__":
    test_chat()
    test_embed()
