"""One shared embeddings backend, reached through several vendor SDKs.

The server loads a single text-embeddings plugin (TEXT_EMBEDDINGS_PLUGIN) and
exposes it on every vendor surface, so these all hit the same model.

    python examples/embeddings_clients.py
"""
from openai import OpenAI
from ollama import Client as OllamaClient

BASE = "http://localhost:8337"
TEXTS = ["hello world", "the moon affects tides"]


def via_openai() -> None:
    client = OpenAI(base_url=f"{BASE}/openai/v1", api_key="unused")
    resp = client.embeddings.create(model="", input=TEXTS)
    print(f"openai   : {len(resp.data)} vectors, dim={len(resp.data[0].embedding)}")


def via_ollama() -> None:
    client = OllamaClient(host=f"{BASE}/ollama")
    resp = client.embed(model="", input=TEXTS)
    print(f"ollama   : {len(resp.embeddings)} vectors, dim={len(resp.embeddings[0])}")


if __name__ == "__main__":
    via_openai()
    via_ollama()
    # Cohere (/cohere/v1/embed), Gemini (:embedContent / :batchEmbedContents),
    # HuggingFace TGI (/tgi/embed) and Bedrock Titan/Cohere embed models all
    # resolve to the same backend — see docs/embeddings.md.
