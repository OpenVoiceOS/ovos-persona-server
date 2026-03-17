# FAQ — `ovos-persona-server`

## What is `ovos-persona-server`?

A FastAPI server that exposes an OVOS `Persona` instance as an HTTP API compatible with both the **OpenAI Chat Completions API** and the **Ollama API**. Any client that speaks OpenAI or Ollama (Open WebUI, LangChain, llm CLI, etc.) works without modification.

## How do I install it?

```bash
pip install ovos-persona-server
```

Development install:

```bash
uv pip install -e .
```

## How do I start the server?

```bash
ovos-persona-server --persona /path/to/persona.json --host 0.0.0.0 --port 8337
```

## What is a persona JSON file?

A JSON file that defines the persona name and the solver plugin chain:

```json
{
  "name": "My Assistant",
  "solvers": ["ovos-solver-openai-plugin"],
  "ovos-solver-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "sk-xxxx"
  }
}
```

## Which API endpoints are available?

| Endpoint | Protocol |
|----------|----------|
| `POST /v1/chat/completions` | OpenAI Chat Completions (streaming supported) |
| `POST /v1/completions` | OpenAI Legacy Completions |
| `GET /v1/models` | OpenAI model listing |
| `POST /api/chat` | Ollama Chat (streaming supported) |
| `POST /api/generate` | Ollama Generate (streaming supported) |
| `GET /api/tags` | Ollama model listing |

## Is authentication required?

No. The server accepts requests from any origin. Deploy behind a reverse proxy with authentication if the endpoint is publicly accessible.

## Does streaming work?

Yes. Both OpenAI (`POST /v1/chat/completions` with `"stream": true`) and Ollama endpoints support streaming. OpenAI returns Server-Sent Events; Ollama returns newline-delimited JSON.

## What Python versions are supported?

Python >=3.9. See `QUICK_FACTS.md`.

## What is the `model` field used for?

The `model` field in requests is accepted but ignored. The persona name loaded at startup is used as the model identifier in all responses.

## How do I run tests?

```bash
uv run pytest test/ -v --cov=ovos_persona_server --cov-report=term-missing
```

Note: there are currently no unit tests — see `AUDIT.md`.

## How do I contribute?

1. Fork the repository and create a feature branch from `dev`.
2. Write tests for your changes.
3. Open a PR targeting `dev`.
4. Ensure CI passes before requesting review.

## Where are schemas defined?

- OpenAI schemas: `ovos_persona_server/schemas/openai_chat.py`
- Ollama schemas: `ovos_persona_server/schemas/ollama.py`

Token counts in `usage` are word-split approximations (`len(text.split())`), not real tokenizer counts.
