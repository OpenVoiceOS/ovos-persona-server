# Custom containers for ovos-persona-server

The persona server is an OpenAI-compatible chat endpoint.  Each persona JSON file
selects which solver plugins provide the answers.  Build custom images by adding
solver plugins on top of the base image.

## Quick example — DDG + Wikipedia + fallback (no LLM, no API key)

```dockerfile
FROM ghcr.io/openvoiceos/ovos-persona-server:dev

RUN pip install --no-cache-dir \
        ovos-solver-ddg-plugin \
        ovos-solver-wikipedia-plugin \
        ovos-solver-wordnet-plugin \
        ovos-solver-failure-plugin

COPY config/persona.json /config/persona.json

CMD ["--persona", "/config/persona.json", \
     "--host", "0.0.0.0", "--port", "8337"]
```

`config/persona.json`:

```json
{
  "name": "KnowledgeBot",
  "solvers": [
    "ovos-solver-ddg-plugin",
    "ovos-solver-wikipedia-plugin",
    "ovos-solver-wordnet-plugin",
    "ovos-solver-failure-plugin"
  ]
}
```

Build and run:

```bash
docker build -t my-persona-knowledge .
docker run -p 8337:8337 my-persona-knowledge
curl -s http://localhost:8337/api/tags | python3 -m json.tool
```

## OpenAI-compatible persona (any LLM endpoint)

```json
{
  "name": "LocalLLM",
  "solvers": [
    "ovos-solver-openai-plugin",
    "ovos-solver-failure-plugin"
  ],
  "ovos-solver-openai-plugin": {
    "api_url": "http://localhost:11434/v1",
    "key": "sk-placeholder",
    "model": "llama3.1:8b",
    "system_prompt": "You are a helpful assistant."
  }
}
```

Works with any OpenAI-compatible backend: Ollama (`http://localhost:11434/v1`),
llama.cpp server, vLLM, or the community demo at `https://llama.smartgic.io/v1`.

**No real API key is required for the community demo server.**

## Compose override

```yaml
services:
  ovos-persona:
    build: .
    image: my-persona-knowledge
    command:
      - "--persona"
      - "/config/persona.json"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8337"
    volumes:
      - ./config:/config
```

## Client usage

The server is OpenAI API-compatible at `http://localhost:8337`:

```python
import openai

openai.api_key = ""
openai.api_base = "http://localhost:8337"
resp = openai.ChatCompletion.create(
    model="",
    messages=[{"role": "user", "content": "What is the speed of light?"}],
)
print(resp.choices[0].message.content)
```

It also exposes Ollama-compatible endpoints at `/api/chat` and `/api/generate`.

## Notes on the public endpoint placeholder

The default `config/persona.json` references `https://llama.smartgic.io/v1`
(the community Llama demo, sourced from `ovos-openai-plugin` defaults).
This endpoint is community-run and may be unavailable.  Edit
`config/persona.json` before starting the container to point at your own
OpenAI-compatible endpoint.

## Runtime requirements baked into the base image

The base image carries several packages the solver pipeline needs at runtime
beyond the declared dependencies:

- `ovos-workshop` — imported by `ovos-persona`
- `uvicorn` — required by the server entrypoint
- `ovos-lang-detector-classics-plugin` + `langdetect` — the solver chat
  pipeline auto-detects the query language when no `lang` is passed;
  `config/mycroft/mycroft.conf` pins `ovos-lang-detector-plugin-langdetect`
  as the detector

If you build a custom image `FROM python` directly instead of the base image,
install these too.

## MCP / UTCP

The persona server does not currently expose a UTCP or MCP endpoint.
Use the OpenAI-compatible API directly or connect via Ollama-compatible clients.
