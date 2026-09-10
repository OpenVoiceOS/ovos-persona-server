# Custom containers for ovos-persona-server

The persona server is an OpenAI-compatible chat endpoint. Each persona JSON file
selects which plugins provide the answers. Build custom images by adding
plugins on top of the base image.

## Quick example — DuckDuckGo, Wikipedia and WordNet (no LLM, no API key)

These three are retrieval plugins. Each answers from its own source, so the
persona needs no model endpoint and no key. The failure plugin ends the chain
with a canned line when none of them answers.

First build the base image from this repository, then build on top of it:

```bash
docker build -t ovos-persona-server:local .
```

```dockerfile
FROM ovos-persona-server:local

RUN pip install --no-cache-dir --pre \
        ovos-ddg-plugin \
        ovos-wikipedia-plugin \
        ovos-wordnet-plugin \
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
    "ovos-ddg-plugin",
    "ovos-wikipedia-plugin",
    "ovos-wordnet-plugin",
    "ovos-solver-failure-plugin"
  ]
}
```

Build and run:

```bash
docker build -t my-persona-knowledge .
docker run -p 8337:8337 my-persona-knowledge
curl -s http://localhost:8337/openai/v1/models | python3 -m json.tool
```

`--pre` is needed because the OVOS plugins publish alphas, and some of them have no stable release yet.

Each name in `solvers` is a plugin name, not a package name. The two match for
the plugins above, but a package can carry several plugins under different
names. `pip show -f <package>` lists what it installed.

## OpenAI-compatible persona (any LLM endpoint)

```json
{
  "name": "LocalLLM",
  "solvers": [
    "ovos-chat-openai-plugin",
    "ovos-solver-failure-plugin"
  ],
  "ovos-chat-openai-plugin": {
    "api_url": "http://localhost:11434/v1",
    "key": "sk-placeholder",
    "model": "big-pickle",
    "system_prompt": "You are a helpful assistant."
  }
}
```

The plugin is `ovos-chat-openai-plugin` and the package that carries it is
`ovos-openai-plugin`, which the base image already installs. It works with any
OpenAI-compatible backend: Ollama (`http://localhost:11434/v1`), the llama.cpp
server, vLLM, or the OpenVoiceOS gateway at `https://llm.openvoiceos.pt/v1`.

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

The server speaks the OpenAI API at `http://localhost:8337/openai/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="unused")
resp = client.chat.completions.create(
    model="KnowledgeBot",
    messages=[{"role": "user", "content": "What is the speed of light?"}],
)
print(resp.choices[0].message.content)
```

The `model` is the persona name. `GET /openai/v1/models` lists the personas the
server loaded. Ollama clients use `/ollama/api/chat` and `/ollama/api/generate`.

## Notes on the public endpoint placeholder

The default `config/persona.json` references `https://llm.openvoiceos.pt/v1`,
the OpenVoiceOS gateway, which needs no API key. This endpoint is community-run
and may be unavailable. Edit `config/persona.json` before you start the
container to point at your own OpenAI-compatible endpoint.

## What the base image carries

The base image installs the published `ovos-persona-server` alpha with its `a2a`
and `mcp` extras, `ovos-openai-plugin` for any OpenAI-compatible model endpoint,
and `ovos-solver-failure-plugin` for the canned line that ends a persona chain.
A custom image `FROM python` directly installs the same three packages.

## Tools over UTCP and MCP

Installed `opm.agents.toolbox` plugins are served over UTCP at `/tools/manual`
and `/tools/{name}`. With the `mcp` extra, which the base image carries, start
the server with `--mcp` to mount an MCP endpoint at `/mcp` (streamable HTTP).
Add `"--mcp"` to the `command` list in `docker-compose.yml` to turn it on.
