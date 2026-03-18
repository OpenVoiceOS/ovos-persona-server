# A2A Endpoint

OVOS Persona Server can expose any loaded persona as an [A2A](https://google.github.io/A2A/) (Agent-to-Agent) agent server.

## Requirements

```bash
uv pip install 'ovos-persona-server[a2a]'
```

## Starting with A2A enabled

```bash
ovos-persona-server \
  --persona my-persona.json \
  --host 0.0.0.0 \
  --port 8337 \
  --a2a-base-url http://myhost:8337/a2a
```

The A2A endpoint is mounted at `/a2a`. Without `--a2a-base-url` the endpoint is disabled.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/a2a/.well-known/agent.json` | Agent Card — capabilities, skills, public URL |
| `POST` | `/a2a/` | JSON-RPC 2.0: `message/send`, `message/stream`, `tasks/get`, `tasks/cancel` |

## Agent Card

`name` and `description` are taken from the persona config. `description` falls back to `"OVOS Persona: <name>"` if not set.

```json
{
  "name": "my-persona",
  "description": "OVOS Persona: my-persona",
  "url": "http://myhost:8337/a2a",
  "version": "1.0",
  "capabilities": { "streaming": true },
  "skills": [{ "id": "chat", "name": "Chat" }],
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"]
}
```

## Streaming

`message/stream` is supported. `Persona.stream()` sentence chunks are emitted as `TaskArtifactUpdateEvent` SSE events; the final `TaskStatusUpdateEvent(state=completed)` closes the stream.

## Architecture

```
A2A client
  └─ POST /a2a/                      JSON-RPC 2.0
       └─ A2AStarletteApplication    [a2a-sdk]
            └─ DefaultRequestHandler [a2a-sdk]
                 └─ OVOSPersonaAgentExecutor  [a2a.py]
                      └─ Persona.stream()     [ovos-persona]
```

Key class: `OVOSPersonaAgentExecutor` — `a2a.py:106`.
`Persona.stream()` is synchronous; it is offloaded via `asyncio.to_thread` to avoid blocking the event loop.
