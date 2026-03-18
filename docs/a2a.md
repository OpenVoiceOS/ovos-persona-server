# A2A Endpoint

`ovos-persona-server` can expose the loaded persona as a standard [A2A](https://google.github.io/A2A/) (Agent-to-Agent) agent server. Any A2A client — including `ovos-a2a-agent`, LangGraph agents, CrewAI flows, and custom A2A clients — can discover and interact with it using the standard A2A protocol.

## Requirements

```bash
uv pip install 'ovos-persona-server[a2a]'
```

The `[a2a]` extra adds `a2a-sdk>=0.3.0` and `httpx>=0.27`. Without it the server starts normally with all other APIs available; A2A is silently disabled (a warning is logged if `--a2a-base-url` is provided).

## Starting the server with A2A enabled

```bash
ovos-persona-server \
  --persona my-persona.json \
  --host 0.0.0.0 \
  --port 8337 \
  --a2a-base-url http://myhost:8337/a2a
```

The `--a2a-base-url` value must be the **publicly reachable URL** of the `/a2a` mount point — this is what A2A clients use to reach the server and what appears in the Agent Card. If you are running behind a reverse proxy or inside Docker, set this to the external URL, not `localhost`.

## Enabling programmatically

```python
from ovos_persona_server import create_persona_app

app = create_persona_app(
    "my-persona.json",
    a2a_base_url="http://myhost:8337/a2a",
)
```

If `a2a_base_url=None` (default), the A2A endpoint is not mounted.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/a2a/.well-known/agent.json` | Agent Card — name, description, capabilities, skills, URL |
| `POST` | `/a2a/` | JSON-RPC 2.0 dispatcher — `message/send`, `message/stream`, `tasks/get`, `tasks/cancel` |

## Agent Card

The Agent Card is auto-generated from the persona config. `name` comes from `persona.name`; `description` from `persona.config["description"]`, falling back to `"OVOS Persona: <name>"`.

Example response from `GET /a2a/.well-known/agent.json`:

```json
{
  "name": "my-persona",
  "description": "OVOS Persona: my-persona",
  "url": "http://myhost:8337/a2a",
  "version": "1.0",
  "capabilities": { "streaming": true },
  "skills": [
    {
      "id": "chat",
      "name": "Chat",
      "description": "Multi-turn conversation with an OVOS persona",
      "inputModes": ["text"],
      "outputModes": ["text"]
    }
  ],
  "defaultInputModes": ["text"],
  "defaultOutputModes": ["text"]
}
```

## Sending a message

`message/send` — synchronous, waits for the full response:

```bash
curl -X POST http://localhost:8337/a2a/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is the capital of France?"}]
      }
    }
  }'
```

`message/stream` — SSE streaming, receives sentence chunks as they arrive:

```bash
curl -X POST http://localhost:8337/a2a/ \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-2",
    "method": "message/stream",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Tell me a story"}]
      }
    }
  }'
```

## Streaming

The A2A endpoint supports SSE streaming. `Persona.stream()` (which yields sentence-level chunks) is called in a thread pool via `asyncio.to_thread`. Each non-empty chunk is emitted as a `TaskArtifactUpdateEvent` SSE event. A final `TaskStatusUpdateEvent(state=completed)` closes the stream.

From the A2A client perspective, streaming provides sentence-level latency — useful when the A2A client feeds text to TTS and wants to start speaking before the full response is ready.

## Multi-turn sessions

The A2A `contextId` in each request maps to an OVOS session. Persona solvers that maintain session state will correlate turns by this ID. Stateless solvers (most knowledge-base solvers) process each message independently.

## Connecting ovos-a2a-agent

On another OVOS instance, create a persona that points to this server:

```json
{
  "name": "remote-persona",
  "chat_module": "ovos-a2a-agent",
  "ovos-a2a-agent": {
    "url": "http://myhost:8337/a2a"
  }
}
```

Then start the remote OVOS with this persona. Spoken utterances will be forwarded to this server via A2A and answered by the persona running here.

## Architecture

```
A2A client request
  └─ POST /a2a/
       └─ A2AStarletteApplication     [a2a-sdk]
            └─ DefaultRequestHandler  [a2a-sdk]
                 └─ OVOSPersonaAgentExecutor          [a2a.py:106]
                      └─ asyncio.to_thread(Persona.stream(messages))
                           └─ Persona.stream()        [ovos-persona]
                                └─ solver chain: wikipedia → ddg → LLM → ...
```

`Persona.stream()` is synchronous; it is run in a thread pool via `asyncio.to_thread` so the A2A event loop is never blocked.

## Troubleshooting

**A2A endpoint returns 404**
`a2a-sdk` is not installed. Install it and restart:
```bash
uv pip install 'ovos-persona-server[a2a]'
ovos-persona-server --persona my.json --a2a-base-url http://localhost:8337/a2a
```

**Agent Card `url` field is wrong**
The `--a2a-base-url` value is used verbatim in the Agent Card. Ensure it matches the URL A2A clients will use to reach the server.

**`message/send` returns an empty task**
The persona solver returned an empty string. Check the solver chain in the persona JSON and that all required API keys are configured.
