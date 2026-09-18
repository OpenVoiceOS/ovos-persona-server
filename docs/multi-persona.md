# Serving several personas from one process

One `ovos-persona-server` process can host any number of personas. Clients pick
one with the `model` field of whichever vendor API they speak — **a persona's
`name` is its model id**.

A persona is not a model: it is a solver/agent chain, a system prompt, and
memory settings. `model` therefore selects a *persona*; it is never forwarded to
the LLM behind that persona. To change the LLM, edit the persona JSON.

---

## Starting the server

```bash
# one persona (unchanged)
ovos-persona-server --persona /path/to/my-persona.json

# several personas
ovos-persona-server \
  --persona /path/to/assistant.json \
  --persona /path/to/rivescript-bot.json

# every *.json in a directory, loaded in file-name order
ovos-persona-server --personas-dir /etc/ovos/personas

# both, with an explicit default
ovos-persona-server \
  --personas-dir /etc/ovos/personas \
  --persona /path/to/extra.json \
  --default-persona assistant
```

| Flag | Purpose |
|------|---------|
| `--persona` | Path to a persona `.json`. Repeat it to load more than one. |
| `--personas-dir` | Load every `*.json` in the directory, sorted by file name. |
| `--default-persona` | Name of the persona that answers when a request names no model. |

The server refuses to start when two personas share a name, or when
`--default-persona` names a persona that was not loaded.

### Which persona is the default?

1. The persona named by `--default-persona`, if given.
2. Otherwise the **first** persona loaded — the first `--persona` on the command
   line, or the first file of `--personas-dir` in sorted order when no
   `--persona` was given.

The default persona answers any request that omits `model`, and it is the
persona used by the surfaces that cannot select one (see below).

---

## Discovering the personas

```bash
curl -s http://localhost:8337/openai/v1/models | python3 -m json.tool
```

```json
{
  "object": "list",
  "data": [
    {"id": "assistant", "object": "model", "created": 1765400000, "owned_by": "ovos"},
    {"id": "rivescript-bot", "object": "model", "created": 1765400000, "owned_by": "ovos"}
  ]
}
```

The Ollama surface lists the same set at `GET /ollama/api/tags` and
`GET /ollama/api/ps`.

---

## Selecting a persona

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="unused")
client.chat.completions.create(model="rivescript-bot",
                               messages=[{"role": "user", "content": "hello"}])
```

An unknown name is rejected with HTTP 404 and an OpenAI-shaped error body that
lists what is available:

```json
{
  "error": {
    "message": "The model `gpt-4` does not exist. Available models: assistant, rivescript-bot",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

### Single-persona deployments keep the old behaviour

When only one persona is loaded the `model` field stays advisory: whatever the
client sends is ignored and the persona answers, exactly as before this feature
existed. Strict routing — and the 404 — apply only from the second persona
onwards.

---

## Per-surface behaviour

| API | Selector | Unknown name |
|-----|----------|--------------|
| OpenAI `/openai/v1` | `model` in the body of `/chat/completions` and `/completions` | 404 |
| Ollama `/ollama/api` | `model` in the body of `/chat` and `/generate`; `?model=` on `/show` | 404 |
| Anthropic `/anthropic/v1` | `model` in the body of `/messages` | 404 |
| Google Gemini `/gemini/v1beta/models` | the `{model}` path segment | 404 |
| Cohere `/cohere/v1` | `model` in the body of `/chat` and `/generate` (optional field → default persona) | 404 |
| AWS Bedrock `/bedrock/model` | the `{model_id}` path segment | falls back to the default persona |
| HuggingFace TGI `/tgi` | none — see below | — |
| A2A `/a2a` | one agent card per persona at `/a2a/<name>` | 404 from the mount |

**AWS Bedrock is permissive on purpose.** The Bedrock `model_id` also selects the
*response format* (`anthropic.*`, `amazon.titan-*`, `cohere.command*` all have
different response shapes), so a real Bedrock id must keep working. A `model_id`
that matches a loaded persona selects it; anything else keeps its vendor response
shape and is answered by the default persona.

**HuggingFace TGI cannot select a persona.** The TGI protocol serves exactly one
model per endpoint — neither `/generate` nor `/info` carries a model field — so
this surface always uses the default persona, and `GET /tgi/info` reports the
default persona's name. Use one of the model-aware surfaces to reach the others.

**A2A** mounts one agent card per persona at `/a2a/<persona name>`, alongside the
default persona's card at `/a2a`. The `url` in each card is
`<--a2a-base-url>/<persona name>`.

**Embeddings** are not persona-scoped. Every `/embeddings` surface delegates to
the single shared embeddings backend (see [embeddings.md](embeddings.md)), so the
`model` field there names the embedding model, not a persona.

---

## Isolation between personas

Each persona is a separate `Persona` instance with its own solver chain and its
own memory plugin instance, so nothing crosses over in process memory.

Memory plugins backed by shared external storage (a vector DB, a database) key
their history by session id alone, which two personas would collide on for the
same caller. In multi-persona mode the server therefore namespaces the key it
passes to the memory plugin as `<persona name>::<session id>`. Single-persona
deployments keep the bare session id, so conversations stored before you added a
second persona stay addressable.
