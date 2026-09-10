# A local LLM that answers from Wikipedia

This is a worked example of server-side tool calling. A 1.1 GB quantised model
runs on a CPU-only box, the persona server does the Wikipedia lookup itself, and
the client gets a finished answer — no GPU, no API key, no `tool_calls` for the
caller to handle.

Ask it something the model cannot know:

```bash
curl -s http://localhost:8338/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-1.7b",
       "messages": [{"role": "user",
                     "content": "How far away is the star system LTT 1445?"}]}'
```

```json
{"choices": [{"finish_reason": "stop",
              "message": {"role": "assistant",
                          "content": "LTT 1445 is a triple M-dwarf system located 22.4 light-years away in the constellation Eridanus.",
                          "tool_calls": []}}],
 "model": "qwen3-1.7b"}
```

The same model with no toolbox installed answers "approximately 120 light-years"
instead. That gap is the whole point of the exercise, so build the control too.

## How the loop works

`chat.py` looks for a chat engine that advertises `supports_tools`. Today
`ovos-chat-openai-plugin` (from `ovos-openai-plugin`) is the only one. It then
offers the model two sets of tools at once: whatever the API client sent in the
OpenAI `tools` field, and every tool exported by the installed `ToolBox`
plugins. When the model calls a client tool, the call is relayed back and the
caller runs it. When the model calls one of the persona's own tools,
`server_tools.run_tool_loop` runs it here, appends the result as a `role:"tool"`
message, and asks the model again — up to `MAX_TOOL_ITERS` (5) rounds. The
client never sees the tool call.

One request from the client therefore produces two calls to the LLM. That is the
easiest way to confirm from the outside that a tool really ran.

## What you need

- `ovos-persona-server` 0.17.0a1 or newer. Earlier versions hand a toolbox the
  whole persona blob instead of its own config section, so the toolbox loads
  zero tools and the persona looks like a model that simply chose not to call
  anything.
- `ovos-openai-plugin` 2.0.8a2 or newer, for the tool-capable chat engine.
- `ovos-wikipedia-plugin`, which registers the `search_wikipedia` tool under the
  `opm.agents.toolbox` entry point. It needs no credentials. As of 1.0.1a2 it
  does not load here at all — see the note below.
- A llama.cpp server, or any other OpenAI-compatible endpoint that emits real
  `tool_calls`.

## The llama.cpp side

```bash
docker run -d --name llamacpp-qwen3-1.7b -p 8101:8080 \
  -v /path/to/gguf:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server \
  -m /models/Qwen3-1.7B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --jinja -c 8192 -t 6
```

`--jinja` is mandatory. Without it llama.cpp does not apply the chat template
that turns a model's tool syntax into a `tool_calls` field. The server still
answers, the model still tries, and the reply arrives as prose — so a tool test
without `--jinja` silently becomes a plain-text test and proves nothing. Check it
before anything else:

```bash
curl -s http://localhost:8101/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"weather in Lisbon?"}],
       "tools":[{"type":"function","function":{"name":"get_weather","parameters":
         {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}' \
  | grep -o tool_calls
```

## The persona

```json
{
  "name": "qwen3-1.7b",
  "handlers": ["ovos-chat-openai-plugin"],
  "ovos-chat-openai-plugin": {
    "api_url": "http://llamacpp_qwen3_17b:8080/v1",
    "key": "unused",
    "model": "/models/Qwen3-1.7B-Q4_K_M.gguf",
    "system_prompt": "You are the OpenVoiceOS assistant. You cannot recall facts reliably, so for any question about a person, place, work, object or event you must first call the search_wikipedia tool with lang set to en. Answer only from what the tool returned, in at most fifty words, with no emojis. If the tool returned nothing useful, say so.",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024
  },
  "ovos-wikipedia-tool": {
    "lang": "en"
  }
}
```

Two details matter more than they look.

A toolbox is handed the config section named after its **plugin name** — the
entry-point name `ovos-wikipedia-tool`, not the package name and not the
`toolbox_id`. The chat engine section is keyed the same way, by
`ovos-chat-openai-plugin` rather than by `ovos-openai-plugin`; using the package
name raises `ImportError: 'ovos-openai-plugin' not installed`.

The sampling overrides are not cosmetic. `ovos-openai-plugin` defaults to
`temperature` 0.5, `top_p` 0.2 and `max_tokens` 300. At `top_p` 0.2 a small model
falls out of the tool-call grammar and emits a broken pseudo-XML tag such as
`<search_wikipediaArgs query="LTT 1445" lang="en">` as ordinary content, which
reaches the client as a nonsense answer. And 300 tokens truncates Qwen3 once its
thinking block is counted. `temperature` 0, `top_p` 1 and `max_tokens` 1024 fixed
both.

That config section reaches the toolbox through `settings.persona_config`, which
reads the `PERSONA_PATH` environment variable — not the `--persona` or
`--personas-dir` argument. Set `PERSONA_PATH` as well as the CLI flag. Toolboxes
are process-global (they come from entry points, not from the persona file), so
one persona file is enough to configure them even when the process serves
several personas.

## Compose

```yaml
services:
  llamacpp_qwen3_17b:
    container_name: llamacpp_qwen3_17b
    image: ghcr.io/ggml-org/llama.cpp:server
    restart: always
    volumes:
      - /media/data/models/gguf:/models:ro
    mem_limit: 4g
    ports:
      - 8101:8080
    command: ["-m", "/models/Qwen3-1.7B-Q4_K_M.gguf",
              "--host", "0.0.0.0", "--port", "8080",
              "--jinja", "-c", "8192", "-t", "6"]

  ovos_persona_wiki:
    container_name: ovos_persona_wiki
    build: ./build-wiki          # persona-server + ovos-wikipedia-plugin
    restart: always
    depends_on:
      - llamacpp_qwen3_17b
    environment:
      PERSONA_PATH: /personas/qwen3-1.7b.json
    volumes:
      - ./persona:/personas:ro
    mem_limit: 1g
    ports:
      - 8338:8337
    command: ["--personas-dir", "/personas",
              "--default-persona", "qwen3-1.7b",
              "--host", "0.0.0.0", "--port", "8337"]
```

The image is `python:3.12-slim` plus:

```dockerfile
RUN pip install --no-cache-dir \
        "ovos-persona>=0.9.0a16" \
        "ovos-openai-plugin>=2.0.8a2" \
        "ovos-persona-server>=0.17.0a1" \
        "ovos-wikipedia-plugin"
```

Because toolboxes are discovered from entry points, every persona in the process
gets the same tools. If you already run a persona server you do not want to hand
Wikipedia to, install the plugin in a separate container instead of that one.

## Build the control

Run a second, identical container with the toolbox plugin left out, on another
port. Same model, same persona, same prompt — the only difference is whether the
tool exists. Without that comparison a correct-sounding answer is not evidence of
anything; the model may simply have got lucky.

Measured on a CPU-only box over four questions, five runs each:

| Question | With `search_wikipedia` | Model alone |
|---|---|---|
| Who discovered Arrokoth, and on what date? | Marc Buie, 26 June 2014 | "NASA, April 15, 2006" |
| Which river does the Vasco da Gama Bridge span? | Tagus, Lisbon | "Cauvery River in Chennai" |
| How far away is LTT 1445? | 22.4 light-years | "approximately 120 light-years" |
| Population of Sibila, Mali, 2009 census? | 19,185 | not known |

Qwen3-1.7B answered 20 of 20 correctly with the tool and 0 of 20 without it.

## Confirming the tool actually ran

An answer that reads well may still be a hallucination, so check the LLM log
rather than the answer. One client request should produce two calls to
llama.cpp, the second with a much larger prompt because the Wikipedia summary was
appended:

```
slot launch_slot_: id 3 | task 8682 | processing task
slot      release: id 3 | task 8682 | stop processing: n_tokens = 493
slot launch_slot_: id 3 | task 8846 | processing task
slot      release: id 3 | task 8846 | stop processing: n_tokens = 974
```

For the call itself, wrap `server_tools._execute_server_call` and run the loop
directly inside the container:

```python
import json
from ovos_persona import Persona
from ovos_persona_server import server_tools
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

cfg = json.load(open("/personas/qwen3-1.7b.json"))
persona = Persona(cfg["name"], cfg)
engine = next(m for m in persona.solvers.modules if getattr(m, "supports_tools", False))
registry = server_tools.get_flat_tool_registry()
print("server tools:", list(registry))

orig = server_tools._execute_server_call
def traced(tc, reg):
    print("TOOL CALL  ->", tc.name, json.dumps(tc.arguments))
    out = orig(tc, reg)
    print("TOOL RESULT<-", out.content[:300])
    return out
server_tools._execute_server_call = traced

msgs = [AgentMessage(role=MessageRole.SYSTEM,
                     content=cfg["ovos-chat-openai-plugin"]["system_prompt"]),
        AgentMessage(role=MessageRole.USER,
                     content="Which river does the Vasco da Gama Bridge span, and in which city?")]
print("FINAL:", server_tools.run_tool_loop(engine, msgs, [], registry=registry).content)
```

```
server tools: ['search_wikipedia']
TOOL CALL  -> search_wikipedia {"query": "Vasco da Gama Bridge", "lang": "en"}
TOOL RESULT<- {"results": [["Vasco da Gama Bridge", "The Vasco da Gama Bridge is a cable-stayed bridge ... spans the Tagus River in Parque das Nações in Lisbon, the capital of Portugal. ...
FINAL: The Vasco da Gama Bridge spans the Tagus River in Lisbon, Portugal.
```

If `server tools` prints an empty list, the toolbox failed to load. The loader
swallows the reason into a warning, so read the startup log:

```
Failed to load ToolBox plugin ovos-wikipedia-tool: ...
```

### The Wikipedia plugin needs a patch right now

`ovos-wikipedia-plugin` up to and including 1.0.1a2 declares
`WikipediaToolbox.__init__(self, config=None)`. The OPM `ToolBox` contract is
`(toolbox_id, config, bus)` and this server's loader calls `cls(config=cfg,
bus=None)`, so instantiation raises

```
Failed to load ToolBox plugin ovos-wikipedia-tool: WikipediaToolbox.__init__() got an unexpected keyword argument 'bus'
```

The loader turns that into a warning and carries on, so the persona reports zero
tools and behaves exactly like a model that chose not to call anything. The same
`__init__` also drops `config` instead of passing it to `super()`, so
`self.config` is never populated. Until a fixed release exists, patch it in the
image after `pip install`:

```dockerfile
RUN python - <<'PY'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/site-packages/ovos_wikipedia/__init__.py")
s = p.read_text()
s = s.replace("    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:\n        \"\"\"\n        Initialise the toolbox.",
              "    def __init__(self, config: Optional[Dict[str, Any]] = None, bus=None) -> None:\n        \"\"\"\n        Initialise the toolbox.")
s = s.replace("        super().__init__(toolbox_id=self.toolbox_id)",
              "        super().__init__(toolbox_id=self.toolbox_id, config=config, bus=bus)")
p.write_text(s)
PY
```

## How small can the model be

Qwen3-0.6B (390 MB) is not reliable enough. It emits correct `tool_calls`
sometimes, and the rest of the time it either answers from memory without calling
anything or leaks a malformed `<search_wikipediaArgs ...>` tag as content. Over
the same twenty runs it scored 6 of 20, and the failures were per-question rather
than random: it never once called the tool for "How far away is LTT 1445?".

Qwen3-1.7B at Q4_K_M is 1.1 GB and scored 20 of 20 on the same box. That looks
like the practical floor for driving this loop. A question takes about six to
eleven seconds on six CPU threads.
