# Persona Server

## Running

`$ ovos-persona-server --persona rivescript_bot.json`

## Personas

personas don't need to use LLMs, you don't need a beefy GPU to use ovos-persona, find solver plugins [here](https://github.com/OpenVoiceOS?q=solver&type=all)

some repos and skills also provide solvers, such as ovos-classifiers (wordnet), skill-ddg, skill-wikipedia and skill-wolfie

```
{
  "name": "OldSchoolBot",
  "solvers": [
    "ovos-solver-wikipedia-plugin",
    "ovos-solver-ddg-plugin",
    "ovos-solver-plugin-wolfram-alpha",
    "ovos-solver-wordnet-plugin",
    "ovos-solver-rivescript-plugin",
    "ovos-solver-failure-plugin"
  ],
  "ovos-solver-plugin-wolfram-alpha": {"appid": "Y7353-9HQAAL8KKA"}
}
```

this persona would search ddg api / wikipedia for "what is"/"tell me about" questions,
falling back to wordnet when offline for dictionary look up,
and finally rivescript for general chitchat,
we also add the failure solver to be sure the persona always says something

wolfram alpha illustrates how to pass solver configs, it has a requirement for an API key

search/knowledge base solvers can be used together with LLM solvers to ensure factual answers and act as a tool/internet access layer,
in the example above you would typically replace rivescript with a LLM.

Some solvers may also use other solvers internally, such as a [MOS (Mixture Of Solvers)](https://github.com/TigreGotico/ovos-MoS)

## Client side usage

OpenAI compatible API, for usage with OVOS see [ovos-solver-plugin-openai-persona](https://github.com/OpenVoiceOS/ovos-solver-plugin-openai-persona)

```python
import openai

openai.api_key = ""
openai.api_base = "http://localhost:8337"

# NOTE - most solvers don't support a chat history,
#  only last message in messages list is considered
chat_completion = openai.ChatCompletion.create(
    model="",  # individual personas might support this, passed under context
    messages=[{"role": "user", "content": "tell me a joke"}],
    stream=False,
)

if isinstance(chat_completion, dict):
    # not stream
    print(chat_completion.choices[0].message.content)
else:
    # stream
    for token in chat_completion:
        content = token["choices"][0]["delta"].get("content")
        if content != None:
            print(content, end="", flush=True)

```

## Embeddings

A single, swappable embeddings service backs **every** vendor surface — the
OpenAI `POST /openai/v1/embeddings` endpoint, the Ollama `POST /ollama/api/embed`
(batch) and legacy `POST /ollama/api/embeddings` (single `prompt`) endpoints, and
the vector-store search path all delegate to the same backend. This mirrors how
inference is backed by one shared persona: swap the embeddings provider once and
it changes everywhere.

The backend is any OVOS text-embeddings plugin, configured through the
environment:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TEXT_EMBEDDINGS_PLUGIN` | embeddings plugin to load | `ovos-gguf-embeddings-plugin` |
| `EMBEDDINGS_URL` | remote embeddings service URL (OpenAI-compatible plugins) | — |
| `EMBEDDINGS_KEY` | API key for a remote embeddings service | — |
| `EMBEDDINGS_MODEL` | model name to request | — |

Point `TEXT_EMBEDDINGS_PLUGIN` at a local model (the default gguf plugin) or at
any remote embeddings API via an OpenAI-compatible plugin and the matching
`EMBEDDINGS_URL` / `EMBEDDINGS_MODEL`. When no embeddings plugin is available the
server falls back to a persona solver exposing `get_embeddings`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8337/openai/v1", api_key="")
resp = client.embeddings.create(model="", input=["hello", "world"])
print(len(resp.data), "vectors")
```

```python
from ollama import Client

client = Client(host="http://localhost:8337/ollama")
print(client.embed(model="", input=["hello", "world"]).embeddings)
```
