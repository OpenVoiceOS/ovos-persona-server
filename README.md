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

## OPM Tool Plugins — MCP and UTCP exposure

Installed `ToolBox` plugins (OPM entry-point group `opm.agents.toolbox`) are
automatically surfaced over two protocols when the server starts.

### Installing the MCP extra

```bash
pip install ovos-persona-server[mcp]
```

Without the `[mcp]` extra only the UTCP endpoints are active.

### UTCP — Universal Tool Calling Protocol

Two endpoints are added at `/tools`:

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/tools/manual` | Returns a UTCP manual JSON listing all tools |
| `POST` | `/tools/{name}` | Invoke a tool by name with a JSON body |

**Fetch the manual:**

```bash
curl http://localhost:8337/tools/manual
```

Response shape:

```json
{
  "utcp_version": "1.0",
  "tools": [
    {
      "name": "my_tool",
      "description": "Does something useful.",
      "tool_provider": {
        "type": "http",
        "method": "POST",
        "url": "http://localhost:8337/tools/my_tool",
        "content_type": "application/json"
      },
      "inputs": [
        {"name": "query", "type": "string", "required": true, "description": "Search query"}
      ],
      "output_schema": { ... }
    }
  ]
}
```

**Invoke a tool:**

```bash
curl -X POST http://localhost:8337/tools/my_tool \
     -H "Content-Type: application/json" \
     -d '{"query": "hello"}'
```

### MCP — Model Context Protocol

When the `[mcp]` extra is installed, the server mounts an MCP SSE endpoint at
`/mcp`.  Each installed `ToolBox` tool is registered as an MCP tool with the
name, description, and JSON Schema derived from its OPM definition.

**Claude Desktop / MCP client config:**

```json
{
  "mcpServers": {
    "ovos-persona-tools": {
      "url": "http://localhost:8337/mcp/sse"
    }
  }
}
```

**Standalone stdio MCP server** (for clients that spawn a subprocess):

```bash
ovos-persona-tools-mcp
```

This runs the same tool set over the stdio MCP transport.

### Writing a ToolBox plugin

Implement `ToolBox` from `ovos_plugin_manager.templates.agent_tools` and
register it under the `opm.agents.toolbox` entry-point group:

```toml
# pyproject.toml
[project.entry-points."opm.agents.toolbox"]
my_toolbox = "my_package.toolbox:MyToolBox"
```

The server picks it up automatically on the next start.

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
