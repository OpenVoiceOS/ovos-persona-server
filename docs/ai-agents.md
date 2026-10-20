# Use ovos-persona-server with AI agents (MCP and UTCP)

## What you can do

The persona server works with agent tools in two directions.

**It gives its tools to other agents.** Every `ToolBox` plugin installed next to the server (entry point group `opm.agents.toolbox`) becomes a set of tools. The server publishes these tools over two protocols:

- **MCP (Model Context Protocol).** Start the server with `--mcp` and it serves an MCP endpoint at `/mcp`. Claude Code, Claude Desktop, opencode, Mistral Vibe, an OVOS agent or any MCP client can then call the tools. The `ovos-persona-tools-mcp` command serves the same tools over stdio, for clients that start the server as a process.
- **UTCP (Universal Tool Calling Protocol).** Every server publishes a UTCP manual at `GET /utcp`. The manual tells a UTCP client how to call each tool over plain HTTP at `POST /tools/{name}`. No extra package is needed.

**Its personas use tools from other servers.** Add an `ovos-mcp-toolbox` or `ovos-utcp-toolbox` section to the persona file, from [ovos-tool-adapters](https://github.com/OpenVoiceOS/ovos-tool-adapters). The persona then calls the tools of another MCP server or UTCP manual while it answers a chat request. The server runs those calls itself.

MCP and UTCP publish **tools**, not the persona. To talk to the persona itself from another agent, use the [A2A endpoint](a2a.md) or the OpenAI-compatible chat API.

| Endpoint | Available when | Install |
| :--- | :--- | :--- |
| `GET /utcp`, `POST /tools/{name}` | Always | `pip install --pre ovos-persona-server` |
| `/mcp` | The server starts with `--mcp` | `pip install --pre "ovos-persona-server[mcp]"` |
| `ovos-persona-tools-mcp` (stdio) | The `mcp` extra is installed | `pip install --pre "ovos-persona-server[mcp]"` |

The examples use the default port `8337`, the [Wikipedia toolbox](https://github.com/OpenVoiceOS/ovos-wikipedia-plugin) (tool `search_wikipedia`) and this persona file, `wiki.json`:

```json
{
  "name": "wiki",
  "solvers": ["ovos-chat-openai-plugin"],
  "ovos-chat-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "none",
    "model": "qwen2.5:7b",
    "temperature": 0,
    "top_p": 1.0,
    "max_tokens": 1024
  },
  "ovos-wikipedia-tools": {"lang": "en"}
}
```

Any OpenAI-compatible endpoint and any toolbox plugin work the same way. Each plugin reads the section of the persona file that has its plugin name.

## Let Claude Code use the persona tools

You get the tools of every installed toolbox inside Claude Code.

1. Install the server, a toolbox and the chat plugin, and start the server with MCP:

   ```bash
   pip install --pre "ovos-persona-server[mcp]" ovos-wikipedia-plugin ovos-openai-plugin
   ovos-persona-server --persona wiki.json --mcp
   ```

2. Add the server to Claude Code:

   ```bash
   claude mcp add --transport http ovos-persona http://localhost:8337/mcp
   claude mcp get ovos-persona
   ```

   On success, `claude mcp get` shows:

   ```
   ovos-persona:
     Scope: Local config (private to you in this project)
     Status: ✔ Connected
     Type: http
     URL: http://localhost:8337/mcp
   ```

3. Ask Claude Code to use a tool:

   ```bash
   claude -p 'Call the search_wikipedia tool of the ovos-persona MCP server with query "Évora" and lang "en". Reply with only the first ten words of the first summary in the result, verbatim.' \
     --allowedTools 'mcp__ovos-persona__search_wikipedia'
   ```

   Output:

   ```
   Évora, officially the Most Noble and Ever Loyal City of Évora,
   ```

If `claude mcp get` does not show `Connected`, see [Troubleshooting](#troubleshooting).

## Let another agent talk to the persona

MCP gives an agent the persona's tools, not the persona. To send the persona a message and get its answer, use A2A. Start the server with `--a2a-base-url` and point an A2A client at `/a2a`. The [A2A page](a2a.md) has the full setup.

```bash
pip install --pre "ovos-persona-server[a2a]"
ovos-persona-server --persona wiki.json --a2a-base-url http://localhost:8337/a2a
```

An `a2a-sdk` client that sent "What is the capital of Portugal? One word." got a completed task with the text `Lisbon.`.

Agents that speak the OpenAI API can use `POST /openai/v1/chat/completions` with `"model": "wiki"` instead.

## Use the tools from Claude Desktop

Claude Desktop starts local MCP servers as processes from `claude_desktop_config.json`. That file takes a command, not a URL. Use the `ovos-persona-tools-mcp` command, which serves the same tools over stdio and needs no running server.

1. Install the server with the `mcp` extra and your toolbox plugins, as in [Let Claude Code use the persona tools](#let-claude-code-use-the-persona-tools).
2. Add this entry to `claude_desktop_config.json`. Use absolute paths:

   ```json
   {
     "mcpServers": {
       "ovos-persona": {
         "command": "/home/me/.venv/bin/ovos-persona-tools-mcp",
         "env": {"PERSONA_PATH": "/home/me/personas/wiki.json"}
       }
     }
   }
   ```

   `PERSONA_PATH` names the persona file that holds the toolbox sections. A toolbox that needs no settings works without it.

3. Quit and restart Claude Desktop. The `ovos-persona` server and its tools show in the connectors menu.

This command was run with the reference MCP client in place of Claude Desktop. It listed `search_wikipedia`, and a call with `{"query": "Porto", "lang": "en"}` returned the Wikipedia summary of Porto.

A custom connector in the Claude settings is a different mechanism. It takes a public HTTPS URL. The server has no authentication, so do not expose it that way. See [Security](#security).

## Let opencode use the persona tools

You get the tools of every installed toolbox in opencode.

1. Install the server and start it with `--mcp`, as in [Let Claude Code use the persona tools](#let-claude-code-use-the-persona-tools).
2. Add this entry to `opencode.json` in your project folder:

   ```json
   {
     "$schema": "https://opencode.ai/config.json",
     "mcp": {
       "ovos-persona": {
         "type": "remote",
         "url": "http://localhost:8337/mcp",
         "enabled": true
       }
     }
   }
   ```

3. Check the connection:

   ```console
   $ opencode mcp list
   ┌  MCP Servers
   │
   ●  ✓ ovos-persona connected
   │      http://localhost:8337/mcp
   │
   └  1 server(s)
   ```

opencode can also start the stdio command, with no running server. Use a
`local` entry with absolute paths:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "ovos-persona": {
      "type": "local",
      "command": ["/home/me/.venv/bin/ovos-persona-tools-mcp"],
      "environment": {"PERSONA_PATH": "/home/me/personas/wiki.json"},
      "enabled": true
    }
  }
}
```

## Let Mistral Vibe use the persona tools

You get the tools of every installed toolbox in Vibe.

1. Install the server and start it with `--mcp`, as in [Let Claude Code use the persona tools](#let-claude-code-use-the-persona-tools).
2. Add this entry to `~/.vibe/config.toml`:

   ```toml
   [[mcp_servers]]
   name = "ovos-persona"
   transport = "streamable-http"
   url = "http://localhost:8337/mcp"
   ```

   Write the entry in the file. Do not use `vibe mcp add` for this server: that
   command sets OAuth login for an HTTP server, and the server does not use
   OAuth.

3. Start `vibe`. Vibe loads the tools of the server when it starts. It gives
   each tool the server name as a prefix, and it changes each hyphen to `_`,
   for example `ovos_persona_search_wikipedia`.

Vibe can also start the stdio command, with no running server. Use a `stdio`
entry with absolute paths:

```toml
[[mcp_servers]]
name = "ovos-persona"
transport = "stdio"
command = "/home/me/.venv/bin/ovos-persona-tools-mcp"
env = { PERSONA_PATH = "/home/me/personas/wiki.json" }
```

## Give the persona tools from another MCP server

Install `ovos-tool-adapters` with its `mcp` extra and add an `ovos-mcp-toolbox` section to the persona file. The persona then offers that server's tools to its chat engine and runs the calls itself.

This persona uses the reference time server, which `uvx` starts as a process:

```json
{
  "name": "agent",
  "solvers": ["ovos-chat-openai-plugin"],
  "ovos-chat-openai-plugin": {
    "api_url": "https://llama.smartgic.io/v1",
    "key": "none",
    "model": "qwen2.5:7b",
    "temperature": 0,
    "top_p": 1.0,
    "max_tokens": 1024
  },
  "ovos-mcp-toolbox": {
    "transport": "stdio",
    "command": "uvx",
    "args": ["mcp-server-time", "--local-timezone=Europe/Lisbon"],
    "timeout": 60
  }
}
```

```bash
pip install --pre ovos-persona-server "ovos-tool-adapters[mcp]" ovos-openai-plugin
ovos-persona-server --persona agent.json
curl -s http://localhost:8337/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "agent", "messages": [{"role": "user", "content": "What is the current time in the Asia/Tokyo timezone?"}]}'
```

The answer came from the tool. The request was sent at 18:27 UTC:

```
The current time in the Asia/Tokyo timezone is Tuesday, October 20, 2026 at 3:27 AM JST (Japan Standard Time).
```

To use an MCP server over HTTP, such as another persona server or `ovos-translate-server` started with `--mcp`, use this section:

```json
"ovos-mcp-toolbox": {"transport": "http", "url": "http://localhost:9686/mcp", "timeout": 30}
```

One persona file configures one MCP server. When two toolboxes export a tool with the same name, both names get the toolbox as a prefix, for example `ovos-wikipedia-tools__search_wikipedia` and `ovos-mcp-toolbox__search_wikipedia`.

## Give the persona tools from a UTCP manual

Add an `ovos-utcp-toolbox` section that points at a UTCP manual, for example the `/utcp` endpoint of another OVOS server.

Install `ovos-tool-adapters` with its `utcp` extra:

```bash
pip install --pre "ovos-tool-adapters[utcp]"
```

```json
"ovos-utcp-toolbox": {
  "utcp_config": {
    "manual_call_templates": [
      {"name": "wiki_peer", "call_template_type": "http",
       "url": "http://peer-host:8337/utcp", "http_method": "GET"}
    ]
  },
  "timeout": 30
}
```

UTCP tool names carry the manual name as a prefix. This persona found `wiki_peer.search_wikipedia` and, asked about Guimarães with that tool, answered:

```
Guimarães is a city and municipality located in northern Portugal, known for its historic town centre, which was listed as a UNESCO World Heritage Site in 2001.
```

The peer server logged the call as `POST /tools/search_wikipedia HTTP/1.1" 200 OK`.

The toolbox gives the chat engine the `body` argument but not the names of the tool arguments inside it, so a model can fail to call the tool with the right arguments.

## Let a UTCP agent discover the tools

Point the agent at `/utcp`. `--mcp` is not needed.

```bash
curl -s http://localhost:8337/utcp
```

Each tool takes one argument, `body`, which holds the tool's own arguments. The server reads that object as the JSON body of `POST /tools/{name}`. This is the UTCP client code that was run, with `utcp` 1.2 and `utcp-http`:

```python
import asyncio
from utcp.utcp_client import UtcpClient
from utcp.data.utcp_client_config import UtcpClientConfig

async def main():
    client = await UtcpClient.create(config=UtcpClientConfig(manual_call_templates=[
        {"name": "ovos_persona", "call_template_type": "http",
         "url": "http://localhost:8337/utcp", "http_method": "GET"}]))
    print([t.name for t in await client.search_tools("", limit=50)])
    print(await client.call_tool("ovos_persona.search_wikipedia",
                                 {"body": {"query": "Lisbon", "lang": "en"}}))

asyncio.run(main())
```

Output (summary shortened):

```
['ovos_persona.search_wikipedia']
{'results': [['Lisbon', 'Lisbon is the capital and most populous city of Portugal, ...']]}
```

## Call the server from your own MCP client

Use the `mcp` Python package, version 2 or later:

```python
import asyncio, json
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    async with streamable_http_client("http://localhost:8337/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print([t.name for t in (await session.list_tools()).tools])
            result = await session.call_tool("search_wikipedia", {"query": "Braga", "lang": "en"})
            print(result.is_error, json.loads(result.content[0].text)["results"][0][0])

asyncio.run(main())
```

Output:

```
['search_wikipedia']
False Braga
```

## Reference

### MCP

The MCP server is named `ovos-persona`. It has one tool for each tool of each installed toolbox, with the toolbox's own name, description and argument schema.

A successful call returns one text item: the tool output as a JSON string. A failed call returns `isError: true` and the reason as text, for example:

```
Tool input validation failed for 'search_wikipedia' in ToolBox 'ovos-wikipedia-tools'
```

The HTTP endpoint is `/mcp` (streamable HTTP). `/mcp` and `/mcp/` both answer with no redirect. There is no `/mcp/sse` path.

### UTCP

`GET /utcp` and `GET /tools/manual` return the same UTCP 1.0 manual. `utcp_version` is `1.0.0`, and `manual_version` is the version of the installed `ovos-persona-server`. Each tool has:

| Field | Value |
| :--- | :--- |
| `inputs` | An object with one required property, `body`, whose schema is the tool's argument schema |
| `outputs` | The tool's output schema |
| `tool_call_template` | `call_template_type: http`, `http_method: POST`, `url: <base>/tools/<name>`, `content_type: application/json`, `body_field: body` |

`POST /tools/{name}` takes the tool arguments as a JSON object and returns the tool output as JSON. It returns 404 for an unknown tool, 422 when the tool rejects its input and 500 when the tool fails.

### Toolbox configuration

A toolbox receives the section of the persona file that has its plugin name, such as `ovos-mcp-toolbox`, `ovos-utcp-toolbox` or `ovos-wikipedia-tools`. Toolboxes belong to the process, not to one persona:

- A running server reads the sections from its default persona, the one that `--persona`, `--personas-dir` or `PERSONA_PATH` loaded.
- `ovos-persona-tools-mcp` reads them from the file in `PERSONA_PATH`.

Every installed toolbox is loaded, and every persona of the process gets the same tools.

### Python

```python
from ovos_persona_server.mcp_server import build_mcp, mount_mcp
from ovos_persona_server.utcp import build_utcp_manual, make_utcp_router
```

- `build_mcp(name="ovos-persona")` returns the `FastMCP` server.
- `mount_mcp(app, path="/mcp")` mounts it on a FastAPI app at exactly `path` and starts its session manager with the app.
- `build_utcp_manual(base_url)` returns the manual as a dict.
- `make_utcp_router()` returns the router for `/utcp`, `/tools/manual` and `/tools/{name}`.

`create_persona_app(..., enable_mcp=True)` does the same as `--mcp`. The names `build_mcp_server`, `mount_mcp_on_app` and `utcp.utcp_router` still work and issue a `DeprecationWarning` that names their removal release.

## Deployment

### Docker

The base image installs the `mcp` and `a2a` extras. To serve MCP, add `"--mcp"` to the `command` list in `docker-compose.yml`:

```yaml
    command:
      - "--persona"
      - "/config/persona.json"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8337"
      - "--mcp"
```

The image has no toolbox plugin. Run `docker compose build` first, which tags the base image as `ghcr.io/openvoiceos/ovos-persona-server:dev`. Then build your own image on top of it and install the toolboxes your personas use:

```dockerfile
FROM ghcr.io/openvoiceos/ovos-persona-server:dev
RUN pip install --no-cache-dir --pre ovos-wikipedia-plugin "ovos-tool-adapters[mcp]"
```

A stdio MCP toolbox runs its command inside the container, so that command must be installed in the image too. An HTTP MCP toolbox or a UTCP toolbox only needs the other server to be reachable from the container.

### Reverse proxy

The tool URLs in the manual come from each request. The proxy must forward the `Host` header. Uvicorn reads `X-Forwarded-Proto` only from a proxy on `127.0.0.1`. These results were measured:

| Request | Tool URL in the manual |
| :--- | :--- |
| Direct to `127.0.0.1:8337` | `http://127.0.0.1:8337/tools/search_wikipedia` |
| `Host: persona.example.org` | `http://persona.example.org/tools/search_wikipedia` |
| `Host: persona.example.org` and `X-Forwarded-Proto: https` | `https://persona.example.org/tools/search_wikipedia` |
| `X-Forwarded-Host: persona.example.org` only | `http://127.0.0.1:8337/tools/search_wikipedia` (header ignored) |
| `GET /utcp?request_base_url=https://ai.example.org/persona` | `https://ai.example.org/persona/tools/search_wikipedia` |

When the proxy runs on another address, such as another container, start the server with `FORWARDED_ALLOW_IPS` set to the proxy's address, for example `FORWARDED_ALLOW_IPS=172.18.0.5 ovos-persona-server ...`. Uvicorn then trusts the proxy's `X-Forwarded-Proto` header, and the tool URLs use `https://`.

Use `request_base_url` when the server sits under a path prefix or the proxy cannot forward the host.

### Security

`/mcp`, `/utcp` and `/tools/{name}` have no authentication. Anybody who reaches the port can run every installed tool with the permissions of the server process. The server binds `0.0.0.0` by default and allows every CORS origin.

- Bind to `127.0.0.1` (`--host 127.0.0.1`) when only local agents use the server.
- Put a reverse proxy with TLS and authentication in front of it when other machines need it.
- Install only the toolboxes you want every caller to use.
- An `ovos-mcp-toolbox` stdio section starts its `command` on the server. Treat persona files as code.

## Troubleshooting

**`--mcp was set but the mcp extra is not installed`.** The server started without `/mcp`. Install the extra and restart: `pip install --pre "ovos-persona-server[mcp]"`.

**The manual or the MCP tool list is empty.** No toolbox loaded. Check the server log for `Failed to load ToolBox plugin <name>`, and check that the toolbox section in the persona file has the plugin name.

**`MCPToolBox: failed to connect to MCP server: 'command'`.** `ovos-tool-adapters` is installed, but the persona file has no `ovos-mcp-toolbox` section, so the toolbox has no settings. Add the section, or ignore the warning when you do not use that toolbox. The same applies to `UTCPToolBox`.


**`claude mcp get` does not show `Connected`, or `opencode mcp list` does not show `connected`.** Check that the server started with `--mcp` and that the log shows `MCP server mounted at /mcp`. Use the URL `http://<host>:8337/mcp`, not `/mcp/sse`.


**The chat engine calls the wrong tool or the wrong arguments.** This is a model problem, not a protocol problem. In one run `qwen2.5:7b` called `convert_time` with a tool name as the time zone. Name the tool or the argument in the prompt, and use the sampling settings from [local-llm-wikipedia-tools.md](local-llm-wikipedia-tools.md).
