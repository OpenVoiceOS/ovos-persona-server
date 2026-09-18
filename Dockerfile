FROM python:3.12-slim

# The published alpha with the A2A and MCP extras, the OpenAI-compatible chat
# engine, and the failure line that ends a persona chain.
RUN pip install --no-cache-dir --pre \
        'ovos-persona-server[a2a,mcp]' \
        ovos-openai-plugin \
        ovos-solver-failure-plugin

ENV XDG_CONFIG_HOME=/config
WORKDIR /app

EXPOSE 8337

ENTRYPOINT ["ovos-persona-server", "--host", "0.0.0.0", "--port", "8337"]
