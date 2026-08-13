# python >= 3.12 required: the server uses multi-line f-string expressions
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install from the repo source; ovos-workshop and uvicorn are required at
# runtime but not declared as dependencies; a language detector plugin is
# required by the solver chat pipeline
COPY . /src
RUN pip install --no-cache-dir /src \
                               ovos-workshop \
                               uvicorn \
                               ovos-lang-detector-classics-plugin \
                               langdetect \
                               ovos-solver-failure-plugin \
                               ovos-openai-plugin \
    && rm -rf /src

ENV XDG_CONFIG_HOME=/config
WORKDIR /app

EXPOSE 8337

ENTRYPOINT ["ovos-persona-server", "--host", "0.0.0.0", "--port", "8337"]
