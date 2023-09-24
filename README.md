# Persona Server

## Running

## Usage

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