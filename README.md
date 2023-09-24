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

personas don't need to use LLMs, find solver plugins [here](https://github.com/OpenVoiceOS?q=solver&type=all)

```
{
  "name": "DictionaryMan",
  "gender": "male",
  "description": "answers 'what is' questions using wordnet",
  "solvers": [
    "ovos-solver-wordnet-plugin",
    "ovos-solver-failure-plugin"
  ]
},

{
  "name": "InfoSeeker",
  "gender": "male",
  "description": "answers questions using duckduckgo, wikipedia and wolfram alpha",
  "solvers": [
    "ovos-solver-wikipedia-plugin",
    "ovos-solver-ddg-plugin",
    "ovos-solver-wofram-plugin",
    "ovos-solver-failure-plugin"
  ]
},

{
  "name": "AIMLia",
  "gender": "female",
  "description": "chatbot without internet access incapable of factual answers, but can do basic chitchat, powered by using AIML",
  "solvers": [
    "ovos-solver-aiml-plugin"
  ]
}
```