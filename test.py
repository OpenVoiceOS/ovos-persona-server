import openai

openai.api_key = ""
openai.api_base = "http://localhost:8337"


def main():
    # NOTE - most solvers don't support a chat history,
    #  only last message in messages list is considered
    chat_completion = openai.ChatCompletion.create(
        model="",  # individual personas might support this, passed under context
        messages=[{"role": "user", "content": "write a poem about a tree"}],
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


if __name__ == "__main__":
    main()