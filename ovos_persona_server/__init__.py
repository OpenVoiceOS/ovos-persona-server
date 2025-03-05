import json
import os.path
import random
import string
import time
from typing import Any

from flask import Flask, request
from ovos_persona import Persona


def get_app(persona_json):
    app = Flask(__name__)

    with open(persona_json) as f:
        persona = json.load(f)
        persona["name"] = persona.get("name") or os.path.basename(persona_json)

    persona = Persona(persona["name"], persona)

    @app.route("/status", methods=["GET"])
    def status():
        return {"persona": persona.name,
                "solvers": list(persona.solvers.loaded_modules.keys()),
                "models": {s: persona.config.get(s, {}).get("model")
                           for s in persona.solvers.loaded_modules.keys()}}

    @app.route("/completions", methods=["POST"])
    def completions():
        prompt = request.get_json().get("prompt")

        completion_id = "".join(random.choices(string.ascii_letters + string.digits, k=28))
        completion_timestamp = int(time.time())

        response = persona.complete(prompt)

        return {"choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "text": response
            }
        ],
            "id": f"chatcmpl-{completion_id}",
            "created": completion_timestamp,
            "model": persona.name,
            "object": "text_completion",
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },

        }

    @app.route("/chat/completions", methods=["POST"])
    def chat_completions():
        data = request.get_json()
        stream = data.get("stream", False)
        messages = data.get("messages")

        response = persona.chat(messages)

        completion_id = "".join(random.choices(string.ascii_letters + string.digits, k=28))
        completion_timestamp = int(time.time())

        if not stream:
            return {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion",
                "created": completion_timestamp,
                "model": persona.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            }

        def streaming():
            for chunk in response:
                completion_data = {
                    "id": f"chatcmpl-{completion_id}",
                    "object": "chat.completion.chunk",
                    "created": completion_timestamp,
                    "model": persona.name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk,
                            },
                            "finish_reason": None,
                        }
                    ],
                }

                content = json.dumps(completion_data, separators=(",", ":"))
                yield f"data: {content}\n\n"
                time.sleep(0.1)

            end_completion_data: dict[str, Any] = {
                "id": f"chatcmpl-{completion_id}",
                "object": "chat.completion.chunk",
                "created": completion_timestamp,
                "model": persona.name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            content = json.dumps(end_completion_data, separators=(",", ":"))
            yield f"data: {content}\n\n"

        return app.response_class(streaming(), mimetype="text/event-stream")

    return app
