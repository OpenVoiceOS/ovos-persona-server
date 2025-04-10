import datetime
import json
import os.path
import random
import string
import time
from typing import Any

from flask import Flask, request
from ovos_bus_client.session import SessionManager
from ovos_persona import Persona


def get_app(persona_json):
    app = Flask(__name__)

    with open(persona_json) as f:
        persona = json.load(f)
        persona["name"] = persona.get("name") or os.path.basename(persona_json)

    persona = Persona(persona["name"], persona)

    #######
    @app.route("/status", methods=["GET"])
    def status():
        return {"persona": persona.name,
                "solvers": list(persona.solvers.loaded_modules.keys()),
                "models": {s: persona.config.get(s, {}).get("model")
                           for s in persona.solvers.loaded_modules.keys()}}

    ##############
    # OpenAI api compat
    @app.route("/chat/completions", methods=["POST"])
    def chat_completions():
        data = request.get_json()
        stream = data.get("stream", False)
        messages = data.get("messages")

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
                            "content": persona.chat(messages),
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
            for chunk in persona.stream(messages):
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

    ############
    # Ollama api compat
    @app.route("/api/chat", methods=["POST"])
    def chat():
        model = request.json.get("model")
        messages = request.json.get("messages")
        tools = request.json.get("tools")
        stream = request.json.get("stream")

        # Format timestamp to the desired format
        completion_timestamp = (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                                + f'.{int(time.time() * 1_000_000) % 1_000_000:06d}Z')

        sess = SessionManager().get()

        if not stream:
            ans = persona.chat(messages, lang=sess.lang, units=sess.system_unit)
            data = {
                "model": persona.name,
                "created_at": completion_timestamp,
                "message": {
                    "role": "assistant",
                    "content": ans,
                },
                "done": True
                # "context": [1, 2, 3],
                # "total_duration": 5043500667,
                # "load_duration": 5025959,
                # "prompt_eval_count": 26,
                # "prompt_eval_duration": 325953000,
                # "eval_count": 290,
                # "eval_duration": 4709213000
            }
            return data

        def streaming():
            for ans in persona.stream(messages, lang=sess.lang, units=sess.system_unit):
                data = {
                    "model": persona.name,
                    "created_at": completion_timestamp,
                    "message": {
                        "role": "assistant",
                        "content": ans
                    },
                    "done": False,
                    # "context": [1, 2, 3],
                    # "total_duration": 10706818083,
                    # "load_duration": 6338219291,
                    # "prompt_eval_count": 26,
                    # "prompt_eval_duration": 130079000,
                    # "eval_count": 259,
                    # "eval_duration": 4232710000
                }
                content = json.dumps(data)
                yield content + "\n"

            end_completion_data = {
                "model": persona.name,
                "created_at": completion_timestamp,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True,
                # "context": [1, 2, 3],
                # "total_duration": 10706818083,
                # "load_duration": 6338219291,
                # "prompt_eval_count": 26,
                # "prompt_eval_duration": 130079000,
                # "eval_count": 259,
                # "eval_duration": 4232710000
            }
            content = json.dumps(end_completion_data)
            yield content + "\n"

        return app.response_class(streaming(), mimetype="application/json")

    @app.route("/api/generate", methods=["POST"])
    def generate():
        model = request.json.get("model")
        prompt = request.json.get("prompt")
        suffix = request.json.get("suffix")
        system = request.json.get("system")
        template = request.json.get("template")
        stream = request.json.get("stream")

        sess = SessionManager().get()

        messages = [{
            "role": "user",
            "content": prompt
        }]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        # Format timestamp to the desired format
        completion_timestamp = (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                                + f'.{int(time.time() * 1_000_000) % 1_000_000:06d}Z')

        sess = SessionManager().get()

        if not stream:
            ans = persona.chat(messages, lang=sess.lang, units=sess.system_unit)
            data = {
                "model": persona.name,
                "created_at": completion_timestamp,
                "message": {
                    "role": "assistant",
                    "content": ans,
                },
                "done": True
                # "context": [1, 2, 3],
                # "total_duration": 5043500667,
                # "load_duration": 5025959,
                # "prompt_eval_count": 26,
                # "prompt_eval_duration": 325953000,
                # "eval_count": 290,
                # "eval_duration": 4709213000
            }
            return data

        def streaming():
            for ans in persona.stream(messages, lang=sess.lang, units=sess.system_unit):
                data = {
                    "model": persona.name,
                    "created_at": completion_timestamp,
                    "message": {
                        "role": "assistant",
                        "content": ans
                    },
                    "done": False,
                    # "context": [1, 2, 3],
                    # "total_duration": 10706818083,
                    # "load_duration": 6338219291,
                    # "prompt_eval_count": 26,
                    # "prompt_eval_duration": 130079000,
                    # "eval_count": 259,
                    # "eval_duration": 4232710000
                }
                content = json.dumps(data)
                yield content + "\n"

            end_completion_data = {
                "model": persona.name,
                "created_at": completion_timestamp,
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True,
                # "context": [1, 2, 3],
                # "total_duration": 10706818083,
                # "load_duration": 6338219291,
                # "prompt_eval_count": 26,
                # "prompt_eval_duration": 130079000,
                # "eval_count": 259,
                # "eval_duration": 4232710000
            }
            content = json.dumps(end_completion_data)
            yield content + "\n"

        return app.response_class(streaming(), mimetype="text/event-stream")

    @app.route("/api/tags", methods=["GET"])
    def tags():
        return {"models": [
            {"name": persona.name, "model": str(persona.solvers.sort_order[0])}
        ]}

    return app
