# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end test for the deployed Docker image.

Builds the image from the repo Dockerfile, runs the container with a
network-free persona (the failure solver returns a fixed reply), and drives
the deployed server with the official ``openai`` SDK against the OpenAI-
compatible ``/v1`` surface plus the Ollama ``/api/tags`` endpoint used by the
compose healthcheck.

Building and running a container is heavy, so this is opt-in: set
``RUN_DOCKER_E2E=1`` to enable it. It is skipped by default (including in the
standard CI test run).

    RUN_DOCKER_E2E=1 pytest tests/e2e/test_e2e_docker.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import httpx
import pytest

if os.environ.get("RUN_DOCKER_E2E") != "1":
    pytest.skip("docker e2e opt-in: set RUN_DOCKER_E2E=1", allow_module_level=True)

if shutil.which("docker") is None:
    pytest.skip("docker CLI not available", allow_module_level=True)

openai = pytest.importorskip("openai", reason="openai SDK not installed")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMAGE = "ovos-persona-server:e2e-test"
_CONTAINER = "ovos-persona-server-e2e-test"
_PORT = 18937


def _run(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("config")
    (cfg / "mycroft").mkdir()
    (cfg / "persona.json").write_text(
        json.dumps({"name": "Failer", "solvers": ["ovos-solver-failure-plugin"]})
    )
    (cfg / "mycroft" / "mycroft.conf").write_text(
        json.dumps({
            "lang": "en-US",
            "language": {"detection_module": "ovos-lang-detector-plugin-langdetect"},
        })
    )

    _run("docker", "build", "-t", _IMAGE, _REPO_ROOT)
    subprocess.run(["docker", "rm", "-f", _CONTAINER], capture_output=True)
    _run(
        "docker", "run", "-d", "--name", _CONTAINER,
        "-p", f"{_PORT}:8337", "-v", f"{cfg}:/config",
        _IMAGE,
        "--persona", "/config/persona.json", "--host", "0.0.0.0", "--port", "8337",
        timeout=60,
    )

    url = f"http://127.0.0.1:{_PORT}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/api/tags", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        subprocess.run(["docker", "rm", "-f", _CONTAINER], capture_output=True)
        raise RuntimeError("container did not become healthy")

    yield url

    subprocess.run(["docker", "rm", "-f", _CONTAINER], capture_output=True)


def test_ollama_tags_healthcheck(base_url):
    """The /api/tags endpoint used by the compose healthcheck serves the persona."""
    data = httpx.get(f"{base_url}/api/tags", timeout=10).json()
    assert any(m["name"] == "Failer" for m in data["models"])


def test_openai_sdk_chat_completion(base_url):
    """The official openai SDK round-trips against the deployed /v1 surface."""
    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key="not-needed")
    resp = client.chat.completions.create(
        model="Failer",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    # the failure solver returns a deterministic canned reply
    assert resp.choices[0].message.content == "404"
