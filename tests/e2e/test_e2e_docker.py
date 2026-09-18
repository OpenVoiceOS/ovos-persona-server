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
compatible ``/openai/v1`` surface plus the models route used by the
compose healthcheck.

It runs with the rest of the e2e suite; the docker CLI is required and its
absence is a failure, not a skip.

    pytest tests/e2e/test_e2e_docker.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import urllib.error
import urllib.request

import pytest

import openai

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMAGE = "ovos-persona-server:e2e-test"
_CONTAINER = "ovos-persona-server-e2e-test"
_PORT = 18937


def _run(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)


def _get(url: str):
    """Return the decoded JSON body of a 200 answer, or None while the server is not up."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read()) if resp.status == 200 else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("config")
    (cfg / "persona.json").write_text(
        json.dumps({"name": "Failer", "solvers": ["ovos-solver-failure-plugin"]})
    )

    if shutil.which("docker") is None:
        pytest.fail("the docker CLI is required for this e2e; it is not a skip")
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
        if _get(f"{url}/openai/v1/models") is not None:
            break
        time.sleep(1)
    else:
        logs = subprocess.run(["docker", "logs", _CONTAINER], capture_output=True, text=True)
        subprocess.run(["docker", "rm", "-f", _CONTAINER], capture_output=True)
        raise RuntimeError(f"container did not become healthy:\n{logs.stdout}\n{logs.stderr}")

    yield url

    subprocess.run(["docker", "rm", "-f", _CONTAINER], capture_output=True)


def test_models_healthcheck(base_url):
    """The /openai/v1/models route used by the compose healthcheck lists the persona."""
    data = _get(f"{base_url}/openai/v1/models")
    assert [m["id"] for m in data["data"]] == ["Failer"]


def _failure_lines() -> list:
    """The en-US lines the failure plugin inside the container can answer with."""
    out = _run("docker", "exec", _CONTAINER, "python3", "-c",
               "import os, ovos_solver_failure_plugin as p;"
               "print(open(os.path.join(os.path.dirname(p.__file__), 'locale', 'en-US', 'no_brain.dialog')).read())",
               timeout=30).stdout
    return [l for l in out.split("\n") if l.strip() and not l.startswith("#")]


def test_openai_sdk_chat_completion(base_url):
    """The official openai SDK round-trips against the deployed /openai/v1 surface."""
    client = openai.OpenAI(base_url=f"{base_url}/openai/v1", api_key="not-needed")
    resp = client.chat.completions.create(
        model="Failer",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    # the failure plugin answers one of its en-US lines; versions that cannot
    # resolve the locale answer "404"
    assert resp.choices[0].message.content in [*_failure_lines(), "404"]
