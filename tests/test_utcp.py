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
"""UTCP endpoints: the manual shape, its loading by the real UTCP 1.x client
models, the base URL, and tool invocation over ``POST /tools/{name}``."""

from typing import Any, Dict

from fastapi.testclient import TestClient
from pydantic import Field

from ovos_plugin_manager.templates.agent_tools import (
    AgentTool, ToolArguments, ToolBox, ToolOutput,
)

from tests.test_tools import _build_fake_registry


# ---------------------------------------------------------------------------
# UTCP endpoint tests
# ---------------------------------------------------------------------------

def _make_test_app(fake_reg):
    """Build a minimal FastAPI app with only the UTCP routes, using a fake registry."""
    from fastapi import FastAPI
    from ovos_persona_server import utcp as utcp_module

    app = FastAPI()

    utcp_module._registry = fake_reg
    utcp_module._registry_ready = True
    app.include_router(utcp_module.make_utcp_router())
    return app


class TestUTCPManual:
    def setup_method(self):
        self.app = _make_test_app(_build_fake_registry())
        self.client = TestClient(self.app)

    def test_manual_returns_200(self):
        resp = self.client.get("/tools/manual")
        assert resp.status_code == 200

    def test_utcp_path_serves_the_same_manual(self):
        assert self.client.get("/utcp").json() == self.client.get("/tools/manual").json()

    def test_build_utcp_manual_matches_the_endpoint(self):
        from ovos_persona_server.utcp import build_utcp_manual
        assert build_utcp_manual("http://testserver/") == self.client.get("/utcp").json()

    def test_utcp_router_alias_is_deprecated(self):
        import pytest
        from ovos_persona_server import utcp as utcp_module
        from ovos_persona_server.version import VERSION_MAJOR, VERSION_MINOR
        with pytest.warns(DeprecationWarning,
                          match=f"removed in {VERSION_MAJOR}.{VERSION_MINOR + 1}.0"):
            router = utcp_module.utcp_router
        assert {r.path for r in router.routes} == {"/utcp", "/tools/manual", "/tools/{name}"}

    def test_manual_utcp_version(self):
        from ovos_persona_server.version import __version__
        body = self.client.get("/tools/manual").json()
        assert body["utcp_version"] == "1.0.0"
        assert body["manual_version"] == __version__

    def test_manual_contains_both_tools(self):
        resp = self.client.get("/tools/manual")
        body = resp.json()
        names = {t["name"] for t in body["tools"]}
        assert "echo" in names
        assert "add" in names

    def test_manual_url_comes_from_the_request(self):
        """Without an override the tool URL names the host the client used."""
        body = self.client.get("/tools/manual").json()
        echo = next(t for t in body["tools"] if t["name"] == "echo")
        assert echo["tool_call_template"]["url"] == "http://testserver/tools/echo"

    def test_manual_call_template_is_utcp_http(self):
        body = self.client.get("/tools/manual").json()
        echo = next(t for t in body["tools"] if t["name"] == "echo")
        assert echo["tool_call_template"] == {
            "call_template_type": "http",
            "name": "ovos_persona_tools",
            "http_method": "POST",
            "url": "http://testserver/tools/echo",
            "content_type": "application/json",
            "body_field": "body",
        }

    def test_manual_inputs_wrap_the_argument_schema(self):
        body = self.client.get("/tools/manual").json()
        echo = next(t for t in body["tools"] if t["name"] == "echo")
        assert echo["inputs"]["required"] == ["body"]
        assert "message" in echo["inputs"]["properties"]["body"]["properties"]

    def test_manual_loads_in_the_utcp_client(self):
        """The real UTCP 1.x models accept every tool of the manual."""
        from utcp.data.utcp_manual import UtcpManualSerializer
        body = self.client.get("/tools/manual").json()
        manual = UtcpManualSerializer().validate_dict(body)
        assert sorted(t.name for t in manual.tools) == ["add", "echo"]
        template = next(t for t in manual.tools if t.name == "echo").tool_call_template
        assert template.http_method == "POST"
        assert template.body_field == "body"


class TestUTCPInvokeEndpoint:
    def setup_method(self):
        self.app = _make_test_app(_build_fake_registry())
        self.client = TestClient(self.app)

    def test_invoke_echo_200(self):
        resp = self.client.post("/tools/echo", json={"message": "world"})
        assert resp.status_code == 200
        assert resp.json() == {"echo": "world"}

    def test_invoke_add_200(self):
        resp = self.client.post("/tools/add", json={"a": 1.5, "b": 2.5})
        assert resp.status_code == 200
        assert resp.json() == {"result": 4.0}

    def test_invoke_unknown_404(self):
        resp = self.client.post("/tools/nonexistent", json={})
        assert resp.status_code == 404

    def test_invoke_missing_required_arg_422(self):
        # 'message' is required for echo; omitting it should fail validation
        resp = self.client.post("/tools/echo", json={})
        assert resp.status_code == 422


class TestUTCPEdgeCases:
    def setup_method(self):
        from fastapi import FastAPI
        from ovos_persona_server import utcp as utcp_module
        app = FastAPI()
        utcp_module._registry = _build_fake_registry()
        utcp_module._registry_ready = True
        app.include_router(utcp_module.make_utcp_router())
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_manual_empty_registry(self):
        """Manual endpoint with empty registry returns tools=[]."""
        from fastapi import FastAPI
        from ovos_persona_server import utcp as utcp_module
        app2 = FastAPI()
        utcp_module._registry = {}
        utcp_module._registry_ready = True
        app2.include_router(utcp_module.make_utcp_router())
        from fastapi.testclient import TestClient
        client2 = TestClient(app2)
        resp = client2.get("/tools/manual")
        assert resp.status_code == 200
        assert resp.json()["tools"] == []

    def test_invoke_tool_500_on_unexpected_exception(self):
        """When a tool raises an unexpected exception, the endpoint returns 500."""
        class BoomArgs(ToolArguments):
            x: str = Field(default="x")

        class BoomOutput(ToolOutput):
            y: str = Field(default="y")

        class BoomBox(ToolBox):
            def __init__(self, config: Dict[str, Any] = None, bus: Any = None) -> None:
                super().__init__(toolbox_id="boom_box", config=config, bus=bus)

            def discover_tools(self):
                return [AgentTool(
                    name="crasher",
                    description="always crashes",
                    argument_schema=BoomArgs,
                    output_schema=BoomOutput,
                    tool_call=lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
                )]

        tb = BoomBox()
        reg = {"crasher": (tb, tb.tools["crasher"])}

        from fastapi import FastAPI
        from ovos_persona_server import utcp as utcp_module
        app3 = FastAPI()
        utcp_module._registry = reg
        utcp_module._registry_ready = True
        app3.include_router(utcp_module.make_utcp_router())
        from fastapi.testclient import TestClient
        client3 = TestClient(app3, raise_server_exceptions=False)
        resp = client3.post("/tools/crasher", json={"x": "hi"})
        assert resp.status_code == 500

    def test_manual_request_base_url_override(self):
        """request_base_url query param is embedded in the tool URLs."""
        resp = self.client.get("/tools/manual?request_base_url=https://myproxy.example.com/")
        body = resp.json()
        echo = next((t for t in body["tools"] if t["name"] == "echo"), None)
        assert echo is not None
        assert echo["tool_call_template"]["url"] == "https://myproxy.example.com/tools/echo"
