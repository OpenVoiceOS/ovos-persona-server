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
"""Console entry point of ``ovos-persona-tools-mcp``: MCP over stdio.

stdout carries the JSON-RPC stream, so nothing else may write to it. The
OVOS loggers attach a handler to ``sys.stdout`` when they are created, and
importing ``ovos_persona_server`` already logs. This module lives outside
the package so that it can point ``sys.stdout`` at stderr before that
import runs.
"""
import sys


def main() -> None:
    stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from ovos_persona_server.mcp_server import _run_stdio
    finally:
        sys.stdout = stdout
    _run_stdio()


if __name__ == "__main__":
    main()
