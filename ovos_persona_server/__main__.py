"""
Command-line entry point for running the OVOS Persona Server.

This module parses command-line arguments to configure and start
the FastAPI application using Uvicorn.
"""

import argparse
from typing import Any

import uvicorn

from ovos_persona_server import create_persona_app



def main() -> None:
    """
    Main function to parse arguments and start the Persona Server.
    """
    parser = argparse.ArgumentParser(description="OVOS Persona Server")
    parser.add_argument("--persona", help="Path to persona .json file", default=None, type=str)
    parser.add_argument("--host", help="Host address to bind to", default="0.0.0.0", type=str)
    parser.add_argument("--port", help="Port to run server on", default=8337, type=int)
    parser.add_argument(
        "--a2a-base-url",
        help=(
            "Enable A2A endpoint at /a2a and set its public base URL "
            "(e.g. http://myhost:8337/a2a). Requires a2a-sdk installed."
        ),
        default=None,
        type=str,
    )
    args: Any = parser.parse_args()  # Using Any for args as argparse.Namespace is dynamic

    app = create_persona_app(args.persona, a2a_base_url=args.a2a_base_url)

    uvicorn.run(app, port=args.port, host=args.host, log_level="debug")


if __name__ == "__main__":
    main()
