import argparse
import os.path

from ovos_persona_server import get_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", help="path to persona .json file", required=True)
    parser.add_argument("--host", help="host", default="0.0.0.0")
    parser.add_argument("--port", help="port to run server", default=8337)
    args = parser.parse_args()

    app = get_app(os.path.expanduser(args.persona))

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
