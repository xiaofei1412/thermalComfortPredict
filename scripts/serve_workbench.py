"""Run the unchanged workbench against an explicitly selected local engine."""

import argparse
from urllib.parse import urlsplit

from scripts.install_demo import install
from thermal_comfort.simulation import Client
from thermal_comfort.workbench import Handler, LocalServer, ROOT, Workbench


def engine_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/") or parsed.port is None or not 1024 <= parsed.port <= 65535):
        raise ValueError("Engine URL must be a local HTTP root with an explicit port")
    return value.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--engine-url", type=engine_url, default="http://127.0.0.1:8080")
    args = parser.parse_args()
    install()
    app = Workbench(ROOT/"outputs/workbench/sessions", client=Client(args.engine_url))
    try:
        server = LocalServer(("127.0.0.1", args.port), Handler)
    except OSError:
        parser.exit(2, "Port occupied; choose another --port.\n")
    server.app = app
    print(f"Workbench http://127.0.0.1:{args.port}; local engine {args.engine_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.cancel.set()
        if app.thread:
            app.thread.join(timeout=310)
        if not app.busy:
            app.release()
            if app.session and app.session["status"] in ("paused", "initializing"):
                app.session["status"] = "interrupted"
                app.session["error"] = "workbench_shutdown_without_final_kpi"
                app.save()
        server.server_close()


if __name__ == "__main__":
    main()
