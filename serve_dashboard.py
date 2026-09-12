"""Serve the generated dashboard on localhost."""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    dashboard = root / "frontend" / "index.html"
    if not dashboard.exists():
        raise SystemExit(
            "frontend/index.html not found. Run `bash run.sh --fast --seeds 0` first."
        )

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(root), **handler_kwargs)

    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        url = f"http://localhost:{args.port}/frontend/index.html"
        print(f"Dashboard running at {url}")
        print("Press Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
