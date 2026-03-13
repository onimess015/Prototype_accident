from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path


def _find_open_port(start_port: int = 8501, max_attempts: int = 50) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No open port found for Streamlit in the checked range.")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    app_file = project_root / "app" / "streamlit_app.py"
    if not app_file.exists():
        raise FileNotFoundError(f"Streamlit app not found: {app_file}")

    port = _find_open_port()
    print(f"Starting Streamlit on port {port}...", flush=True)
    print(f"Local URL: http://localhost:{port}", flush=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
