"""classroom-noise-ai entry point.

Usage:
    python main.py init-db      # create the SQLite schema
    python main.py devices      # list audio devices
    python main.py api          # FastAPI backend only
    python main.py ui           # Streamlit dashboard only
    python main.py all          # backend + dashboard
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from config import BASE_DIR, get_settings


def _api_cmd(settings) -> list:
    return [sys.executable, "-m", "uvicorn", "api.main:app", "--host", settings.api_host, "--port", str(settings.api_port)]


def _ui_cmd(settings) -> list:
    return [
        sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "ui" / "app.py"),
        "--server.port", str(settings.ui_port), "--browser.gatherUsageStats", "false",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="classroom-noise-ai")
    parser.add_argument("command", choices=["init-db", "devices", "api", "ui", "all"])
    args = parser.parse_args()
    settings = get_settings()

    if args.command == "init-db":
        from database.db_session import init_db

        init_db()
        print(f"Database ready: {settings.sqlite_url}")
        return 0

    if args.command == "devices":
        from core.audio_stream import AudioStream

        for dev in AudioStream.list_devices():
            print(f"[{dev['index']:>2}] in={dev['max_input_channels']} out={dev['max_output_channels']}  {dev['name']}")
        return 0

    env = {**os.environ, "CNAI_API_URL": f"http://{settings.api_host}:{settings.api_port}"}
    if args.command == "api":
        return subprocess.call(_api_cmd(settings), cwd=BASE_DIR)
    if args.command == "ui":
        return subprocess.call(_ui_cmd(settings), cwd=BASE_DIR, env=env)

    procs = [subprocess.Popen(_api_cmd(settings), cwd=BASE_DIR), subprocess.Popen(_ui_cmd(settings), cwd=BASE_DIR, env=env)]
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
