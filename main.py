"""classroom-noise-ai entry point.

Usage:
    python main.py init-db      # create the SQLite schema
    python main.py devices      # list audio devices
    python main.py api          # FastAPI backend only
    python main.py web          # Next.js dashboard only (needs Node.js; first run does `npm install`)
    python main.py all          # backend + Next.js dashboard
    python main.py ui-legacy    # the previous Streamlit dashboard (optional, pip install -r requirements-legacy-ui.txt)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

from config import BASE_DIR, get_settings


def _api_cmd(settings) -> list:
    return [sys.executable, "-m", "uvicorn", "api.main:app", "--host", settings.api_host, "--port", str(settings.api_port)]


def _web_cmd(settings) -> list:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        raise SystemExit("Node.js / npm not found. Install Node.js 20 or 22 LTS from https://nodejs.org then retry.")
    return [npm, "run", "dev", "--", "-p", str(settings.web_port)]


def _ensure_web_deps() -> None:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm and not (BASE_DIR / "web" / "node_modules").exists():
        print("First run: installing the web dashboard dependencies (npm install)...")
        subprocess.check_call([npm, "install"], cwd=BASE_DIR / "web")


def _ui_cmd(settings) -> list:
    return [
        sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "ui" / "app.py"),
        "--server.port", str(settings.ui_port), "--browser.gatherUsageStats", "false",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="classroom-noise-ai")
    parser.add_argument("command", choices=["init-db", "devices", "api", "web", "ui-legacy", "ui", "all"])
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

    api_url = f"http://{settings.api_host}:{settings.api_port}"
    env = {**os.environ, "CNAI_API_URL": api_url, "NEXT_PUBLIC_API_URL": api_url}
    if args.command == "api":
        return subprocess.call(_api_cmd(settings), cwd=BASE_DIR)
    if args.command in ("ui", "ui-legacy"):
        return subprocess.call(_ui_cmd(settings), cwd=BASE_DIR, env=env)
    if args.command == "web":
        _ensure_web_deps()
        return subprocess.call(_web_cmd(settings), cwd=BASE_DIR / "web", env=env)

    _ensure_web_deps()
    procs = [subprocess.Popen(_api_cmd(settings), cwd=BASE_DIR),
             subprocess.Popen(_web_cmd(settings), cwd=BASE_DIR / "web", env=env)]
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
