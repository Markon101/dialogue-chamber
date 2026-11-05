#!/usr/bin/env python3
"""Start the separately installed Dory server under the shared launch guard."""
import os
import sys
from pathlib import Path

from server_manager import assert_single_llama_server, server_launch_lock


def main():
    launcher = Path(os.environ.get("DORY_LAUNCHER", str(Path.home() / "gitapps/dory-test/start-server.sh")))
    if not launcher.is_file():
        raise RuntimeError(f"Dory launcher not found: {launcher}")
    with server_launch_lock():
        servers = assert_single_llama_server()
        if servers:
            raise RuntimeError(f"A llama-server is already running (PID {servers[0][0]}); stop it before starting Dory")
        print(f"Starting Dory with {launcher}", flush=True)
        os.execv(str(launcher), [str(launcher)])


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"Dory was not started: {exc}", file=sys.stderr)
        sys.exit(1)
