"""Checks single-server launch policy without starting a model server."""
import subprocess
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import server_manager
import start_dory


class ServerGuardTests(unittest.TestCase):
    def test_process_list_identifies_real_server_executables(self):
        output = ("UID PID PPID C STIME TTY TIME CMD\n"
                  "u0 101 1 0 2022 ? 00:00:00 /build-a/bin/llama-server --port 8080\n"
                  "u0 102 1 0 2022 ? 00:00:00 /build-b/bin/llama-server --port 8081\n"
                  "u0 103 1 0 2022 ? 00:00:00 sh -c echo llama-server\n")
        with patch("server_manager.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")):
            self.assertEqual(server_manager.running_llama_servers(), [(101, "/build-a/bin/llama-server"),
                                                                     (102, "/build-b/bin/llama-server")])
            with self.assertRaisesRegex(RuntimeError, "Multiple llama-server"):
                server_manager.assert_single_llama_server()

    def test_managed_launcher_refuses_server_on_another_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = server_manager.LlamaServerManager(state_dir=Path(tmp))
            with patch("server_manager.running_llama_servers", return_value=[(123, "/other/llama-server")]), \
                 patch.object(manager, "check_health", return_value=False), \
                 patch("server_manager.subprocess.Popen") as launch, \
                 self.assertRaisesRegex(RuntimeError, "refusing to start another"):
                manager.ensure_server(log_file=Path(tmp) / "server.log")
            launch.assert_not_called()

    def test_dory_launcher_refuses_existing_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "launcher.sh"
            launcher.touch()
            with patch.dict("os.environ", {"DORY_LAUNCHER": str(launcher)}), \
                 patch("server_manager.running_llama_servers", return_value=[(123, "/other/llama-server")]), \
                 patch("start_dory.os.execv") as launch, \
                 self.assertRaisesRegex(RuntimeError, "already running"):
                start_dory.main()
            launch.assert_not_called()

    def test_process_inspection_failure_is_not_treated_as_empty(self):
        with patch("server_manager.subprocess.run", side_effect=OSError("not available")), \
             self.assertRaisesRegex(RuntimeError, "Cannot inspect"):
            server_manager.running_llama_servers()

    def test_guarded_launchers_cannot_acquire_lock_together(self):
        with server_manager.server_launch_lock():
            with self.assertRaisesRegex(RuntimeError, "launch is active"):
                with server_manager.server_launch_lock():
                    self.fail("second launcher acquired the guard")

    def test_guarded_snapdragon_wrapper_forwards_settings_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "env.sh").write_text(":\n")
            guard = root / "guard.sh"
            guard.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$GUARD_CAPTURE"\n')
            guard.chmod(0o700)
            capture = root / "args.txt"
            env = dict(os.environ, SNAPDRAGON_DIR=tmp, GUARD_CAPTURE=str(capture), CTX_SIZE="2048")
            wrapper = Path(server_manager.__file__).parent / "guarded_serve_best.sh"
            subprocess.run(["bash", str(wrapper), "gemma4-htp", "8899"], env=env, check=True)
            args = capture.read_text().splitlines()
            self.assertIn("--ctx-size", args)
            self.assertIn("2048", args)
            self.assertIn("8899", args)
            self.assertIn("HTP0", args)


if __name__ == "__main__":
    unittest.main()
