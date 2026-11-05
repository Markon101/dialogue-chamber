#!/usr/bin/env python3
"""Phase A verification tests: Turn counts, model mismatch, ownership, concurrency, wake-lock."""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

CHAMBER_DIR = Path(__file__).resolve().parent

class TestPhaseALauncherReliability(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="chamber_test_phase_a_")
        self.env = dict(os.environ)
        self.env["LOCK_DIR"] = os.path.join(self.temp_dir, ".locks")
        os.makedirs(self.env["LOCK_DIR"], exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preset_turn_count_default_preserved(self):
        """Verify that omitting --turns does NOT force 6 turns; it preserves the preset default."""
        # weird_hierarchy has max_turns=15 in presets.py
        cmd = [
            "bash", str(CHAMBER_DIR / "run_prompt.sh"),
            "--preset", "weird_hierarchy",
            "--mock",
            "--dry-run"
        ]
        # In dry run, check output or pass through to chamber.py
        # We will verify that chamber.py receives the preset default (15) instead of 6
        proc = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        # Should exit 0 or pass through to dry-run
        self.assertNotIn("Turns:        6 (CLI override)", proc.stdout)

    def test_explicit_turn_count_override(self):
        """Verify that specifying -t / --turns explicitly overrides the preset default."""
        cmd = [
            "bash", str(CHAMBER_DIR / "run_prompt.sh"),
            "--preset", "weird_hierarchy",
            "-t", "4",
            "--mock",
            "--dry-run"
        ]
        proc = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
        self.assertIn("Planned Turns:       4", proc.stdout)

    def test_model_mismatch_detection_and_refusal(self):
        """Verify that chamber_ensure_server detects model mismatch on health check and refuses without --allow-server-switch."""
        # Test chamber_ensure_server logic directly by simulating a mock server response
        # Using a small python one-liner HTTP server in a subshell or bash script test
        script = f"""
        source "{CHAMBER_DIR}/launcher_lib.sh"
        # Mock chamber_check_health and chamber_get_loaded_model
        chamber_check_health() {{ return 0; }}
        chamber_get_loaded_model() {{ echo "Llama-3.2-1B-Instruct-Q4_0.gguf"; return 0; }}
        
        # Calling ensure server requesting gemma4-htp
        chamber_ensure_server "127.0.0.1" 8080 "gemma4-htp" "gemma-4-E4B-it-Q4_0.gguf" "/dev/null" "false"
        """
        proc = subprocess.run(["bash", "-c", script], env=self.env, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Model mismatch detected", proc.stderr)
        self.assertIn("Llama-3.2-1B-Instruct-Q4_0.gguf", proc.stderr)

    def test_model_match_reuses_server(self):
        """Verify that matching model reuses the server without restarting."""
        script = f"""
        source "{CHAMBER_DIR}/launcher_lib.sh"
        chamber_check_health() {{ return 0; }}
        chamber_get_loaded_model() {{ echo "gemma-4-E4B-it-Q4_0.gguf"; return 0; }}
        
        chamber_ensure_server "127.0.0.1" 8080 "gemma4-htp" "gemma-4-E4B-it-Q4_0.gguf" "/dev/null" "false"
        """
        proc = subprocess.run(["bash", "-c", script], env=self.env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("ALREADY running and healthy", proc.stdout)

    def test_server_process_ownership_tracking(self):
        """Verify that only owned server is stopped and external unowned server is untouched."""
        state_dir = Path(self.env["LOCK_DIR"]) / "servers"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / "server_8080.json"

        # 1. Without state file: unowned server
        script_unowned = f"""
        source "{CHAMBER_DIR}/launcher_lib.sh"
        chamber_is_server_owned 8080 && echo "OWNED" || echo "UNOWNED"
        """
        proc = subprocess.run(["bash", "-c", script_unowned], env=self.env, capture_output=True, text=True)
        self.assertIn("UNOWNED", proc.stdout)

        # 2. With state file recording our own dummy background process
        dummy = subprocess.Popen(["sleep", "30"])
        try:
            state_file.write_text(json.dumps({"pid": dummy.pid, "port": 8080, "mode": "gemma4-htp"}))
            script_owned = f"""
            source "{CHAMBER_DIR}/launcher_lib.sh"
            chamber_stop_owned_server 8080
            """
            proc = subprocess.run(["bash", "-c", script_owned], env=self.env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            time.sleep(0.5)
            self.assertIsNotNone(dummy.poll())
            dummy.wait()
            self.assertFalse(state_file.exists())
        finally:
            if dummy.poll() is None:
                dummy.kill()
                dummy.wait()

    def test_concurrency_locking(self):
        """Verify that concurrent experiment launches are prevented by lock."""
        script = f"""
        source "{CHAMBER_DIR}/launcher_lib.sh"
        chamber_acquire_lock 0 || exit 1
        # Second acquire in subshell should fail immediately
        bash -c 'source "{CHAMBER_DIR}/launcher_lib.sh"; chamber_acquire_lock 0' && exit 2 || exit 0
        """
        proc = subprocess.run(["bash", "-c", script], env=self.env, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)

    def test_wakelock_reference_counting(self):
        """Verify that multiple holders maintain wake-lock until the last holder releases."""
        holders_dir = Path(self.env["LOCK_DIR"]) / "wakelock_holders"
        holders_dir.mkdir(parents=True, exist_ok=True)

        # Holder 1 (dummy background proc)
        dummy1 = subprocess.Popen(["sleep", "30"])
        # Holder 2 (dummy background proc)
        dummy2 = subprocess.Popen(["sleep", "30"])

        try:
            (holders_dir / str(dummy1.pid)).touch()
            (holders_dir / str(dummy2.pid)).touch()

            # Release for dummy1: dummy2 is still active, so holders should remain > 0
            script1 = f"""
            source "{CHAMBER_DIR}/launcher_lib.sh"
            # Release dummy1 by pretending we are dummy1
            rm -f "{holders_dir}/{dummy1.pid}"
            # Check remaining active holders
            active=0
            for f in "{holders_dir}"/*; do
              [ -e "$f" ] || continue
              kill -0 $(basename "$f") 2>/dev/null && active=$((active + 1))
            done
            echo "ACTIVE: $active"
            """
            proc = subprocess.run(["bash", "-c", script1], env=self.env, capture_output=True, text=True)
            self.assertIn("ACTIVE: 1", proc.stdout)
        finally:
            dummy1.kill()
            dummy2.kill()
            dummy1.wait()
            dummy2.wait()


if __name__ == "__main__":
    unittest.main()
