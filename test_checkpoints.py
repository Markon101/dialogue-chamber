#!/usr/bin/env python3
"""Verification tests for Checkpointing, Recovery, Run Manifests, and Event Replay."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from checkpoint import CheckpointManager, write_atomic
from chamber import Chamber
from inference import MockBackend


class TestCheckpointsAndRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_chk_"))
        self.mgr = CheckpointManager(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_atomic_file_write(self):
        """Verify atomic writes replace target file cleanly without leaving partial tmp files."""
        fpath = self.temp_dir / "atomic_test.txt"
        write_atomic(fpath, "Hello atomic world!")
        self.assertTrue(fpath.exists())
        self.assertEqual(fpath.read_text(encoding="utf-8"), "Hello atomic world!")
        # Verify no .tmp files left over
        tmp_files = list(self.temp_dir.glob("*.tmp*"))
        self.assertEqual(len(tmp_files), 0)

    def test_run_manifest_initialization_and_sanitization(self):
        """Verify run manifest captures effective config without storing API keys."""
        cfg = {
            "name": "manifest_test",
            "endpoints": {
                "secret_ep": {"provider": "openrouter", "model": "test", "api_key": "SUPER_SECRET_KEY"}
            },
        }
        roster = [{"name": "ALPHA", "model": "gemma", "api_key": "LEAKED_KEY"}]
        manifest = self.mgr.init_manifest(
            run_id="run_101",
            preset_name="test_preset",
            topic="Test Topic",
            effective_config=cfg,
            agent_roster=roster,
            token_budgets={"max_total_tokens": 50000},
            max_turns=10,
            seed=42,
        )
        self.assertTrue(self.mgr.manifest_file.exists())
        loaded = self.mgr.load_manifest()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "run_101")
        # Verify secrets were stripped
        self.assertNotIn("api_key", loaded.effective_config["endpoints"]["secret_ep"])
        self.assertNotIn("api_key", loaded.agent_roster[0])

    def test_interrupted_run_resumption(self):
        """Verify resuming an interrupted run loads previously committed state and continues without repeating turns."""
        run_dir = self.temp_dir / "interrupted_run"
        cfg = {
            "name": "resume_exp",
            "topic": "Debate gravity",
            "scheduler": "round_robin",
            "limits": {"max_turns": 4},
            "agents": [
                {"name": "ALPHA", "endpoint": "local"},
                {"name": "BETA", "endpoint": "local"},
            ],
        }

        # Step 1: Run 2 turns using MockBackend
        backend1 = MockBackend({
            "ALPHA": ["Alpha turn 1 speech"],
            "BETA": ["Beta turn 2 speech"],
        })
        ch1 = Chamber(cfg, run_dir=run_dir, backend=backend1, quiet=True, copy_dir="")
        # Artificially limit to 2 turns to simulate interruption
        ch1.limits["max_turns"] = 2
        ch1.run()
        self.assertEqual(len(ch1.public_messages), 2)
        self.assertTrue((run_dir / "checkpoint.json").exists())

        # Step 2: Resume run for remaining 2 turns with new backend
        backend2 = MockBackend({
            "ALPHA": ["Alpha turn 3 speech"],
            "BETA": ["Beta turn 4 speech"],
        })
        ch2 = Chamber(cfg, run_dir=run_dir, backend=backend2, quiet=True, copy_dir="")
        # Load saved checkpoint
        chk_data = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        ch2.resume_from_checkpoint(chk_data)
        ch2.run()

        # Total public messages should now be 4: turns 1, 2, 3, 4
        self.assertEqual(len(ch2.public_messages), 4)
        self.assertEqual(ch2.public_messages[0].content, "Alpha turn 1 speech")
        self.assertEqual(ch2.public_messages[1].content, "Beta turn 2 speech")
        self.assertEqual(ch2.public_messages[2].content, "Alpha turn 3 speech")
        self.assertEqual(ch2.public_messages[3].content, "Beta turn 4 speech")

    def test_deterministic_event_log_replay(self):
        """Verify previously captured events can be deterministically replayed from events.jsonl."""
        run_dir = self.temp_dir / "replay_run"
        cfg = {
            "name": "replay_exp",
            "topic": "Event replay test",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2},
            "agents": [
                {"name": "ALPHA", "endpoint": "local"},
                {"name": "BETA", "endpoint": "local"},
            ],
        }
        backend = MockBackend({
            "ALPHA": [json.dumps({"message": "Hello from Alpha", "actions": [{"type": "CLAIM", "kind": "hypothesis", "content": "H1"}]})],
            "BETA": ["Hello from Beta"],
        })
        ch = Chamber(cfg, run_dir=run_dir, backend=backend, quiet=True, copy_dir="")
        ch.run()

        events_file = run_dir / "events.jsonl"
        self.assertTrue(events_file.exists())

        replayed = CheckpointManager.replay_events(events_file)
        self.assertGreater(len(replayed), 2)
        # Check that CLAIM action event was captured in order
        claim_events = [e for e in replayed if e.get("type") == "CLAIM"]
        self.assertEqual(len(claim_events), 1)
        self.assertEqual(claim_events[0]["actor"], "ALPHA")
        self.assertEqual(claim_events[0]["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
