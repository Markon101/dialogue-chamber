import copy
import hashlib
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from artifacts import ArtifactError
from chamber import Chamber, SAVED_AGENT_FIELDS, SAVED_RUNTIME_FIELDS, main
from inference import DemoBackend, InferenceError, MockBackend


class RunContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "name": "continuation_test", "topic": "Organize freely", "scheduler": "self_directed",
            "society": {"authorities": ["A"], "max_groups": 4},
            "artifacts": {"seed": {"world": "# World\n"}, "visible": ["world"]},
            "agents": [{"name": name, "endpoint": "local", "capabilities": "full", "max_tokens": 256}
                       for name in ("A", "B", "C")],
            "limits": {"max_turns": 6, "max_requests": 30},
        }
        self.addCleanup(patch.stopall)
        patch("chamber.COPY_DIR", "").start()

    def chamber(self, name, backend=None, config=None, checkpoint=None):
        return Chamber(config or self.config, run_dir=self.root / name, seed=19,
                       backend=backend or MockBackend(default="Thinking."), quiet=True,
                       copy_dir="", checkpoint_data=checkpoint)

    def saved(self, chamber):
        return json.loads(chamber.checkpoints.checkpoint_file.read_text())

    def fingerprints(self, directory):
        return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    def test_complete_social_state_artifacts_and_budgets_round_trip(self):
        original = self.chamber("source")
        for actor, action in (
            ("A", {"type": "CREATE_GROUP", "channel": "map"}),
            ("A", {"type": "COMPEL_GROUP", "channel": "map", "recipient": "B"}),
            ("A", {"type": "PUBLISH", "channel": "map", "content": "private coordinates"}),
            ("A", {"type": "WHISPER", "recipient": "B", "content": "private promise"}),
            ("A", {"type": "UPDATE_NOTE", "content": "remember the disagreement"}),
            ("A", {"type": "SET_STYLE", "content": "terse"}),
            ("A", {"type": "PROPOSE_TOPIC", "content": "A new purpose"}),
            ("A", {"type": "CLAIM", "content": "Keep the bridge open"}),
            ("C", {"type": "VOTE", "target": "c1", "value": "yes"}),
            ("A", {"type": "ARTIFACT_WRITE", "name": "world", "content": "A bridge appeared"}),
            ("B", {"type": "FINISH", "content": "I choose to stop"}),
        ):
            original._apply_action(original.agents[actor], action)
        original.next_speaker = "C"
        original.nominated_next = "C"
        original.requests_used = 7
        original.retries_used = 2
        original.remote_cost_usd = 0.125
        original.remote_calls_without_cost = 1
        original.agents["C"].last_turn = 4
        original.agents["C"].consecutive_turns = 1
        original._save_turn_checkpoint(0)
        before = self.fingerprints(original.run_dir)
        saved = self.saved(original)
        restored = self.chamber("continued", checkpoint=saved)

        self.assertEqual(restored.society.groups, original.society.groups)
        self.assertEqual(restored.artifacts.snapshot(), original.artifacts.snapshot())
        self.assertEqual(restored.rng.getstate(), original.rng.getstate())
        for attribute in SAVED_RUNTIME_FIELDS:
            self.assertEqual(json.dumps(getattr(restored, attribute)), json.dumps(getattr(original, attribute)))
        for name in original.agents:
            for attribute in SAVED_AGENT_FIELDS:
                self.assertEqual(getattr(restored.agents[name], attribute), getattr(original.agents[name], attribute))
        self.assertTrue(restored.remote_cost_blocked(0.25))
        self.assertFalse(restored.agents["B"].active)
        self.assertNotIn("private coordinates", str(restored.build_context(restored.agents["C"])[0]))
        self.assertIn("private coordinates", str(restored.build_context(restored.agents["A"])[0]))
        with self.assertRaises(ArtifactError):
            restored.artifacts.write("B", "world", "replace", "stale overwrite")
        restored._apply_action(restored.agents["C"], {"type": "UPDATE_NOTE", "content": "A new memory"})
        self.assertEqual(restored.events[-1].seq, original.events[-1].seq + 1)
        self.assertEqual(before, self.fingerprints(original.run_dir))

    def test_segmented_weighted_run_matches_uninterrupted_demo(self):
        config = copy.deepcopy(self.config)
        config["scheduler"] = "weighted_random"
        config["limits"]["max_turns"] = 12
        full = self.chamber("full", DemoBackend(seed=51), config).run()
        partial = self.chamber("partial", DemoBackend(seed=51), config)
        partial.limits["max_turns"] = 5
        partial.run()
        continued = self.chamber("continued", DemoBackend(seed=0), config, self.saved(partial)).run()

        self.assertEqual([(message.speaker, message.content) for message in continued.public_messages],
                         [(message.speaker, message.content) for message in full.public_messages])
        self.assertEqual(continued.requests_used, full.requests_used)
        self.assertEqual(continued.tracked_tokens, full.tracked_tokens)
        self.assertEqual(continued.artifacts.snapshot(), full.artifacts.snapshot())
        self.assertEqual([(event.actor, event.type, event.status, event.target) for event in continued.events],
                         [(event.actor, event.type, event.status, event.target) for event in full.events])

    def test_cli_uses_saved_config_and_preserves_original(self):
        original = self.chamber("source", MockBackend({
            "A": ["@CREATE_GROUP bridge\n@WORLD: A narrow bridge connects the islands.\nHello."],
        }, default="Still here."))
        original.limits["max_turns"] = 2
        original.run()
        before = self.fingerprints(original.run_dir)
        target = self.root / "cli_continued"
        self.assertEqual(main(["--resume", str(original.run_dir), "--turns", "5", "--requests", "40",
                               "--run-dir", str(target), "--quiet"]), 0)
        saved = json.loads((target / "checkpoint.json").read_text())
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(saved["turn"], 5)
        self.assertEqual(saved["state"]["runtime"]["requests_used"], 5)
        self.assertIn("bridge", saved["state"]["society"])
        self.assertIn("A narrow bridge", (target / "artifacts" / "world.md").read_text())
        self.assertEqual(manifest["effective_config"]["continuation"]["turn"], 2)
        events = [json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()]
        self.assertEqual([event["seq"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(before, self.fingerprints(original.run_dir))

    def test_ctrl_c_saves_a_continuable_interrupted_run(self):
        def interrupted_reply(payload):
            signal.raise_signal(signal.SIGINT)
            return "@NOTE: Remember this meeting\nA moment to think."

        original = self.chamber("interrupted", MockBackend({"A": [interrupted_reply]})).run()
        self.assertEqual(original.turn, 1)
        self.assertEqual(original.checkpoints.load_manifest().status, "interrupted")
        continued = self.chamber("after_interrupt", checkpoint=self.saved(original)).run()
        self.assertEqual(continued.public_messages[1].speaker, "B")
        self.assertEqual(continued.agents["A"].note, "Remember this meeting")
        self.assertEqual(continued.turn, 6)

    def test_failures_and_exhausted_budgets_are_preserved(self):
        config = copy.deepcopy(self.config)
        config["limits"]["max_requests"] = 1
        original = self.chamber("failure", MockBackend({"A": [InferenceError("offline", transient=False)]}), config).run()
        self.assertEqual(original.requests_used, 1)
        self.assertEqual(original.agents["A"].failures, 1)
        continued = self.chamber("still_exhausted", checkpoint=self.saved(original), config=config).run()
        self.assertEqual(continued.requests_used, 1)
        self.assertEqual(continued.public_messages, [])
        self.assertEqual(continued.agents["A"].failures, 1)

    def test_corrupt_artifact_fails_before_restoring_files(self):
        original = self.chamber("source").run()
        saved = self.saved(original)
        saved["state"]["artifacts"]["files"]["world.md"]["content"] = "tampered"
        with self.assertRaisesRegex(ArtifactError, "checksum mismatch"):
            self.chamber("corrupt", checkpoint=saved)
        self.assertFalse((self.root / "corrupt" / "artifacts" / "world.md").exists())

    def test_missing_legacy_and_in_place_resume_fail_without_overwriting(self):
        target = self.root / "should_not_exist"
        self.assertEqual(main(["--resume", str(self.root / "missing"), "--run-dir", str(target), "--mock"]), 1)
        legacy = self.root / "legacy.json"
        legacy.write_text(json.dumps({"turn": 1, "state": {}}))
        self.assertEqual(main(["--resume", str(legacy), "--run-dir", str(target), "--mock"]), 1)
        self.assertFalse(target.exists())
        original = self.chamber("source").run()
        before = self.fingerprints(original.run_dir)
        self.assertEqual(main(["--resume", str(original.run_dir), "--turns", "9",
                               "--run-dir", str(original.run_dir), "--mock"]), 1)
        self.assertEqual(before, self.fingerprints(original.run_dir))

    def test_nested_endpoint_credentials_are_not_saved(self):
        config = copy.deepcopy(self.config)
        config["agents"][0]["endpoint"] = {
            "name": "custom", "provider": "mock", "model": "mock", "api_key": "DO_NOT_SAVE_PRIMARY",
        }
        config["agents"][0]["fallback_models"] = [{"model": "mock", "Authorization": "DO_NOT_SAVE_FALLBACK"}]
        original = self.chamber("credentials", config=config).run()
        self.assertNotIn("DO_NOT_SAVE", original.checkpoints.checkpoint_file.read_text())
        self.assertNotIn("DO_NOT_SAVE", original.checkpoints.manifest_file.read_text())


if __name__ == "__main__":
    unittest.main()
