import json
import tempfile
import unittest
from pathlib import Path

from chamber import Chamber
from inference import MockBackend
from protocol import parse_reply


class ArtifactContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "name": "artifact_context", "topic": "Track the changing record",
            "agents": [{"name": "A", "endpoint": "local", "capabilities": "full"},
                       {"name": "B", "endpoint": "local", "capabilities": "full"}],
            "artifacts": {"seed": {"world": "# World\n"}, "visible": ["world"]},
            "limits": {"max_turns": 2},
        }

    def chamber(self, name="run", checkpoint=None):
        return Chamber(self.config, run_dir=self.root / name, backend=MockBackend(default="Hello"),
                       quiet=True, copy_dir="", checkpoint_data=checkpoint)

    def context(self, chamber, name):
        return "\n".join(message["content"] for message in chamber.build_context(chamber.agents[name])[0])

    def test_long_visible_ledger_keeps_intro_and_latest_change(self):
        chamber = self.chamber()
        chamber.artifacts.write("A", "world", "append", "x" * 3000 + "MIDDLE_MARKER " + "m" * 9000)
        chamber.artifacts.write("A", "world", "append", "LATEST_MARKER changes the bridge")
        prompt = self.context(chamber, "B")
        self.assertIn("# World", prompt)
        self.assertIn("LATEST_MARKER changes the bridge", prompt)
        self.assertNotIn("MIDDLE_MARKER", prompt)
        self.assertIn("Middle omitted", prompt)
        self.assertIn("@READ world OFFSET", prompt)

    def test_explicit_offset_recovers_middle_and_survives_checkpoint(self):
        chamber = self.chamber()
        chamber.artifacts.write("A", "world", "append", "p" * 1600 + "MIDDLE_MARKER" + "q" * 1600)
        text = chamber.artifacts.read("A", "world")[2]
        offset = text.index("MIDDLE_MARKER")
        parsed = parse_reply(f"@READ world {offset}\nShow me the middle.", me="B", names=("A", "B"))
        self.assertEqual(parsed.actions[0]["offset"], offset)
        chamber._dispatch_actions(chamber.agents["B"], parsed)
        self.assertEqual(chamber.events[-1].status, "accepted")
        prompt = self.context(chamber, "B")
        self.assertIn("MIDDLE_MARKER", prompt)
        self.assertIn(f"chars {offset}-", prompt)
        self.assertNotIn("# World\n", prompt)
        chamber._save_turn_checkpoint(0)
        saved = json.loads(chamber.checkpoints.checkpoint_file.read_text())
        restored = self.chamber("restored", saved)
        self.assertIn("MIDDLE_MARKER", self.context(restored, "B"))
        self.assertEqual(restored.agents["B"].pending_reads, saved["state"]["agents"]["B"]["pending_reads"])

    def test_short_artifact_remains_complete_and_default_read_starts_at_zero(self):
        chamber = self.chamber()
        self.assertIn("# World", self.context(chamber, "B"))
        parsed = parse_reply("@READ world\nCheck the beginning.", me="B", names=("A", "B"))
        self.assertEqual(parsed.actions[0]["offset"], 0)
        chamber._dispatch_actions(chamber.agents["B"], parsed)
        self.assertEqual(chamber.events[-1].status, "accepted")
        self.assertIn("# World", self.context(chamber, "B"))

    def test_invalid_offsets_do_not_grant_read_marker_or_queue_content(self):
        chamber = self.chamber()
        chamber.artifacts.write("A", "world", "append", "new version")
        self.assertEqual(chamber.artifacts.seen[("B", "world.md")], 1)
        for offset in (-1, True, 99999):
            chamber._apply_action(chamber.agents["B"],
                                  {"type": "ARTIFACT_READ", "name": "world", "offset": offset})
            self.assertEqual(chamber.events[-1].status, "rejected")
        self.assertEqual(chamber.artifacts.seen[("B", "world.md")], 1)
        self.assertEqual(chamber.agents["B"].pending_reads, [])
        parsed = parse_reply('{"message":"", "actions":[{"type":"ARTIFACT_READ","name":"world","offset":-1}]}',
                             me="B", names=("A", "B"))
        self.assertFalse(parsed.actions)
        self.assertTrue(any("offset" in issue for issue in parsed.issues))


if __name__ == "__main__":
    unittest.main()
