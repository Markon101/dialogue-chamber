"""Behavioral checks for opt-in society membership and information flow."""
import json
import tempfile
import unittest
from pathlib import Path

from chamber import Chamber
from inference import MockBackend
from protocol import parse_reply


class SocietyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = {
            "name": "society_test", "topic": "Build a world", "scheduler": "round_robin",
            "society": {"authorities": ["A"], "max_groups": 2},
            "agents": [{"name": n, "endpoint": "local", "capabilities": "full", "max_tokens": 256}
                       for n in ("A", "B", "C")],
            "limits": {"max_turns": 6},
        }

    def chamber(self, name="run", backend=None):
        return Chamber(self.cfg, run_dir=Path(self.temp.name) / name,
                       backend=backend or MockBackend({}), quiet=True, copy_dir="")

    def action(self, ch, name, kind, group, recipient=None, content=None):
        ch._apply_action(ch.agents[name], {"type": kind, "channel": group,
                                          "recipient": recipient, "content": content})

    def test_invitation_privacy_and_authority(self):
        ch = self.chamber()
        self.action(ch, "A", "CREATE_GROUP", "map")
        self.action(ch, "A", "PUBLISH", "map", content="private evidence")
        self.action(ch, "B", "JOIN_GROUP", "map")
        self.assertEqual(ch.events[-1].status, "rejected")
        self.action(ch, "B", "PUBLISH", "map", content="intrusion")
        self.assertEqual(ch.events[-1].status, "rejected")
        self.action(ch, "B", "SUBSCRIBE", "map")
        self.assertEqual(ch.events[-1].status, "rejected")
        self.assertNotIn("private evidence", str(ch.build_context(ch.agents["B"])[0]))
        self.action(ch, "A", "INVITE_GROUP", "map", "B")
        self.assertNotIn("B", ch.society.groups["map"]["members"])
        self.action(ch, "B", "JOIN_GROUP", "map")
        self.assertIn("private evidence", str(ch.build_context(ch.agents["B"])[0]))
        self.action(ch, "B", "COMPEL_GROUP", "map", "C")
        self.assertEqual(ch.events[-1].status, "rejected")
        self.action(ch, "A", "COMPEL_GROUP", "map", "C")
        self.assertIn("private evidence", str(ch.build_context(ch.agents["C"])[0]))
        self.action(ch, "C", "LEAVE_GROUP", "map")
        self.assertEqual(ch.events[-1].status, "rejected")
        self.action(ch, "A", "RELEASE_GROUP", "map", "C")
        self.action(ch, "C", "LEAVE_GROUP", "map")
        self.assertNotIn("private evidence", str(ch.build_context(ch.agents["C"])[0]))

    def test_model_commands_end_to_end_and_checkpoint(self):
        backend = MockBackend({
            "A": ["A group might help.\n@CREATE_GROUP map\n@INVITE_GROUP map B\n@COMPEL_GROUP map C\n@PUBLISH map: stone moves", "The map matters."],
            "B": ["I accept.\n@JOIN_GROUP map", "I can see the evidence."],
            "C": ["I am required to attend.", "I disagree."],
        })
        ch = self.chamber(backend=backend).run()
        self.assertEqual(ch.society.groups["map"]["members"], ["A", "C", "B"])
        self.assertIn("stone moves", str(backend.calls[2][1]["messages"]))
        checkpoint = json.loads((ch.run_dir / "checkpoint.json").read_text())
        restored = self.chamber("restored")
        restored.resume_from_checkpoint(checkpoint)
        self.assertEqual(restored.society.groups, ch.society.groups)
        self.assertIn("stone moves", str(restored.build_context(restored.agents["B"])[0]))
        self.action(restored, "C", "LEAVE_GROUP", "map")
        self.assertEqual(restored.events[-1].status, "rejected")

    def test_invalid_names_and_json_actions(self):
        ch = self.chamber()
        self.action(ch, "A", "CREATE_GROUP", "../map")
        self.assertEqual(ch.events[-1].status, "rejected")
        parsed = parse_reply('{"message":"Hello", "actions":[{"type":"CREATE_GROUP","channel":"map"}]}',
                             "", me="A", names=("A", "B", "C"), max_chars=1000)
        ch._dispatch_actions(ch.agents["A"], parsed)
        self.assertIn("map", ch.society.groups)

    def test_world_command_does_not_swallow_public_speech(self):
        parsed = parse_reply("@WORLD: The river moved.\nThe council should investigate.",
                             "", me="A", names=("A", "B", "C"), max_chars=1000)
        self.assertEqual(parsed.message, "The council should investigate.")
        self.assertEqual(parsed.actions[0]["content"], "The river moved.")

    def test_action_only_turn_counts_as_completed(self):
        self.cfg["limits"]["max_turns"] = 1
        backend = MockBackend({"A": ["@CREATE_GROUP map"]})
        ch = self.chamber(backend=backend).run()
        self.assertEqual(len(ch.public_messages), 0)
        self.assertIn("map", ch.society.groups)
        self.assertEqual(ch.agents["A"].turns_taken, 1)
        self.assertEqual(ch.compute_metrics()["turns_completed"], 1)
        self.assertIn("**Turns**: 1/1", (ch.run_dir / "transcript.md").read_text())


if __name__ == "__main__":
    unittest.main()
