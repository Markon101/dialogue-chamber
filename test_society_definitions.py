import copy
import json
import tempfile
import unittest
from pathlib import Path

from chamber import Chamber
from inference import MockBackend
from protocol import parse_reply


class SocietyDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "name": "defined", "society": {"groups": {"guild": {"members": ["A", "B"], "frozen": True}}},
            "channels": {"bulletin": {}},
            "communication": {"public": {"A": ["B"]}, "whispers": {"B": ["A"]},
                              "channels": {"bulletin": {"readers": ["A", "B"], "writers": ["A"]}}},
            "agents": [{"name": name, "endpoint": "local", "capabilities": "full"} for name in ("A", "B", "C")],
            "limits": {"max_turns": 1},
        }

    def chamber(self, config=None, name="run", checkpoint=None):
        return Chamber(config or self.config, run_dir=self.root / name, backend=MockBackend(default="Hello"),
                       quiet=True, copy_dir="", checkpoint_data=checkpoint)

    def action(self, chamber, actor, kind, **fields):
        chamber._apply_action(chamber.agents[actor], {"type": kind, **fields})
        return chamber.events[-1].status

    def test_seeded_groups_freeze_membership_but_allow_posts(self):
        chamber = self.chamber()
        self.assertIn("guild", chamber.agents["B"].subscriptions)
        self.assertEqual(self.action(chamber, "A", "LEAVE_GROUP", channel="guild"), "rejected")
        self.assertEqual(self.action(chamber, "A", "INVITE_GROUP", channel="guild", recipient="C"), "rejected")
        self.assertEqual(self.action(chamber, "A", "PUBLISH", channel="guild", content="private-history"), "accepted")
        self.assertIn("private-history", str(chamber.build_context(chamber.agents["B"])[0]))
        self.assertNotIn("private-history", str(chamber.build_context(chamber.agents["C"])[0]))

    def test_global_freeze_and_open_admission(self):
        config = copy.deepcopy(self.config)
        config["society"]["frozen"] = True
        chamber = self.chamber(config)
        self.assertEqual(self.action(chamber, "C", "CREATE_GROUP", channel="other"), "rejected")
        config["society"]["frozen"] = False
        config["society"]["groups"]["guild"] = {"members": ["A"], "admission": "open"}
        chamber = self.chamber(config, "open")
        self.assertEqual(self.action(chamber, "C", "JOIN_GROUP", channel="guild"), "accepted")
        self.assertEqual(self.action(chamber, "C", "LEAVE_GROUP", channel="guild"), "accepted")

    def test_directed_speech_whispers_and_channel_permissions(self):
        chamber = self.chamber()
        chamber._dispatch_actions(chamber.agents["A"], parse_reply("B C unique-speech", me="A", names=("A", "B", "C")))
        self.assertIn("unique-speech", str(chamber.build_context(chamber.agents["B"])[0]))
        self.assertNotIn("unique-speech", str(chamber.build_context(chamber.agents["C"])[0]))
        self.assertNotIn("C", chamber.reactive_queue)
        self.assertEqual(self.action(chamber, "A", "WHISPER", recipient="B", content="secret"), "rejected")
        self.assertEqual(self.action(chamber, "B", "WHISPER", recipient="A", content="secret"), "accepted")
        self.assertEqual(self.action(chamber, "C", "SUBSCRIBE", channel="bulletin"), "rejected")
        self.assertEqual(self.action(chamber, "B", "PUBLISH", channel="bulletin", content="forged"), "rejected")
        self.assertEqual(self.action(chamber, "B", "SUBSCRIBE", channel="bulletin"), "accepted")
        self.assertEqual(self.action(chamber, "A", "PUBLISH", channel="bulletin", content="notice"), "accepted")

    def test_definition_and_permissions_survive_continuation(self):
        chamber = self.chamber().run()
        saved = json.loads(chamber.checkpoints.checkpoint_file.read_text())
        restored = self.chamber(saved["state"]["config"], "restored", saved)
        self.assertEqual(restored.society.groups, chamber.society.groups)
        self.assertEqual(self.action(restored, "A", "LEAVE_GROUP", channel="guild"), "rejected")
        self.assertFalse(restored.communication.allows("public", "A", "C"))

    def test_bad_definitions_fail_before_generation(self):
        variants = [
            ("unknown member", {"members": ["MISSING"]}),
            ("duplicate member", {"members": ["A", "a"]}),
            ("compelled outsider", {"members": ["A"], "compelled": ["B"]}),
            ("typo", {"members": ["A"], "frozn": True}),
        ]
        for index, (label, definition) in enumerate(variants):
            config = copy.deepcopy(self.config)
            config["society"]["groups"]["guild"] = definition
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.chamber(config, f"bad_{index}")
        config = copy.deepcopy(self.config)
        config["communication"]["whispers"] = {"A": ["MISSING"]}
        with self.assertRaises(ValueError):
            self.chamber(config, "bad_route")

    def test_example_is_runnable_offline(self):
        config = json.loads((Path(__file__).parent / "examples" / "defined_society.json").read_text())
        config["limits"]["max_turns"] = 2
        chamber = self.chamber(config).run()
        self.assertEqual(chamber.turn, 2)
        self.assertTrue(chamber.society.frozen)
