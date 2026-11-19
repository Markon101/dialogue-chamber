import copy
import json
import tempfile
import unittest
from pathlib import Path

from autonomy import validate_policy
from chamber import Chamber
from inference import MockBackend
from protocol import parse_reply


class AgentAutonomyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config = json.loads(Path(__file__).with_name("examples").joinpath("self_directed_workshop.json").read_text())
        self.config["limits"]["max_turns"] = 6
        self.config["agents"][0]["autonomy"]["history_messages"].append(0)

    def chamber(self, name, backend, checkpoint=None):
        return Chamber(self.config, run_dir=self.root / name, backend=backend, quiet=True,
                       copy_dir="", checkpoint_data=checkpoint)

    def test_changes_reach_own_next_prompt_and_payload_and_resume(self):
        backend = MockBackend({
            "WITNESS": ['first-marker\n@CONFIGURE_SELF {"history_messages":0,"focus":"Find the missing diary","reply_tokens":96}',
                        '{"message":"second visit","actions":[{"type":"CONFIGURE_SELF","settings":{"history_messages":4}}]}'],
            "EDITOR": ['editor-marker\n@CONFIGURE_SELF {"history_messages":1,"reply_tokens":64}'],
            "READER": ['reader-marker\n@CONFIGURE_SELF {"focus":"unauthorized"}'],
        }, default="Continue")
        ch = self.chamber("first", backend).run()
        before, editor, reader, after = [payload for _, payload in backend.calls[:4]]
        self.assertEqual(before["max_tokens"], 256)
        self.assertEqual(after["max_tokens"], 96)
        prompt = after["messages"][-1]["content"]
        self.assertIn("Find the missing diary", prompt)
        self.assertIn("3 earlier public message(s) omitted by your history setting", prompt)
        self.assertNotIn("reader-marker", prompt)
        self.assertIn("first-marker", editor["messages"][-1]["content"])
        self.assertNotIn("Find the missing diary", reader["messages"][-1]["content"])
        self.assertEqual(ch.agents["EDITOR"].self_settings, {})
        self.assertEqual(ch.agents["READER"].self_settings, {})
        changes = [(e.actor, e.status) for e in ch.events if e.type == "CONFIGURE_SELF"]
        self.assertEqual(changes, [("WITNESS", "accepted"), ("EDITOR", "rejected"),
                                   ("READER", "rejected"), ("WITNESS", "accepted")])
        checkpoint = json.loads(ch.checkpoints.checkpoint_file.read_text())
        restored = self.chamber("next", MockBackend(), checkpoint)
        self.assertEqual(restored.agents["WITNESS"].self_settings, ch.agents["WITNESS"].self_settings)
        messages, budget = restored.build_context(restored.agents["WITNESS"])
        self.assertEqual(budget.reserved_output, 96)
        self.assertIn("reader-marker", messages[-1]["content"])

    def test_invalid_values_and_cross_agent_target_are_atomic(self):
        ch = self.chamber("limits", MockBackend())
        witness = ch.agents["WITNESS"]
        parsed = parse_reply('@CONFIGURE_SELF {"focus":"x","actions":[{"type":"FINISH"}]}')
        self.assertEqual([a["type"] for a in parsed.actions], ["CONFIGURE_SELF"])
        ch._apply_action(witness, parsed.actions[0])
        self.assertEqual(ch.events[-1].status, "rejected")
        for extra in ({"settings": {"focus": "x" * 181}}, {"settings": {"reply_tokens": True}},
                      {"settings": {"history_messages": -1}}, {"settings": {"reply_tokens": 257}},
                      {"settings": {"endpoint": "other"}}, {"settings": {"focus": "valid"}, "target": "READER"}):
            ch._apply_action(witness, {"type": "CONFIGURE_SELF", **extra})
            self.assertEqual(ch.events[-1].status, "rejected")
            self.assertEqual(witness.self_settings, {})

    def test_policy_validation_and_legacy_checkpoint(self):
        for policy in (None, {"reply_tokens": [257]}, {"focus_chars": True}, {"history_messages": []},
                       {"history_messages": [1, 1]}, {"model": ["other"]}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                validate_policy(policy, 256)
        for agent in self.config["agents"]:
            agent.pop("autonomy", None)
        self.config["limits"]["max_turns"] = 1
        old = self.chamber("legacy", MockBackend(default="hello")).run()
        checkpoint = json.loads(old.checkpoints.checkpoint_file.read_text())
        for agent in checkpoint["state"]["agents"].values():
            agent.pop("self_settings")
        restored = self.chamber("restored", MockBackend(), checkpoint)
        self.assertEqual(restored.agents["WITNESS"].self_settings, {})
        forged = copy.deepcopy(checkpoint)
        forged["state"]["agents"]["WITNESS"]["self_settings"] = {"reply_tokens": 123}
        with self.assertRaisesRegex(ValueError, "no permission"):
            self.chamber("invalid", MockBackend(), forged)
