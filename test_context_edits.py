import json
import tempfile
import unittest
from pathlib import Path

from chamber import Chamber
from inference import MockBackend
from protocol import parse_reply


class ContextEditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = {
            "name": "private_context",
            "scheduler": "round_robin",
            "limits": {"max_turns": 4},
            "communication": {
                "public": {"A": ["B"], "B": ["A", "C"], "C": ["B"]},
                "channels": {"workshop": {"readers": ["A", "B"], "writers": ["A"]}},
            },
            "channels": {"workshop": {}},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full", "subscriptions": ["workshop"]},
                {"name": "B", "endpoint": "local", "capabilities": "full", "subscriptions": ["workshop"],
                 "autonomy": {"context_edit": {"public": ["hide", "replace", "summarize"],
                                               "whispers": ["hide"], "#workshop": ["summarize"]}}},
                {"name": "C", "endpoint": "local",
                 "autonomy": {"context_edit": {"public": ["replace"], "whispers": ["hide"],
                                               "#workshop": ["summarize"]}}},
            ],
        }

    def chamber(self, name, backend, checkpoint=None):
        return Chamber(self.cfg, run_dir=self.root / name, backend=backend, quiet=True,
                       copy_dir="", checkpoint_data=checkpoint)

    def test_private_views_filter_prior_messages_and_survive_continuation(self):
        backend = MockBackend({
            "A": ["original public marker\n@WHISPER B: original whisper marker\n@PUBLISH workshop: original channel marker",
                  "new public marker\n@PUBLISH workshop: new channel marker"],
            "B": ['@EDIT_CONTEXT {"source":"public","op":"replace","target":"m1","text":"personal replacement"}\n'
                  '@EDIT_CONTEXT {"source":"whispers","op":"hide","target":"dm1"}\n'
                  '@EDIT_CONTEXT {"source":"#workshop","op":"summarize","text":"personal channel summary"}\n'
                  'B speaks.'],
            "C": ['@EDIT_CONTEXT {"source":"public","op":"replace","target":"m1","text":"forbidden"}\n'
                  '@EDIT_CONTEXT {"source":"whispers","op":"hide","target":"dm1"}\n'
                  '@EDIT_CONTEXT {"source":"#workshop","op":"summarize","text":"forbidden"}\n'
                  'C speaks.'],
        })
        first = self.chamber("first", backend).run()
        accepted = [e for e in first.events if e.type == "EDIT_CONTEXT" and e.status == "accepted"]
        rejected = [e for e in first.events if e.type == "EDIT_CONTEXT" and e.status == "rejected"]
        self.assertEqual([e.actor for e in accepted], ["B"] * 3)
        self.assertEqual([e.actor for e in rejected], ["C"] * 3)
        a_prompt = backend.calls[3][1]["messages"][-1]["content"]
        c_prompt = backend.calls[2][1]["messages"][-1]["content"]
        self.assertIn("original public marker", a_prompt)
        self.assertIn("original channel marker", a_prompt)
        self.assertNotIn("personal replacement", a_prompt)
        self.assertNotIn("original public marker", c_prompt)
        self.assertNotIn("original whisper marker", c_prompt)
        self.assertNotIn("original channel marker", c_prompt)

        checkpoint = json.loads(first.checkpoints.checkpoint_file.read_text())
        self.cfg["limits"]["max_turns"] = 5
        resumed_backend = MockBackend({
            "B": ['{"message":"I will consolidate.","actions":[{"type":"EDIT_CONTEXT","edit":'
                  '{"source":"public","op":"summarize","text":"personal public summary"}}]}'],
        })
        resumed = self.chamber("continued", resumed_backend, checkpoint).run()
        prompt = resumed_backend.calls[0][1]["messages"][-1]["content"]
        for text in ("[your replacement for m1] personal replacement", "new public marker",
                     "personal channel summary", "new channel marker"):
            self.assertIn(text, prompt)
        for text in ("original public marker", "original whisper marker", "original channel marker"):
            self.assertNotIn(text, prompt)
        self.assertEqual(resumed.public_messages[0].content, "original public marker")
        self.assertEqual(resumed.dms[0].content, "original whisper marker")

        b = resumed.agents["B"]
        messages, accounting = resumed.build_context(b)
        prompt = messages[-1]["content"]
        self.assertIn("Your summary of public through turn 4", prompt)
        self.assertIn("I will consolidate.", prompt)
        self.assertNotIn("new public marker", prompt)
        self.assertEqual(accounting.omitted_messages_count, 4)
        restored = parse_reply('@EDIT_CONTEXT {"source":"public","op":"restore"}').actions[0]
        resumed._apply_action(b, restored)
        prompt = resumed.build_context(b)[0][-1]["content"]
        self.assertIn("original public marker", prompt)
        self.assertNotIn("personal public summary", prompt)
        b.subscriptions.remove("workshop")
        prompt = resumed.build_context(b)[0][-1]["content"]
        self.assertNotIn("personal channel summary", prompt)
        self.assertNotIn("new channel marker", prompt)

    def test_edit_commands_remain_separate_from_structured_turn_json(self):
        raw = '@EDIT_CONTEXT {"source":"public","op":"hide","target":"m1","actions":[{"type":"FINISH"}]}'
        actions = parse_reply(raw).actions
        self.assertEqual([action["type"] for action in actions], ["EDIT_CONTEXT"])
        ch = self.chamber("invalid", MockBackend()).run()
        b = ch.agents["B"]
        ch._apply_action(b, actions[0])
        self.assertEqual(ch.events[-1].status, "rejected")
        self.assertTrue(b.active)
        self.assertEqual(b.context_edits, {})
