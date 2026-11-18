#!/usr/bin/env python3
"""Tests for per-agent sampling parameters (config + dynamic SET_SAMPLING) and agent-decided run ending."""
import json
import tempfile
import unittest
from pathlib import Path

from chamber import Chamber, main
from inference import MockBackend, build_payload
from protocol import parse_reply


class SamplingAndEndModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_build_payload_forwards_llamacpp_samplers_and_filters_openrouter(self):
        agent_cfg = {
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "xtc_probability": 0.4,
            "xtc_threshold": 0.12,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.1,
            "dry_multiplier": 0.8,
            "typical_p": 0.95,
            "dynatemp_range": 0.25,
        }
        llama_ep = {"provider": "llamacpp", "model": "gemma"}
        p_llama, _ = build_payload(llama_ep, agent_cfg, [{"role": "user", "content": "hi"}], 0.8, 256)
        for k, v in agent_cfg.items():
            self.assertEqual(p_llama[k], v)

        or_ep = {"provider": "openrouter", "model": "google/gemma:free"}
        p_or, _ = build_payload(or_ep, agent_cfg, [{"role": "user", "content": "hi"}], 0.8, 256)
        self.assertEqual(p_or["top_p"], 0.9)
        self.assertEqual(p_or["min_p"], 0.05)
        self.assertNotIn("xtc_probability", p_or)
        self.assertNotIn("xtc_threshold", p_or)
        self.assertNotIn("dry_multiplier", p_or)

    def test_protocol_parses_sampling_lines_commands_and_json(self):
        p1 = parse_reply(
            "Adjusting my cadence.\nMIN_P: 0.06\nTOP_K: 32\nXTC_PROB: 0.5\nXTC_THRESH: 0.15\n@END_RUN: consensus reached",
            me="A",
            names=("A", "B"),
        )
        self.assertEqual(p1.format, "commands")
        sampling_acts = [a for a in p1.actions if a["type"] == "SET_SAMPLING"]
        self.assertEqual(len(sampling_acts), 4)
        merged = {}
        for a in sampling_acts:
            merged.update(a["params"])
        self.assertEqual(merged, {"min_p": 0.06, "top_k": 32, "xtc_probability": 0.5, "xtc_threshold": 0.15})
        end_acts = [a for a in p1.actions if a["type"] == "END_RUN"]
        self.assertEqual(len(end_acts), 1)
        self.assertEqual(end_acts[0]["content"], "consensus reached")

        p2 = parse_reply(
            "@SET_SAMPLING min_p=0.04 top_k=50 dry_mult=0.7\nShort line.",
            me="A",
            names=("A", "B"),
        )
        self.assertEqual(p2.actions[0]["type"], "SET_SAMPLING")
        self.assertEqual(p2.actions[0]["params"], {"min_p": 0.04, "top_k": 50, "dry_multiplier": 0.7})

        p3 = parse_reply(
            json.dumps({
                "message": "Structured shift.",
                "sampling_request": {"min_p": 0.08, "xtc_probability": 0.25},
                "actions": [{"type": "END_RUN", "content": "done"}],
            }),
            me="A",
            names=("A", "B"),
        )
        self.assertEqual(p3.format, "json")
        self.assertTrue(any(a["type"] == "SET_SAMPLING" and a["params"]["min_p"] == 0.08 for a in p3.actions))
        self.assertTrue(any(a["type"] == "END_RUN" for a in p3.actions))

    def test_chamber_updates_sampling_per_agent_and_sends_next_request(self):
        cfg = {
            "name": "sampling_test",
            "topic": "Test sampling",
            "scheduler": "round_robin",
            "limits": {"max_turns": 3},
            "agents": [
                {"name": "A", "endpoint": "local", "min_p": 0.05, "xtc_probability": 0.2, "xtc_threshold": 0.1},
                {"name": "B", "endpoint": "local", "top_k": 20},
            ],
        }
        backend = MockBackend({
            "A": [
                "First A.\nMIN_P: 0.12\nXTC_PROB: 0.6\nTOP_K: 45",
                "Second A.",
            ],
            "B": ["First B."],
        })
        ch = Chamber(cfg, run_dir=self.root / "sampling_run", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(len(backend.calls), 3)
        # Call 0: A initial config
        self.assertEqual(backend.calls[0][1]["min_p"], 0.05)
        self.assertEqual(backend.calls[0][1]["xtc_probability"], 0.2)
        # Call 1: B initial config (independent of A)
        self.assertEqual(backend.calls[1][1]["top_k"], 20)
        self.assertNotIn("min_p", backend.calls[1][1])
        # Call 2: A updated dynamically
        self.assertEqual(backend.calls[2][1]["min_p"], 0.12)
        self.assertEqual(backend.calls[2][1]["xtc_probability"], 0.6)
        self.assertEqual(backend.calls[2][1]["top_k"], 45)
        self.assertEqual(len(ch.sampling_changes), 3)

    def test_agent_decided_end_majority_and_withdraw(self):
        cfg = {
            "name": "agent_end_test",
            "topic": "Agree when done",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "limits": {"max_turns": 10},
            "agents": [
                {"name": "A", "endpoint": "local"},
                {"name": "B", "endpoint": "local"},
                {"name": "C", "endpoint": "local"},
            ],
        }
        backend = MockBackend({
            "A": ["Let's wrap?\n@END_RUN: ready", "Actually keep going.\n@END_RUN: withdraw"],
            "B": ["Not yet."],
            "C": ["Wait, let's finish.\n@END_RUN: done"],
        })
        # Turn 1: A votes end (1/3) -> continues
        # Turn 2: B says not yet (1/3) -> continues
        # Turn 3: C votes end (2/3 = majority!) -> stops on turn 3!
        ch = Chamber(cfg, run_dir=self.root / "end_maj", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(len(ch.public_messages), 3)
        self.assertIn("agents decided to end run (majority:", ch.stop_reason)

    def test_agent_decided_end_unanimous_and_fixed_turns_rejection(self):
        # In fixed_turns mode, END_RUN is rejected
        fixed_cfg = {
            "name": "fixed_test",
            "topic": "Fixed",
            "scheduler": "round_robin",
            "end_mode": "fixed_turns",
            "limits": {"max_turns": 2},
            "agents": [{"name": "A", "endpoint": "local"}, {"name": "B", "endpoint": "local"}],
        }
        ch_fixed = Chamber(
            fixed_cfg,
            run_dir=self.root / "fixed",
            backend=MockBackend({"A": ["Stop?\n@END_RUN: now"], "B": ["Still fixed."]}),
            quiet=True,
            copy_dir="",
        ).run()
        self.assertEqual(len(ch_fixed.public_messages), 2)
        self.assertEqual(ch_fixed.stop_reason, "reached max_turns (2)")
        self.assertTrue(any(e.type == "END_RUN" and e.status == "rejected" for e in ch_fixed.events))

        # CLI flags --agent-end --end-policy any --agent-sampling
        rc = main([
            "--preset", "two_agent_baseline",
            "--agent-end", "--end-policy", "any",
            "--agent-sampling", "LOCAL=min_p=0.05,xtc_probability=0.3",
            "--mock", "--quiet",
            "--run-dir", str(self.root / "cli_run"),
        ])
        self.assertEqual(rc, 0)
        manifest = json.loads((self.root / "cli_run" / "manifest.json").read_text())
        self.assertEqual(manifest["max_turns"], 40)


if __name__ == "__main__":
    unittest.main()
