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

    def test_pipe_separated_and_compound_sampling_from_real_run(self):
        """Exact pipe-separated sampling formats emitted by Gemma 4 12B parse into one atomic SET_SAMPLING."""
        p = parse_reply(
            "Opening verse.\n"
            "TOP_P: 0.9 | TOP_K: 40 | MIN_P: 0.05 | XTC_PROB: 0.1 | XTC_THRESH: 0.3 | REPEAT_PENALTY: 1.5\n"
            "NEXT: BETA",
            me="ALPHA",
            names=("ALPHA", "BETA"),
        )
        self.assertEqual(p.issues, [])
        self.assertEqual(p.next_speaker, "BETA")
        self.assertEqual(len(p.actions), 1)
        self.assertEqual(p.actions[0]["type"], "SET_SAMPLING")
        self.assertEqual(
            p.actions[0]["params"],
            {
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.05,
                "xtc_probability": 0.1,
                "xtc_threshold": 0.3,
                "repeat_penalty": 1.5,
            },
        )

        # Compound starting with TEMP:
        p_temp = parse_reply(
            "Tuning.\nTEMP: 0.85 | TOP_K: 40 | MIN_P: 0.05",
            me="ALPHA",
            names=("ALPHA", "BETA"),
        )
        self.assertEqual(p_temp.issues, [])
        self.assertIsNone(p_temp.temperature)
        self.assertEqual(p_temp.actions[0]["params"], {"temperature": 0.85, "top_k": 40, "min_p": 0.05})

        # Colon-comma syntax inside @SET_SAMPLING
        p_colon = parse_reply(
            "@SET_SAMPLING min_p: 0.05, top_k: 40\nReply.",
            me="ALPHA",
            names=("ALPHA", "BETA"),
        )
        self.assertEqual(p_colon.issues, [])
        self.assertEqual(p_colon.actions[0]["params"], {"min_p": 0.05, "top_k": 40})

    def test_atomic_sampling_validation_rejects_bad_compounds_without_partial_mutation(self):
        # 1. Protocol-level rejection: out-of-range, unknown param, ambiguous token, malformed number
        bad_inputs = [
            "TOP_P: 1.8 | TOP_K: 40",
            "TOP_P: 0.9 | XTC_THRESH: 0.9",
            "TOP_P: 0.9 | TURBO: 5",
            "TOP_P: 0.9 | GARBAGE",
            "@SET_SAMPLING min_p=0.05 top_k=3.5",
            "@SET_SAMPLING min_p=abc top_k=40",
        ]
        for raw in bad_inputs:
            p = parse_reply(f"Body.\n{raw}", me="A", names=("A", "B"))
            self.assertEqual([a for a in p.actions if a["type"] == "SET_SAMPLING"], [], msg=raw)
            self.assertEqual(len(p.issues), 1, msg=f"expected single diagnostic for {raw}: {p.issues}")

        # 2. Orchestrator-level atomic rejection: temp out of agent's temp_min..temp_max or llamacpp-only on openrouter
        cfg = {
            "name": "atomic_sampling",
            "topic": "Atomic",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2, "temp_min": 0.2, "temp_max": 1.3},
            "agents": [
                {"name": "A", "endpoint": "local", "temperature": 0.8, "min_p": 0.05, "top_k": 40},
                {"name": "B", "endpoint": "hermes", "temperature": 0.8, "min_p": 0.05},
            ],
        }
        backend = MockBackend({
            # A sends valid min_p=0.12 alongside temperature=1.9 (> temp_max=1.3) -> whole action rejected, min_p stays 0.05!
            "A": ['{"message": "A", "actions": [{"type": "SET_SAMPLING", "params": {"min_p": 0.12, "temperature": 1.9}}]}'],
            # B on openrouter sends valid top_p=0.9 alongside xtc_probability=0.3 -> whole action rejected, top_p stays None!
            "B": ['{"message": "B", "actions": [{"type": "SET_SAMPLING", "params": {"top_p": 0.9, "xtc_probability": 0.3}}]}'],
        })
        ch = Chamber(cfg, run_dir=self.root / "atomic", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(ch.agents["A"].min_p, 0.05)
        self.assertEqual(ch.agents["A"].temperature, 0.8)
        self.assertIsNone(ch.agents["B"].top_p)
        self.assertEqual(len(ch.sampling_changes), 0)
        rej = [e for e in ch.events if e.type == "SET_SAMPLING" and e.status == "rejected"]
        self.assertEqual(len(rej), 2)

    def test_claim_id_case_normalization_and_next_none_handling(self):
        # Claim ID normalization in protocol & chamber; prose targets stay rejected
        p = parse_reply(
            "Checking.\n@CHALLENGE C1: uppercase id\n@VOTE #C1 yes\n@CHALLENGE <honkytonk's: bad\nNEXT: NONE",
            me="A",
            names=("A", "B"),
        )
        self.assertEqual(p.issues, [])
        self.assertIsNone(p.next_speaker)
        self.assertEqual(p.actions[0]["target"], "c1")
        self.assertEqual(p.actions[1]["target"], "c1")
        self.assertEqual(p.actions[2]["target"], "<honkytonk's")

        # Real unknown speaker still produces an issue
        p_bad_next = parse_reply("Hi.\nNEXT: GHOST", me="A", names=("A", "B"))
        self.assertTrue(any("unknown next speaker" in i for i in p_bad_next.issues))

        cfg = {
            "name": "claim_norm",
            "topic": "Claims",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2},
            "agents": [{"name": "A", "endpoint": "local"}, {"name": "B", "endpoint": "local"}],
        }
        backend = MockBackend({
            "A": ["First.\n@CLAIM: Sky is blue"],
            "B": ["Second.\n@CHALLENGE C1: Needs evidence\n@VOTE #C1 no\n@CHALLENGE <honkytonk's: missing\nNEXT: NONE"],
        })
        ch = Chamber(cfg, run_dir=self.root / "claim_norm", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(ch.board["c1"].status, "disputed")
        self.assertEqual(ch.board["c1"].votes["B"], "no")
        rejected = [e for e in ch.events if e.status == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].target, "<honkytonk's")

    def test_action_budget_dedup_and_end_run_priority_respects_safety_limits(self):
        # 1. Duplicate votes don't waste action slots; END_RUN after many VOTE lines gets prioritized inside max_actions_per_response=4
        cfg = {
            "name": "budget_priority",
            "topic": "Wrap up",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 2,
            "limits": {"max_turns": 10, "max_actions_per_response": 4},
            "agents": [
                {"name": "A", "endpoint": "local"},
                {"name": "B", "endpoint": "local"},
                {"name": "C", "endpoint": "local"},
            ],
        }
        backend = MockBackend({
            "A": ["Seed.\n@CLAIM: c1\n@CLAIM: c2\n@CLAIM: c3\n@END_RUN: too early"],
            "B": ["Vote.\n@VOTE c1 yes\n@VOTE c2 yes\n@END_RUN: ready"],
            # C votes on c1..c3 twice + new vote on c3 + END_RUN at the end (exceeds 4 raw actions,
            # but duplicate c1/c2 votes are deduped and END_RUN gets prioritized into the 4-action window!)
            "C": ["Lock.\n@VOTE c1 yes\n@VOTE c1 yes\n@VOTE c2 yes\n@VOTE c3 yes\n@CLAIM: extra\n@END_RUN: majority now"],
        })
        ch = Chamber(cfg, run_dir=self.root / "priority", backend=backend, quiet=True, copy_dir="").run()
        # Turn 1: A's END_RUN rejected (turn 1 < min_turns_before_end=2)
        self.assertTrue(any(e.turn == 1 and e.type == "END_RUN" and e.status == "rejected" for e in ch.events))
        # Turn 2: B votes END_RUN (1/3)
        # Turn 3: C's END_RUN succeeds -> majority (B, C) ends on turn 3!
        self.assertEqual(len(ch.public_messages), 3)
        self.assertIn("agents decided to end run (majority: B, C)", ch.stop_reason)

        # 2. END_RUN is NOT exempt when max_actions_per_response=0
        cfg_zero = {
            "name": "zero_actions",
            "topic": "Zero cap",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "any",
            "limits": {"max_turns": 2, "max_actions_per_response": 0},
            "agents": [{"name": "A", "endpoint": "local"}],
        }
        ch_zero = Chamber(
            cfg_zero,
            run_dir=self.root / "zero_cap",
            backend=MockBackend({"A": ["Try.\n@END_RUN: bypass?", "Turn 2."]}),
            quiet=True,
            copy_dir="",
        ).run()
        self.assertEqual(len(ch_zero.public_messages), 2)
        self.assertEqual(ch_zero.stop_reason, "reached max_turns (2)")
        self.assertTrue(any(e.type == "END_RUN" and "exceeded max_actions_per_response" in e.detail for e in ch_zero.events))

    def test_multi_agent_sampling_and_majority_termination_integration(self):
        """Lightweight 3-agent integration test: pipe sampling updates persist to payloads and majority END_RUN stops cleanly."""
        cfg = {
            "name": "integration_sampling_end",
            "topic": "Song collaboration",
            "scheduler": "self_directed",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 3,
            "artifacts": {"seed": {"lyrics": "# Song\n"}, "visible": ["lyrics"]},
            "limits": {"max_turns": 12, "max_actions_per_response": 6},
            "agents": [
                {"name": "GRIME_MC", "endpoint": "local", "capabilities": "full", "min_p": 0.04, "top_k": 60},
                {"name": "HONKYTONK", "endpoint": "local", "capabilities": "full", "min_p": 0.06, "top_k": 40},
                {"name": "GLITCH_DIR", "endpoint": "local", "capabilities": "full", "min_p": 0.05, "top_k": 50},
            ],
        }
        backend = MockBackend({
            "GRIME_MC": [
                "Bars one.\nTOP_P: 0.9 | TOP_K: 40 | MIN_P: 0.05 | XTC_PROB: 0.1 | REPEAT_PENALTY: 1.2\n@CLAIM: Hook is tight\n@APPEND lyrics: Verse 1\nNEXT: HONKYTONK",
                "Bars two locked.\n@VOTE C1 yes\n@APPEND lyrics: Final Cut\n@END_RUN: locked\nNEXT: NONE",
            ],
            "HONKYTONK": [
                "Verse two.\n@SET_SAMPLING min_p=0.08 repeat_penalty=1.15\n@CHALLENGE C1: Needs grit\n@APPEND lyrics: Verse 2\nNEXT: GLITCH_DIR",
            ],
            "GLITCH_DIR": [
                "Bridging.\n@VOTE c1 yes\n@END_RUN: ready to wrap\nNEXT: GRIME_MC",
            ],
        })
        ch = Chamber(cfg, run_dir=self.root / "integ", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(len(ch.public_messages), 4)
        self.assertIn("agents decided to end run (majority: GRIME_MC, GLITCH_DIR)", ch.stop_reason)
        rejected = [e for e in ch.events if e.status == "rejected"]
        self.assertEqual(rejected, [], msg=f"Unexpected rejections: {rejected}")
        # Verify GRIME_MC's 2nd API call (call index 3) received the updated sampling params from Turn 1
        grime_call_2_payload = backend.calls[3][1]
        self.assertEqual(grime_call_2_payload["top_p"], 0.9)
        self.assertEqual(grime_call_2_payload["top_k"], 40)
        self.assertEqual(grime_call_2_payload["min_p"], 0.05)
        self.assertEqual(grime_call_2_payload["xtc_probability"], 0.1)
        self.assertEqual(grime_call_2_payload["repeat_penalty"], 1.2)


if __name__ == "__main__":
    unittest.main()
