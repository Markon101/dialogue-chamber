#!/usr/bin/env python3
"""Tests for Dialogue Chamber harness upgrades:
1. Large artifact writes, replacement, append, per-artifact limits, and informative errors.
2. Successful final-artifact validation.
3. Rejected premature completion followed by successful repair.
4. Voluntary ending when no required deliverable exists.
5. Larger model responses and multiple protocol actions.
6. Sampling parameter changes (aliases, trailing commentary, strict rejection) and telemetry.
7. Context-window constraints and output-token reservations.
8. Checkpoint/resume with large artifacts and updated limits.
"""
import json
import tempfile
import unittest
from pathlib import Path

from artifacts import ArtifactError, ArtifactManager, validate_artifact
from chamber import Chamber
from context import ContextIntelligence
from inference import MockBackend, Result
from presets import LIMITS
from protocol import parse_reply


class HarnessUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_effective_default_limits_raised(self):
        self.assertGreaterEqual(LIMITS["max_artifact_bytes"], 131072)
        self.assertGreaterEqual(LIMITS["max_tokens_per_response"], 16384)
        self.assertGreaterEqual(LIMITS["max_message_chars"], 32768)
        self.assertGreaterEqual(LIMITS["max_actions_per_response"], 16)
        self.assertGreaterEqual(LIMITS["max_total_tokens"], 1000000)
        self.assertGreaterEqual(LIMITS["max_duration_s"], 7200)

    def test_large_artifact_write_replace_append_and_informative_error(self):
        mgr = ArtifactManager(self.root / "arts", max_bytes=131072, max_total_bytes=524288,
                              per_artifact_max_bytes={"tiny": 500})
        big_text = "Verse line\n" * 4000  # > 44KB (well above old 16KB cap)
        fname, ver = mgr.write("A", "lyrics", "create", big_text, turn=1)
        self.assertEqual(fname, "lyrics.md")
        self.assertEqual(ver, 1)

        _, ver2 = mgr.write("A", "lyrics", "append", "Outro line", turn=2)
        self.assertEqual(ver2, 2)

        _, ver3 = mgr.write("A", "lyrics", "replace", "# Clean Master\nAll consolidated.", turn=3)
        self.assertEqual(ver3, 3)
        self.assertIn("Clean Master", mgr.read("A", "lyrics")[2])

        # Per-artifact limit and informative error
        mgr.write("A", "tiny", "create", "hello" * 20, turn=1)
        with self.assertRaises(ArtifactError) as ctx:
            mgr.write("A", "tiny", "append", "z" * 450, turn=2)
        err = str(ctx.exception)
        self.assertIn("current=", err)
        self.assertIn("update=", err)
        self.assertIn("exceeds limit=500B", err)
        self.assertIn("@REPLACE tiny", err)

    def test_premature_end_run_blocked_then_repaired_and_verified(self):
        valid_song = "\n".join([
            "# Track Title: The Bellman Blast of Princess Flatulencia",
            "[Intro]\nHarpsichord shrieks in D-minor over an 808 methane drone, setting the royal stage.",
            "[Chorus]\nFlatulencia, queen of the Q-table, tearing through the velvet veil at fifty reward!",
            "[Verse 1]\nRoyal MDP states from the Moat Crossing to the Throne, alpha beans updating every cell.",
            "[Verse 2]\nEpsilon at zero point nine, exploring wildly through the banquet until greedy exploitation wins.",
            "[Bridge]\nTarget network frozen while the King speaks, sampling mini-batches from the bean diary.",
            "[Verse 3]\nTD-error delta spikes to ninety-eight decibels, fainting the Royal Corgi and burning the wig.",
            "[Verse 4]\nPrince SARSA warns of on-policy cliffs while off-policy thrust vectoring clears the moat.",
            "[Grand Finale Outro]\nConvergence to optimal Q-star flatulence, gamma decays into a royal supernova!",
        ])
        cfg = {
            "name": "deliverable_test",
            "topic": "Write the song",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "artifacts": {
                "seed": {"lyrics": "# Track Title: TBD\n"},
                "visible": ["lyrics"],
                "validation": {
                    "artifact": "lyrics",
                    "require_modified": True,
                    "min_bytes": 400,
                    "disallow_placeholders": True,
                    "required_sections": [
                        "[Intro]", "[Chorus]", "[Verse 1]", "[Verse 2]",
                        "[Bridge]", "[Verse 3]", "[Verse 4]", "[Grand Finale Outro]",
                    ],
                },
            },
            "limits": {"max_turns": 8},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full"},
                {"name": "B", "endpoint": "local", "capabilities": "full"},
                {"name": "C", "endpoint": "local", "capabilities": "full"},
            ],
        }
        # Turn 1: A appends a placeholder and votes END_RUN
        # Turn 2: B votes END_RUN -> majority reached, MUST BE BLOCKED by validation!
        # Turn 3: C sees feedback, replaces lyrics.md with full valid song, and votes END_RUN -> VERIFIED & ENDS!
        backend = MockBackend({
            "A": [
                "@APPEND lyrics: # Track Title: The Bellman Blast ... (full lyrics content)\n@END_RUN: done",
            ],
            "B": [
                "Looks great!\n@END_RUN: agree",
            ],
            "C": [
                f"@REPLACE lyrics: {valid_song}\n@END_RUN: repaired and complete",
            ],
        })
        ch = Chamber(cfg, run_dir=self.root / "repair_run", backend=backend, quiet=True, copy_dir="")
        ch.run()

        self.assertEqual(ch.turn, 3)
        self.assertTrue(ch.end_requested)
        self.assertTrue(ch.deliverable_verified)
        self.assertEqual(ch.deliverable_issues, [])
        self.assertTrue(any(e.type == "END_RUN" and e.status == "rejected" for e in ch.events))
        self.assertTrue(any(e.type == "DELIVERABLE_VERIFIED" for e in ch.events))

        metrics = ch.compute_metrics()
        self.assertTrue(metrics["agents_voted_to_stop"])
        self.assertTrue(metrics["run_ended"])
        self.assertTrue(metrics["deliverable_required"])
        self.assertTrue(metrics["deliverable_verified"])

    def test_voluntary_ending_without_deliverable_validation(self):
        cfg = {
            "name": "freeform_end",
            "topic": "Chat freely",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "limits": {"max_turns": 6},
            "agents": [
                {"name": "A", "endpoint": "local"},
                {"name": "B", "endpoint": "local"},
            ],
        }
        backend = MockBackend({
            "A": ["Nice chat.\n@END_RUN: satisfied"],
            "B": ["Agreed.\n@END_RUN: done"],
        })
        ch = Chamber(cfg, run_dir=self.root / "voluntary_run", backend=backend, quiet=True, copy_dir="")
        ch.run()
        self.assertEqual(ch.turn, 2)
        self.assertTrue(ch.end_requested)
        metrics = ch.compute_metrics()
        self.assertFalse(metrics["deliverable_required"])
        self.assertIsNone(metrics["deliverable_verified"])

    def test_large_message_and_many_protocol_actions(self):
        long_verse = "La " * 4000  # ~12,000 chars (> old 3500 limit, < 32768)
        action_lines = [f"@CLAIM idea_{i}" for i in range(1, 13)]
        action_lines.append("@VOTE c1,c2,c3 yes")
        raw = long_verse + "\n" + "\n".join(action_lines)
        parsed = parse_reply(raw, me="A", names=("A", "B"))
        self.assertGreater(len(parsed.message), 10000)
        # 12 claims + 3 expanded votes = 15 actions (within default 16)
        self.assertEqual(len(parsed.actions), 15)

        # Lyrical text must NOT accidentally execute as protocol instructions
        lyric_only = parse_reply(
            "Verse 1:\nPrincess Q(s,a) sings NEXT to the throne\nTEMPest in a teapot!",
            me="A", names=("A", "B"),
        )
        self.assertEqual(lyric_only.actions, [])
        self.assertIsNone(lyric_only.next_speaker)

    def test_sampling_trailing_commentary_strict_rejection_and_telemetry(self):
        # Trailing parenthetical commentary allowed
        p_ok = parse_reply(
            "Boosting fuzz.\n@SET_SAMPLING xtc_prob=0.4 xtc_thresh=0.2 rep_pen=1.3 top_k=38 min_p=0.07   (Maxing the fuzz!)",
            me="A", names=("A", "B"),
        )
        self.assertEqual(len(p_ok.actions), 1)
        self.assertEqual(p_ok.actions[0]["params"]["xtc_probability"], 0.4)
        self.assertEqual(p_ok.actions[0]["params"]["repeat_penalty"], 1.3)

        # Typo in parameter key must still be strictly rejected with valid param hint
        p_bad = parse_reply(
            "@SET_SAMPLING extxct_thresh=0.2 min_p=0.07",
            me="A", names=("A", "B"),
        )
        self.assertEqual(p_bad.actions, [])
        self.assertTrue(any("valid parameters:" in iss for iss in p_bad.issues))

        # Telemetry distinguishes configured, transmitted, and backend-confirmed cached_tokens
        cfg = {
            "name": "telemetry_test",
            "topic": "Check telemetry",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2},
            "agents": [
                {"name": "A", "endpoint": "local", "min_p": 0.05, "top_k": 40},
                {"name": "B", "endpoint": "local", "temperature": 0.7},
            ],
        }
        backend = MockBackend({
            "A": [Result(content="Turn 1.\n@SET_SAMPLING min_p=0.09", finish_reason="stop",
                         prompt_tokens=100, completion_tokens=20, cached_tokens=64,
                         cache_source="usage.prompt_tokens_details.cached_tokens", latency=0.5, actual_model="mock-gemma")],
            "B": [Result(content="Turn 2.", finish_reason="stop",
                         prompt_tokens=120, completion_tokens=10, latency=0.25, actual_model="mock-gemma")],
        })
        ch = Chamber(cfg, run_dir=self.root / "telemetry_run", backend=backend, quiet=True, copy_dir="")
        ch.run()
        self.assertEqual(len(ch.inference_history), 2)
        h0 = ch.inference_history[0]
        self.assertEqual(h0["configured_sampling"]["min_p"], 0.05)
        self.assertEqual(h0["transmitted_sampling"]["min_p"], 0.05)
        self.assertEqual(h0["cached_tokens"], 64)
        self.assertEqual(h0["cache_source"], "usage.prompt_tokens_details.cached_tokens")
        self.assertEqual(h0["sampling_requests"][0]["status"], "accepted")
        self.assertIsNone(ch.inference_history[1]["cached_tokens"])

    def test_context_window_negotiation_above_131k_and_output_reservation(self):
        ci = ContextIntelligence()
        limit, src, conf = ci.negotiate_context_limit(server_reported_ctx=262144, user_configured_ctx="auto")
        self.assertEqual(limit, 262144)
        self.assertEqual(src, "server-reported")
        self.assertEqual(conf, "exact")

        # Never exceed server-reported limit even if user asks for more
        limit2, src2, _ = ci.negotiate_context_limit(server_reported_ctx=73728, user_configured_ctx=262144)
        self.assertEqual(limit2, 73728)
        self.assertEqual(src2, "server-reported")

    def test_checkpoint_resume_preserves_large_artifacts_and_telemetry(self):
        cfg = {
            "name": "resume_large",
            "topic": "Resume test",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2, "max_artifact_bytes": 131072, "max_workspace_bytes": 524288},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full"},
                {"name": "B", "endpoint": "local", "capabilities": "full"},
            ],
        }
        big_art = "Methane line\n" * 3000  # ~39KB
        backend1 = MockBackend({"A": [f"@APPEND big: {big_art}"], "B": ["Ack."]})
        ch1 = Chamber(cfg, run_dir=self.root / "r1", backend=backend1, quiet=True, copy_dir="")
        ch1.run()
        saved = json.loads(ch1.checkpoints.checkpoint_file.read_text())

        cfg2 = {**cfg, "limits": {**cfg["limits"], "max_turns": 3}}
        backend2 = MockBackend({"A": ["Turn 3."], "B": ["Turn 3."]})
        ch2 = Chamber(cfg2, run_dir=self.root / "r2", backend=backend2, quiet=True, copy_dir="", checkpoint_data=saved)
        self.assertEqual(len(ch2.inference_history), 2)
        self.assertGreater(len(ch2.artifacts.read("A", "big")[2]), 35000)
        ch2.run()
        self.assertEqual(ch2.turn, 3)


if __name__ == "__main__":
    unittest.main()
