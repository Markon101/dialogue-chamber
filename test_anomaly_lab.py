#!/usr/bin/env python3
"""Tests for AnomalyLaboratory, Prompt Leakage Sanitization, and Epistemic Metrics."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from anomaly_lab import AnomalyLaboratory
from chamber import Chamber
from inference import MockBackend
from protocol import canonicalize_agent_id, parse_reply, sanitize_speaker


class TestAnomalyLab(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="anomaly_test_"))
        self.lab = AnomalyLaboratory(lab_dir=self.tmp / "anomalies")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_canonicalize_parenthetical_echo(self):
        """Verify prompt-template parenthetical echo is stripped cleanly."""
        names = {"ALPHA", "BETA", "HERMES"}
        self.assertEqual(canonicalize_agent_id("BETA (ALPHA, BETA, HERMES)", names), "BETA")
        self.assertEqual(canonicalize_agent_id("HERMES (ALPHA, BETA, HERMES)", names), "HERMES")
        self.assertEqual(canonicalize_agent_id("@BETA (ALPHA, BETA)", names), "BETA")

    def test_sanitize_transcript_tags_and_section_leaks(self):
        """Verify hallucinatory turn-tag headers and prompt section delimiters are stripped."""
        names = {"ALPHA", "BETA", "HERMES"}
        flags = []
        raw = (
            "I deduce that Dax is the thief.\n"
            "--- Shared Blackboard ---\n"
            "- c1 [claim by BETA]: Sela is the thief.\n"
            "[m2 | turn 2] BETA: I agree with that."
        )
        cleaned = sanitize_speaker(raw, me="ALPHA", names=names, flags=flags)
        self.assertEqual(cleaned.strip(), "I deduce that Dax is the thief.")
        self.assertTrue(any("leaked prompt section" in f for f in flags))

        # Repeated self turn tag mid-reply
        flags2 = []
        raw2 = (
            "Initial thought.\n"
            "[m1 | turn 4] ALPHA: A continuation that should be cut."
        )
        cleaned2 = sanitize_speaker(raw2, me="ALPHA", names=names, flags=flags2)
        self.assertEqual(cleaned2.strip(), "Initial thought.")
        self.assertTrue(any("hallucinated subsequent transcript turn" in f for f in flags2))

    def test_verified_synthesis_vs_regex_proxy(self):
        """Verify that suspect enumeration triggers regex proxy but NOT verified_synthesis_turn until all clue-holders speak."""
        backend = MockBackend({
            "ALPHA": [
                # Turn 1: enumerates suspects -> triggers regex proxy because 'thief' and 'Dax' are within 80 chars
                "The thief used the east path at dawn; only Mira, Sela, and Dax used that path.",
                # Turn 4: genuine synthesis after BETA and HERMES have spoken
                "Since Mira and Sela are eliminated, Dax took the lantern.",
            ],
            "BETA": [
                # Turn 2: Sela alibi
                "Sela was confirmed sick with fever in bed all night by the doctor.",
            ],
            "HERMES": [
                # Turn 3: Mira glasses clue
                "The thief wore no glasses; Mira always wears her glasses.",
            ],
        })

        cfg = {
            "name": "synth_test",
            "topic": "Lantern mystery.",
            "scheduler": "round_robin",
            "limits": {"max_turns": 4},
            "answer_key": {
                "answer": "Dax",
                "pattern": r"(\bDax\b[^.?!\n]{0,80}\b(took|stole|thief|culprit|did it|guilty)\b)|"
                           r"(\b(thief|culprit|took it|took the lantern|stole it)\b[^.?!\n]{0,80}\bDax\b)",
            },
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "knowledge": "East path: Mira, Sela, Dax."},
                {"name": "BETA", "endpoint": "local", "knowledge": "Sela was in bed with fever."},
                {"name": "HERMES", "endpoint": "local", "knowledge": "Thief had no glasses; Mira wears glasses."},
            ],
        }

        ch = Chamber(cfg, run_dir=self.tmp / "run_synth", backend=backend, quiet=True, copy_dir="")
        ch.run()
        m = ch.compute_metrics()

        # Turn 1 triggered regex proxy
        self.assertEqual(m["solved_turn"], 1)
        # Turn 4 triggered verified synthesis (all clue-holders had spoken and it was not a suspect list)
        self.assertEqual(m["verified_synthesis_turn"], 4)

    def test_anomaly_lab_specimen_detection_and_catalog(self):
        """Test scanning a run with known anomalies, saving JSON specimens, and generating CATALOG.md."""
        run_dir = self.tmp / "run_with_anomalies"
        run_dir.mkdir(parents=True, exist_ok=True)

        events = [
            {"seq": 1, "turn": 1, "actor": "ALPHA", "type": "PROTOCOL", "status": "rejected",
             "detail": "unknown next speaker 'BETA_ALPHA_BETA_'"},
            {"seq": 2, "turn": 1, "actor": "ALPHA", "type": "PROPOSE_NEXT", "status": "rejected",
             "target": "MIRA", "detail": "invalid or inactive participant"},
            {"seq": 3, "turn": 1, "actor": "ALPHA", "type": "SAY", "status": "accepted",
             "target": "m1", "detail": "The thief was Mira, Sela, or Dax."},
        ]
        (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
        (run_dir / "metrics.json").write_text(json.dumps({
            "solved_turn": 1,
            "verified_synthesis_turn": None,
            "turns_completed": 1,
            "participation": {"ALPHA": {"model": "test-model"}},
        }))
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": "test_anomaly_run_01",
            "effective_config": {
                "scheduler": "self_directed",
                "agents": [
                    {"name": "ALPHA", "knowledge": "Clue A"},
                    {"name": "BETA", "knowledge": "Clue B"},
                    {"name": "HERMES", "knowledge": "Clue C"},
                ],
            },
            "roster": [
                {"name": "ALPHA", "model": "test-model"},
                {"name": "BETA", "model": "test-model"},
                {"name": "HERMES", "model": "test-model"},
            ],
        }))
        (run_dir / "transcript.md").write_text(
            "# Transcript\n### [Turn 01 | m1] ALPHA\n"
            "The thief was Mira, Sela, or Dax.\n"
            "--- Shared Blackboard ---\n"
            "[m2 | turn 2] BETA: The blackboard is just a red herring distraction.\n"
        )

        specimens = self.lab.scan_directory(run_dir, save=True)
        self.assertGreaterEqual(len(specimens), 3)

        types_found = {s.anomaly_type for s in specimens}
        self.assertIn("template_echo_speaker_id", types_found)
        self.assertIn("ontological_boundary_crossing", types_found)
        self.assertIn("premature_regex_answer_proxy", types_found)
        self.assertIn("prompt_leakage_and_institutional_paranoia", types_found)

        # Verify JSON artifacts exist
        json_files = list(self.lab.specimens_dir.glob("*.json"))
        self.assertEqual(len(json_files), len(specimens))

        # Verify CATALOG.md is regenerated
        self.assertTrue(self.lab.catalog_path.exists())
        catalog_text = self.lab.catalog_path.read_text()
        self.assertIn("Dialogue Chamber — Anomaly Laboratory Catalog", catalog_text)
        self.assertIn("Classification Taxonomy", catalog_text)
        self.assertIn("template_echo_speaker_id", catalog_text)


if __name__ == "__main__":
    unittest.main()
