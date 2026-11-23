#!/usr/bin/env python3
"""Regression tests for section-level artifact collaboration, replace guards, citation audits, and scheduling hygiene."""
import json
import tempfile
import unittest
from pathlib import Path

from anomaly_lab import AnomalyLaboratory
from artifacts import ArtifactError, ArtifactManager, find_section_span, validate_artifact
from chamber import Chamber
from inference import MockBackend
from protocol import parse_reply


class ArtifactCollaborationAndAuditsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc_dir = self.root / "docs"
        self.doc_dir.mkdir()
        (self.doc_dir / "alpoge_2026_remark.txt").write_text("Iterated unitary theta lifts and Bost slope.", encoding="utf-8")
        (self.doc_dir / "siegel_brauer_2026.txt").write_text("Effective base-field c(k) > 0 via Masser.", encoding="utf-8")
        (self.doc_dir / "manifest.json").write_text(
            json.dumps({
                "documents": [
                    {"id": "alpoge_2026_remark", "title": "Alpoge Remark", "path": "alpoge_2026_remark.txt"},
                    {"id": "siegel_brauer_2026", "title": "Banks Siegel-Brauer", "path": "siegel_brauer_2026.txt"},
                ]
            }),
            encoding="utf-8",
        )

    def test_flexible_section_validation_matches_numbered_and_plain_headings(self):
        mgr = ArtifactManager(self.root / "val_arts")
        doc = "\n\n".join([
            "# Master Report",
            "## [1. Executive Overview & Post-RH Landscape]\nDetailed synthesis covering the post-RH hierarchy.",
            "## Primary-Source Audit - Kernel TCB & Estimate C\nSection III.2 proves Estimate C for clause-(b) primes.",
            "[Appendix Benchmarks]\nAll latency and KV-cache metrics verified.",
        ])
        mgr.write("ALPHA", "report", "create", doc, turn=1)
        mgr.write("ALPHA", "report", "append", "Closing verification note.", turn=2)
        ok, issues, details = validate_artifact(mgr, {
            "artifact": "report",
            "require_modified": True,
            "min_bytes": 100,
            "required_sections": [
                "[Executive Overview & Post-RH Landscape]",
                "[Primary-Source Audit – Kernel TCB & Estimate C]",
                "[Appendix Benchmarks]",
            ],
        })
        self.assertTrue(ok, msg=f"Validation failed unexpectedly: {issues}")
        self.assertEqual(details["matched_sections_count"], 3)

        # Body prose mentioning a section title must NOT satisfy the heading check
        mgr2 = ArtifactManager(self.root / "prose_arts")
        mgr2.write("A", "report", "create", "# Title\nSeed line.", turn=1)
        mgr2.write("A", "report", "replace", "Executive Overview & Post-RH Landscape is mentioned only inside a sentence here.", turn=2)
        ok2, issues2, _ = validate_artifact(mgr2, {
            "artifact": "report",
            "require_modified": True,
            "required_sections": ["[Executive Overview & Post-RH Landscape]"],
        })
        self.assertFalse(ok2)
        self.assertIn("missing required section", issues2[0])

    def test_section_command_parsing_and_surgical_section_updates(self):
        # Bracketed headings containing colons inside [...] must preserve the colon inside the title
        reply = (
            "Updating Section 2 cleanly.\n"
            "@SECTION report | [Primary-Source Audit: Lean TCB]: Section III.2 is distinct from H0-H12.\n"
            "Additional line in the same section.\n"
            "@SECTION report | Appendix Benchmarks\nBody starting on next line.\n"
            "@FORCE_REPLACE scratch: Intentionally reset scratchpad."
        )
        p = parse_reply(reply, me="A", names=("A", "B"))
        self.assertEqual(p.issues, [])
        self.assertEqual(len(p.actions), 3)
        self.assertEqual(p.actions[0]["mode"], "replace_section")
        self.assertEqual(p.actions[0]["section"], "[Primary-Source Audit: Lean TCB]")
        self.assertIn("Additional line in the same section.", p.actions[0]["content"])
        self.assertEqual(p.actions[1]["section"], "Appendix Benchmarks")
        self.assertEqual(p.actions[1]["content"], "Body starting on next line.")
        self.assertTrue(p.actions[2]["force"])

        # Apply @SECTION on multi-section artifact without disturbing neighboring sections or fenced code blocks
        mgr = ArtifactManager(self.root / "sec_arts")
        initial = (
            "# Report\n\n"
            "## [1. Overview]\nOriginal overview text.\n\n"
            "## [2. Primary-Source Audit: Lean TCB]\nOld stub.\n```md\n## Fake Heading Inside Code Fence\n```\nMore old text.\n\n"
            "## [3. Open Questions]\nKeep this third section intact.\n"
        )
        mgr.write("A", "report", "create", initial, turn=1)
        mgr.write("B", "report", "append", "Minor footer note.", turn=2)
        mgr.read("A", "report", turn=3)

        span = find_section_span(mgr.read("A", "report", turn=3)[2], "[Primary-Source Audit: Lean TCB]")
        self.assertIsNotNone(span)
        mgr.write(
            "A", "report", "replace_section",
            "Updated rigorous audit citing `rubinstein_2026_zeta_guide @ 826576`.",
            section="[Primary-Source Audit: Lean TCB]",
            turn=3,
        )
        # Append brand-new sections via replace_section when missing
        mgr.write(
            "B", "report", "replace_section",
            "Brand new section four content.",
            section="[Future Work]",
            turn=4,
            expected_version=3,
        )
        mgr.read("A", "report", turn=5)
        mgr.write(
            "A", "report", "replace_section",
            "Plain-heading appendix body.",
            section="Unbracketed Appendix",
            turn=5,
        )
        final_text = mgr.read("A", "report", turn=6)[2]
        self.assertIn("Original overview text.", final_text)
        self.assertIn("Updated rigorous audit citing", final_text)
        self.assertNotIn("Old stub.", final_text)
        self.assertIn("Keep this third section intact.", final_text)
        self.assertIn("[Future Work]\n\nBrand new section four content.", final_text)
        self.assertIn("## Unbracketed Appendix\n\nPlain-heading appendix body.", final_text)

    def test_destructive_replace_guards(self):
        long_body = ("Detailed derivation step and verified bound.\n" * 60)  # > 2KB
        cfg = {
            "name": "replace_guard",
            "topic": "Protect large drafts from accidental overwrites",
            "scheduler": "round_robin",
            "artifacts": {"seed": {"report": f"# Draft\n## [Intro]\n{long_body}"}, "visible": ["report"]},
            "limits": {"max_turns": 4},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
                {"name": "GAMMA", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            # Turn 1: ALPHA emits two @REPLACEs in the same turn -> second rejected!
            "ALPHA": [
                f"Turn 1 write.\n@REPLACE report: # Draft\n## [Intro]\n{long_body}\n@REPLACE report: Tiny stub!",
            ],
            # Turn 2: BETA tries tiny @REPLACE -> rejected by min_replace_ratio; then Turn 3 GAMMA uses @FORCE_REPLACE!
            "BETA": [
                "Overwriting accidentally.\n@REPLACE report: Short accidental wipe.",
            ],
            "GAMMA": [
                "Placeholder then force.\n@REPLACE report: ... (Insert full draft here)\n@FORCE_REPLACE report: Intentional concise reset.",
            ],
        })
        ch = Chamber(cfg, run_dir=self.root / "guard_run", backend=backend, quiet=True, copy_dir="").run()
        rejected = [e for e in ch.events if e.type == "ARTIFACT_WRITE" and e.status == "rejected"]
        self.assertTrue(any("duplicate @REPLACE" in e.detail for e in rejected))
        self.assertTrue(any("would shrink artifact" in e.detail for e in rejected))
        self.assertTrue(any("placeholder stub" in e.detail for e in rejected))
        # GAMMA's @FORCE_REPLACE intentionally succeeded on Turn 3
        self.assertEqual(ch.artifacts.read("ALPHA", "report")[2].strip(), "Intentional concise reset.")

    def test_premature_end_run_feedback_stray_artifact_hint_and_status_formatting(self):
        cfg = {
            "name": "deliv_fb_test",
            "topic": "Complete the required deliverable",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "artifacts": {
                "seed": {"frontier_monograph": "# Seed\n"},
                "visible": ["frontier_monograph"],
                "validation": {
                    "artifact": "frontier_monograph",
                    "require_modified": True,
                    "min_bytes": 200,
                    "required_sections": ["[Overview]", "[Conclusion]"],
                },
            },
            "limits": {"max_turns": 3},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
                {"name": "GAMMA", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            # Turn 1: ALPHA creates lookalike stray monograph.md and votes END_RUN early (1/3 < majority)
            "ALPHA": ["Done?\n@APPEND monograph: Wrote to wrong file.\n@END_RUN: ready", "Turn 3 check."],
            "BETA": ["Checking status."],
            "GAMMA": ["Turn 3."],
        })
        ch = Chamber(cfg, run_dir=self.root / "fb_run", backend=backend, quiet=True, copy_dir="").run()
        # BETA on Turn 2 sees byte count and bullet issues in Session Status
        t2_tail = backend.calls[1][1]["messages"][-1]["content"]
        self.assertIn("frontier_monograph.md (v1,", t2_tail)
        self.assertIn("Deliverable `frontier_monograph.md`: INCOMPLETE (v1,", t2_tail)
        self.assertIn("- frontier_monograph.md is still at v1", t2_tail)

        # ALPHA gets both DELIVERABLE_HINT and DELIVERABLE_WARNING in events (and next-turn feedback if called)
        self.assertTrue(any(e.type == "DELIVERABLE_HINT" and e.target == "monograph.md" for e in ch.events))
        self.assertTrue(any(e.type == "DELIVERABLE_WARNING" and e.actor == "ALPHA" for e in ch.events))

    def test_doc_citation_audit_and_anomaly_detectors(self):
        cfg = {
            "name": "cite_audit_test",
            "topic": "Cite sources responsibly",
            "scheduler": "round_robin",
            "doc_manifest": str(self.doc_dir / "manifest.json"),
            "limits": {"max_turns": 3},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            # Turn 1: ALPHA cites both mounted docs without reading either
            "ALPHA": ["As shown in [alpoge_2026_remark] and [siegel_brauer_2026], c(k) > 0."],
            # Turn 2: BETA actually reads alpoge_2026_remark
            "BETA": ["Inspecting first doc.\n@DOC_READ alpoge_2026_remark 0"],
        })
        ch = Chamber(cfg, run_dir=self.root / "cite_run", backend=backend, quiet=True, copy_dir="").run()
        # Turn 2 prompt shows unverified citations for both docs; metrics show only siegel_brauer_2026 unverified at end
        t2_tail = backend.calls[1][1]["messages"][-1]["content"]
        self.assertIn("Unverified doc citations", t2_tail)
        self.assertIn("alpoge_2026_remark", t2_tail)
        metrics = ch.compute_metrics()
        self.assertEqual(metrics["unverified_doc_citations"], {"siegel_brauer_2026": 1})

        # AnomalyLaboratory detects unverified_source_citation when >= 2 docs are cited unread + destructive overwrite
        fake_run = self.root / "fake_anom_run"
        fake_run.mkdir()
        (fake_run / "metrics.json").write_text(json.dumps({
            "turns_completed": 5,
            "unverified_doc_citations": {"alpoge_2026_remark": 3, "siegel_brauer_2026": 2},
        }), encoding="utf-8")
        (fake_run / "checkpoint.json").write_text(json.dumps({
            "schema_version": 2,
            "state": {
                "config": {"scheduler": "self_directed", "agents": [{"name": "A"}, {"name": "B"}]},
                "artifacts": {
                    "log": [
                        {"turn": 1, "agent": "A", "op": "create", "name": "doc.md", "version": 1, "ok": True, "detail": "2500 chars"},
                        {"turn": 2, "agent": "A", "op": "replace", "name": "doc.md", "version": 2, "ok": True, "detail": "22 chars"},
                    ]
                },
            },
        }), encoding="utf-8")
        (fake_run / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
        (fake_run / "events.jsonl").write_text("", encoding="utf-8")
        lab = AnomalyLaboratory(lab_dir=self.root / "anom_lab")
        found = {sp.anomaly_type for sp in lab.inspect_run(fake_run)}
        self.assertIn("destructive_artifact_overwrite", found)
        self.assertIn("unverified_source_citation", found)

    def test_temperature_noop_claim_hygiene_and_hub_breaker(self):
        cfg = {
            "name": "hygiene_hub_test",
            "topic": "Prevent no-op temp, duplicate/ID claims, and hub star lock-in",
            "scheduler": "self_directed",
            "limits": {
                "max_turns": 8,
                "max_consecutive": 1,
                "max_dyad_turns": 4,
                "starvation_turns": 20,
                "max_speaker_window_turns": [4, 2],
            },
            "agents": [
                {"name": "HUB", "endpoint": "local", "temperature": 0.8},
                {"name": "A", "endpoint": "local", "temperature": 0.8},
                {"name": "B", "endpoint": "local", "temperature": 0.8},
                {"name": "C", "endpoint": "local", "temperature": 0.8},
                {"name": "D", "endpoint": "local", "temperature": 0.8},
            ],
        }
        backend = MockBackend({
            # T1 HUB -> A, T3 HUB -> B, T5 HUB -> C (meanwhile A & B both nominate HUB -> HUB hits 2/4 turns!)
            "HUB": [
                "Hub T1.\nTEMP: 0.8\n@CLAIM: Exact shared hypothesis.\n@CLAIM: Exact   shared hypothesis.\nNEXT: A",
                "Hub T3.\nNEXT: B",
                "Hub T5 fallback.\nNEXT: C",
            ],
            "A": ["A T2.\n@CLAIM: c1\nNEXT: HUB", "A second."],
            "B": ["B T4.\nNEXT: HUB", "B second."],
            "C": ["C speaks after hub breaker!\nNEXT: D"],
            "D": ["D speaks!\nNEXT: A"],
        })
        ch = Chamber(cfg, run_dir=self.root / "hub_run", backend=backend, quiet=True, copy_dir="").run()
        # No-op temperature 0.8 -> 0.8 did not pollute temp_changes
        self.assertEqual(ch.temp_changes, [])
        # Duplicate claim + bare existing claim ID 'c1' were rejected; only one c1 exists
        self.assertEqual(len(ch.board), 1)
        rejected_claims = [e for e in ch.events if e.type == "CLAIM" and e.status == "rejected"]
        self.assertEqual(len(rejected_claims), 2)
        # Hub-lock-in: T1=HUB, T2=A, T3=HUB, T4=B -> HUB now has 2/4 turns -> T5 breaks away from HUB to C!
        speakers = [m.speaker for m in ch.public_messages]
        self.assertEqual(speakers[:4], ["HUB", "A", "HUB", "B"])
        self.assertNotEqual(speakers[4], "HUB")
        self.assertTrue(any("hub lock-in breaker" in e.detail for e in ch.events))


if __name__ == "__main__":
    unittest.main()
