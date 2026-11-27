#!/usr/bin/env python3
"""Regression tests for section-level artifact collaboration, replace guards, citation audits, and scheduling hygiene."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anomaly_lab import AnomalyLaboratory
from artifacts import ArtifactError, ArtifactManager, find_section_span, normalize_escaped_newlines, validate_artifact
from chamber import Chamber
from context import ContextIntelligence
from documents import DocumentLibrary
from inference import HttpBackend, MockBackend, Result, build_payload
from protocol import canonicalize_agent_id, parse_reply


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

    def test_artifact_version_history_revert_and_checkpoint_compat(self):
        mgr = ArtifactManager(self.root / "hist_arts", max_history_per_artifact=8)
        v1_text = "# Research Report\n\n## [Overview]\nInitial seed body.\n"
        mgr.write("SYSTEM", "report", "create", v1_text, turn=0, attribution=False)

        rich_v2_body = ("Verified theorem citation and proof sketch.\n" * 65)  # ~2.7KB
        mgr.write("ALPHA", "report", "append", rich_v2_body, turn=1, attribution=False)
        mgr.read("BETA", "report", turn=2)

        # Partial shrinking rewrite (~1.4KB, > 25% floor so allowed under replace guard)
        shrunk_v3 = "# Research Report\n\n" + ("Compressed partial summary.\n" * 50)
        mgr.write("BETA", "report", "replace", shrunk_v3, turn=2, min_replace_ratio=0.25)

        trail = mgr.recent_history_summary("report", limit=4)
        self.assertIn("v1(", trail)
        self.assertIn("v2(", trail)
        self.assertIn(", +", trail)
        self.assertIn("v3(", trail)
        self.assertIn(", -", trail)

        # Validation errors on revert: invalid token, current version, missing version
        with self.assertRaisesRegex(ArtifactError, "invalid target version"):
            mgr.revert("CHAIR", "report", "not_a_ver", turn=3)
        with self.assertRaisesRegex(ArtifactError, "already at v3"):
            mgr.revert("CHAIR", "report", "v3", turn=3)
        with self.assertRaisesRegex(ArtifactError, "no saved history for v99"):
            mgr.revert("CHAIR", "report", 99, turn=3)

        # Successful revert to v2 restores exact v2 content and bumps to v4
        fname, new_ver, restored_from, size_b, delta_b = mgr.revert("CHAIR", "report", "v2", turn=3)
        self.assertEqual(fname, "report.md")
        self.assertEqual(restored_from, 2)
        self.assertEqual(new_ver, 4)
        self.assertGreater(delta_b, 0)
        _, check_ver, check_text = mgr.read("ALPHA", "report", turn=4)
        self.assertEqual(check_ver, 4)
        self.assertIn("Initial seed body.", check_text)
        self.assertIn("Verified theorem citation", check_text)
        self.assertEqual(mgr.seen[("CHAIR", "report.md")], 4)

        # Snapshot/restore round-trip preserves full history cleanly
        snap = mgr.snapshot()
        mgr_restored = ArtifactManager(self.root / "hist_restored")
        mgr_restored.restore(snap)
        self.assertEqual(mgr_restored.snapshot(), snap)
        self.assertEqual(len(mgr_restored.history["report.md"]), 4)

        # Legacy snapshot without 'history' key restores gracefully
        legacy_snap = {k: v for k, v in snap.items() if k != "history"}
        mgr_legacy = ArtifactManager(self.root / "hist_legacy")
        mgr_legacy.restore(legacy_snap)
        self.assertEqual(mgr_legacy.versions["report.md"], 4)
        self.assertEqual(len(mgr_legacy.history["report.md"]), 1)

        # Corrupt history entry fails checksum check BEFORE touching files on disk
        tampered_snap = json.loads(json.dumps(snap))
        tampered_snap["history"]["report.md"][0]["content"] = "tampered history!"
        mgr_corrupt_dir = self.root / "hist_corrupt"
        mgr_corrupt = ArtifactManager(mgr_corrupt_dir)
        with self.assertRaisesRegex(ArtifactError, "history checksum mismatch"):
            mgr_corrupt.restore(tampered_snap)
        self.assertFalse((mgr_corrupt_dir / "report.md").exists())

    def test_revert_protocol_parsing_and_chamber_recovery(self):
        # Single-line @REVERT must NOT swallow subsequent speech lines
        reply = (
            "Rolling back after seeing shrinkage.\n"
            "@REVERT report v2\n"
            "@ARTIFACT_REVERT ledger: 4\n"
            "@ROLLBACK notes to v1\n"
            "This closing line remains public speech."
        )
        p = parse_reply(reply, me="CHAIR", names=("CHAIR", "ALPHA"))
        self.assertEqual(p.issues, [])
        self.assertEqual(len(p.actions), 3)
        self.assertEqual(p.actions[0], {"type": "ARTIFACT_WRITE", "name": "report", "mode": "revert", "target_version": 2, "content": ""})
        self.assertEqual(p.actions[1], {"type": "ARTIFACT_WRITE", "name": "ledger", "mode": "revert", "target_version": 4, "content": ""})
        self.assertEqual(p.actions[2], {"type": "ARTIFACT_WRITE", "name": "notes", "mode": "revert", "target_version": 1, "content": ""})
        self.assertIn("Rolling back after seeing shrinkage.", p.message)
        self.assertIn("This closing line remains public speech.", p.message)

        p_json = parse_reply(json.dumps({
            "message": "Reverting via JSON.",
            "actions": [{"type": "ARTIFACT_REVERT", "name": "report", "version": 2}],
        }), me="CHAIR", names=("CHAIR",))
        self.assertEqual(p_json.issues, [])
        self.assertEqual(p_json.actions[0]["mode"], "revert")
        self.assertEqual(p_json.actions[0]["target_version"], 2)

        p_bad = parse_reply("@REVERT report nope", me="CHAIR", names=("CHAIR",))
        self.assertTrue(any("invalid target version" in iss for iss in p_bad.issues))

        # End-to-end Chamber recovery: ALPHA fills [Overview], BETA degrades it via @REPLACE, CHAIR @REVERTs to v2, ALPHA finishes [Audit]
        overview_body = "Rigorous Section 1 analysis of Stepanov hyperderivatives and theta lifts. " * 6
        audit_body = "Verified Section 2 primary-source audit and Lean TCB check. " * 6
        cfg = {
            "name": "revert_recovery_test",
            "topic": "Collaborate and recover from a shrinking rewrite",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 4,
            "artifacts": {
                "seed": {"report": "# Master Report\n\n## [Overview]\n\n## [Audit]\n"},
                "visible": ["report"],
                "validation": {
                    "artifact": "report",
                    "require_modified": True,
                    "min_bytes": 250,
                    "required_sections": ["[Overview]", "[Audit]"],
                },
            },
            "limits": {"max_turns": 6, "min_replace_ratio": 0.25},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
                {"name": "CHAIR", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            "ALPHA": [
                f"Drafted Overview.\n@SECTION report | [Overview]: {overview_body}",
                f"Completed Audit and voting end.\n@SECTION report | [Audit]: {audit_body}\n@END_RUN: Deliverable complete",
            ],
            "BETA": [
                # Degrades artifact by replacing everything with ~200B (> 25% of ~440B, and < 1500B threshold) dropping [Overview]
                "Rewrote quickly.\n@REPLACE report: # Master Report\n\n## [Audit]\nTruncated draft without overview section at all, losing prior peer work in the process.",
            ],
            "CHAIR": [
                "Noticed size drop and lost [Overview]; reverting to v2!\n@REVERT report v2",
                "Verified both sections intact >= 250B; concurring.\n@END_RUN: Verified ready",
            ],
        })
        ch = Chamber(cfg, run_dir=self.root / "revert_run", backend=backend, quiet=True, copy_dir="").run()
        # CHAIR on Turn 3 saw version history & delta trail in prompt
        t3_tail = backend.calls[2][1]["messages"][-1]["content"]
        self.assertIn("Version history:", t3_tail)
        self.assertIn("Versions: v1(", t3_tail)
        self.assertTrue(ch.deliverable_verified)
        self.assertIn("agents decided to end run", ch.stop_reason)
        _, final_ver, final_text = ch.artifacts.read("CHAIR", "report")
        self.assertEqual(final_ver, 5)  # v1 seed -> v2 Overview -> v3 BETA wipe -> v4 CHAIR revert(v2) -> v5 Audit
        self.assertIn("Stepanov hyperderivatives", final_text)
        self.assertIn("Lean TCB check", final_text)
        revert_events = [e for e in ch.events if e.type == "ARTIFACT_WRITE" and "revert(v2)" in e.detail]
        self.assertEqual(len(revert_events), 1)
        self.assertEqual(revert_events[0].status, "accepted")

    def test_section_heading_demotion_and_duplicate_section_validation(self):
        mgr = ArtifactManager(self.root / "demote_arts")
        seed = "# Monograph\n\n## [Overview]\n\n## [Analysis]\nKeep analysis untouched.\n"
        mgr.write("SYSTEM", "report", "create", seed, turn=0, attribution=False)
        mgr.read("ALPHA", "report", turn=1)

        first_body = (
            "## [Overview]\n"
            "# Internal Title\n"
            "Intro paragraph.\n"
            "## Subheading One\n"
            "First detail.\n"
            "```md\n"
            "# Inside Code Fence\n"
            "## Keep Fenced Heading Exact\n"
            "```\n"
            "### Deeper Point\n"
            "Closing thought."
        )
        mgr.write("ALPHA", "report", "replace_section", first_body, section="[Overview]", turn=1)
        _, v2, t2 = mgr.read("BETA", "report", turn=2)
        self.assertIn("## [Overview]\n\n### Internal Title\n", t2)
        self.assertIn("#### Subheading One\n", t2)
        self.assertIn("##### Deeper Point\n", t2)
        self.assertIn("# Inside Code Fence\n## Keep Fenced Heading Exact\n", t2)
        self.assertEqual(t2.count("## [Overview]"), 1)
        self.assertIn("## [Analysis]\nKeep analysis untouched.", t2)

        # Updating [Overview] a 2nd and 3rd time with internal '## Another Subheading' must NOT orphan duplicates
        mgr.write(
            "BETA", "report", "replace_section",
            "## Another Subheading\nUpdated pass two with verified citations.",
            section="[Overview]", turn=2,
        )
        mgr.read("ALPHA", "report", turn=3)
        mgr.write(
            "ALPHA", "report", "replace_section",
            "## Final Subheading\nFinal pass three content replacing pass two cleanly.",
            section="[Overview]", turn=3,
        )
        _, v4, t4 = mgr.read("ALPHA", "report", turn=4)
        self.assertEqual(v4, 4)
        self.assertEqual(t4.count("## [Overview]"), 1)
        self.assertNotIn("Another Subheading", t4)
        self.assertNotIn("Subheading One", t4)
        self.assertIn("### Final Subheading\n", t4)
        self.assertIn("## [Analysis]\nKeep analysis untouched.", t4)

        # validate_artifact must reject duplicate matching headings for a required section
        mgr_dup = ArtifactManager(self.root / "dup_val_arts")
        dup_doc = (
            "# Master\n\n"
            "## [Overview]\nFirst copy with enough body length to pass size checks.\n\n"
            "## Overview\nDuplicate second copy that should fail validation.\n"
        )
        mgr_dup.write("A", "report", "create", dup_doc, turn=1)
        mgr_dup.write("A", "report", "append", "Footer note.", turn=2)
        ok, issues, details = validate_artifact(mgr_dup, {
            "artifact": "report",
            "require_modified": True,
            "min_bytes": 50,
            "required_sections": ["[Overview]"],
        })
        self.assertFalse(ok)
        self.assertEqual(details["duplicate_sections"], ["[Overview]"])
        self.assertTrue(any("duplicate heading(s)" in iss for iss in issues))

    def test_kv_messages_stride_eviction_preserves_prefix_stability(self):
        ctx = ContextIntelligence(default_safety_margin=100)
        sys_sec = {"preamble": "Chamber system prompt."}
        fixed_sec = {"topic": "Long-horizon mathematical proof verification."}
        msgs = [
            {
                "id": f"m{i}",
                "turn": i,
                "speaker": f"AGENT_{i % 4}",
                "content": f"Turn {i} analytical derivation and theorem check. " * 3,
            }
            for i in range(1, 46)
        ]

        # Choose an effective_context budget where 35 msgs fit, so 37 omits 2 and 40 omits 5 naively;
        # stride=16 should round omitted to 16 for BOTH, keeping identical prefix [system, fixed, notice+m17..m37]!
        r37, acc37 = ctx.assemble_and_budget(
            agent_name="AGENT_1",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=msgs[:37],
            effective_context=2600,
            context_source="config",
            reserved_output=400,
            safety_margin=100,
            tail_sections={"agent_identity": "You are AGENT_1."},
            prompt_layout="kv_messages",
            kv_evict_stride=16,
        )
        r40, acc40 = ctx.assemble_and_budget(
            agent_name="AGENT_2",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=msgs[:40],
            effective_context=2600,
            context_source="config",
            reserved_output=400,
            safety_margin=100,
            tail_sections={"agent_identity": "You are AGENT_2."},
            prompt_layout="kv_messages",
            kv_evict_stride=16,
        )
        self.assertEqual(acc37.omitted_messages_count, 16)
        self.assertEqual(acc40.omitted_messages_count, 16)
        # Shared prefix of r37 (all messages except the final per-agent tail) must match r40[:len(r37)-1] verbatim
        self.assertEqual(r37[:-1], r40[: len(r37) - 1])
        self.assertIn("[16 earlier public message(s) omitted", r40[2]["content"])

    def test_anomaly_lab_detects_duplicate_artifact_section_headings(self):
        lab = AnomalyLaboratory(lab_dir=self.root / "dup_anom_out")

        # 1. Run with duplicated level-2 headings outside code fences -> detected
        bad_run = self.root / "bad_dup_run"
        (bad_run / "artifacts").mkdir(parents=True)
        (bad_run / "artifacts" / "monograph.md").write_text(
            "# Master\n\n## The Rank Wall\nFirst copy.\n\n## The Rank Wall\nOrphaned copy.\n",
            encoding="utf-8",
        )
        (bad_run / "metrics.json").write_text(json.dumps({"turns_completed": 10}), encoding="utf-8")
        (bad_run / "events.jsonl").write_text("", encoding="utf-8")
        (bad_run / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
        (bad_run / "checkpoint.json").write_text(json.dumps({
            "schema_version": 2,
            "state": {"config": {"scheduler": "self_directed", "agents": [{"name": "A"}]}},
        }), encoding="utf-8")

        found_bad = [sp for sp in lab.inspect_run(bad_run) if sp.anomaly_type == "duplicate_artifact_section_headings"]
        self.assertEqual(len(found_bad), 1)
        self.assertIn("monograph.md", found_bad[0].evidence["artifacts"])

        # 2. Run with unique level-2 headings and duplicate headings only inside ``` fences -> 0 specimens
        good_run = self.root / "good_nodup_run"
        (good_run / "artifacts").mkdir(parents=True)
        (good_run / "artifacts" / "monograph.md").write_text(
            "# Master\n\n## [Section One]\nBody.\n```md\n## Fenced\n```\n## [Section Two]\nMore.\n```md\n## Fenced\n```\n",
            encoding="utf-8",
        )
        (good_run / "metrics.json").write_text(json.dumps({"turns_completed": 10}), encoding="utf-8")
        (good_run / "events.jsonl").write_text("", encoding="utf-8")
        (good_run / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
        (good_run / "checkpoint.json").write_text(json.dumps({
            "schema_version": 2,
            "state": {"config": {"scheduler": "self_directed", "agents": [{"name": "A"}]}},
        }), encoding="utf-8")
        found_good = [sp for sp in lab.inspect_run(good_run) if sp.anomaly_type == "duplicate_artifact_section_headings"]
        self.assertEqual(found_good, [])

    def test_append_section_preserves_coauthor_content_and_demotes_headings(self):
        reply = (
            "Adding Lean TCB note.\n"
            "@APPEND_SECTION monograph | [Blueprint]: Initial text?\n"
            "## Co-Author Subheading\n"
            "Added paragraph.\n"
            "@ADD_TO_SECTION monograph | [Notes]: Extra note line."
        )
        p = parse_reply(reply, me="PROSE_BRIDGE_LEAD", names=("PROSE_BRIDGE_LEAD", "LEAN_TCB_AUDITOR"))
        self.assertEqual(p.issues, [])
        self.assertEqual(len(p.actions), 2)
        self.assertEqual(p.actions[0]["mode"], "append_section")
        self.assertEqual(p.actions[0]["section"], "[Blueprint]")
        self.assertEqual(p.actions[1]["mode"], "append_section")
        self.assertEqual(p.actions[1]["section"], "[Notes]")

        mgr = ArtifactManager(self.root / "append_sec_arts")
        mgr.write("SYSTEM", "monograph", "create", "# Monograph\n\n## [Blueprint]\n\n## [Footer]\nKeep footer.\n", turn=0, attribution=False)
        mgr.read("LEAN_TCB_AUDITOR", "monograph", turn=1)
        mgr.write(
            "LEAN_TCB_AUDITOR", "monograph", "replace_section",
            "### Kernel DAG\nCore module structure.",
            section="[Blueprint]", turn=1,
        )
        # PROSE_BRIDGE_LEAD appends additively to [Blueprint] WITHOUT having re-read after v2
        mgr.write(
            "PROSE_BRIDGE_LEAD", "monograph", "append_section",
            "## [Blueprint]\n## Prose Reading Guide\nComplementary exposition.",
            section="[Blueprint]", turn=2,
        )
        _, v3, text = mgr.read("RED_TEAM_CHAIR", "monograph", turn=3)
        self.assertEqual(v3, 3)
        self.assertEqual(text.count("## [Blueprint]"), 1)
        self.assertIn("### Kernel DAG\nCore module structure.", text)
        self.assertIn("### Prose Reading Guide\nComplementary exposition.", text)
        self.assertIn("## [Footer]\nKeep footer.", text)

    def test_sanitize_speaker_bold_peer_review_and_agent_typo_rescue(self):
        names = ("RED_TEAM_CHAIR", "PROSE_BRIDGE_LEAD", "LEAN_TCB_AUDITOR", "ERH_ARAKELOV_LEAD")
        multi_para_review = (
            "Reviewing submissions.\n\n"
            "**LEAN_TCB_AUDITOR**: The nanoda count and theorem list look solid.\n\n"
            "**PROSE_BRIDGE_LEAD**: Before locking, expand the H6 derivation scope.\n"
            "NEXT: ERH_ARAKELOW_LEAD"
        )
        p = parse_reply(multi_para_review, me="RED_TEAM_CHAIR", names=names)
        self.assertEqual(p.flags, [])
        self.assertEqual(p.issues, [])
        self.assertEqual(p.next_speaker, "ERH_ARAKELOV_LEAD")
        self.assertIn("**PROSE_BRIDGE_LEAD**: Before locking", p.message)
        self.assertIn("**LEAN_TCB_AUDITOR**: The nanoda count", p.message)

        # Unbolded label, turn-tagged label, and first-person bolded impersonation MUST still be cut
        for impersonation in (
            "Audit note.\nLEAN_TCB_AUDITOR: I confirm all axioms.",
            "Audit note.\n[m2 | turn 2] **LEAN_TCB_AUDITOR**: Looks good.",
            "Audit note.\n**LEAN_TCB_AUDITOR**: I confirm all axioms hold.",
        ):
            p_bad = parse_reply(impersonation, me="RED_TEAM_CHAIR", names=names)
            self.assertEqual(p_bad.message, "Audit note.")
            self.assertTrue(any("wrote a line as LEAN_TCB_AUDITOR" in f for f in p_bad.flags), msg=impersonation)

        # Short or genuinely unknown names are NOT fuzzy-rescued
        known_set = set(names) | {"AGENT_A", "AGENT_B"}
        self.assertEqual(canonicalize_agent_id("GHOST", known_set), "GHOST")
        self.assertEqual(canonicalize_agent_id("AGENT_A_AGENT_B", known_set), "AGENT_A_AGENT_B")
        self.assertEqual(canonicalize_agent_id("ERH_ARAKELOW_LEAD", known_set), "ERH_ARAKELOV_LEAD")

    def test_cross_tool_hints_and_blackboard_session_status(self):
        cfg = {
            "name": "cross_hint_bb_test",
            "topic": "Verify blackboard status and cross-tool hints",
            "scheduler": "round_robin",
            "board_visible": True,
            "channels": {"proof_core": {}},
            "doc_manifest": str(self.doc_dir / "manifest.json"),
            "artifacts": {"seed": {"frontier_ledger": "# Ledger\n"}, "visible": ["frontier_ledger"]},
            "limits": {"max_turns": 4},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
                {"name": "CHAIR", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            "ALPHA": [
                "Posting two claims and trying wrong commands.\n"
                "@CLAIM: Claim one\n"
                "@CLAIM: Claim two\n"
                "@READ siegel_brauer_2026 0\n"
                "@PUBLISH frontier_ledger: wrong channel\n",
                "Checking feedback.",
            ],
            "BETA": ["Challenging c2.\n@CHALLENGE c2: Check exponent"],
            "CHAIR": ["Inspecting status."],
        })
        ch = Chamber(cfg, run_dir=self.root / "cross_hint_run", backend=backend, quiet=True, copy_dir="").run()
        # CHAIR on Turn 3 sees live Blackboard status count in Session Status
        t3_tail = backend.calls[2][1]["messages"][-1]["content"]
        self.assertIn("Blackboard: 1 active, 1 disputed, 0 resolved, 0 retracted (1 challenge(s))", t3_tail)
        # ALPHA on Turn 4 sees persistent cross-tool rejection hints from Turn 1
        t4_tail = backend.calls[3][1]["messages"][-1]["content"]
        self.assertIn("use @DOC_READ siegel_brauer_2026 0 instead", t4_tail)
        self.assertIn("`frontier_ledger.md` is an artifact — use @APPEND frontier_ledger:", t4_tail)

    def test_tex_corpus_normalization_comment_skip_and_theorem_toc(self):
        self.assertEqual(DocumentLibrary.normalize_id("siegel_brauer_2026.tex"), "siegel_brauer_2026")
        tex_dir = self.root / "tex_docs"
        tex_dir.mkdir()
        (tex_dir / "sample.tex").write_text(
            "\\section{Introduction}\n"
            "% \\begin{theorem} Commented out PARI benchmark should not match search.\n"
            "% commented_secret_token\n"
            "Real prose before theorem.\n"
            "\\begin{theorem}[Theorem 1.3]\n"
            "Real bound $h_K R_K \\ge c(k) d_K^{1/2}$.\n"
            "\\end{theorem}\n",
            encoding="utf-8",
        )
        (tex_dir / "manifest.json").write_text(json.dumps({
            "documents": [{"id": "sample", "title": "Sample TeX", "path": "sample.tex"}]
        }), encoding="utf-8")
        lib = DocumentLibrary(tex_dir / "manifest.json")
        # % comments are skipped during normalized search
        _, hits_comment, _ = lib.search("sample", "commented_secret_token")
        self.assertEqual(hits_comment, [])
        # Exact raw offset of theorem body is preserved across skipped comments
        _, hits_thm, _ = lib.search("sample", "d_K")
        self.assertEqual(len(hits_thm), 1)
        self.assertIn("d_K^{1/2}", hits_thm[0]["snippet"])
        # Compact TOC includes both \section and \begin{theorem}
        _, _, toc_entries, _ = lib.table_of_contents("sample")
        titles = [e["title"] for e in toc_entries]
        self.assertTrue(any("Introduction" in t for t in titles))
        self.assertTrue(any("Theorem 1.3" in t for t in titles))

        # Verify actual repository 13-doc manifest loads with .tex sources
        real_manifest = Path(__file__).resolve().parent / "docs" / "rh_2026_deep" / "manifest.json"
        if real_manifest.exists():
            real_lib = DocumentLibrary(real_manifest)
            self.assertEqual(len(real_lib.ids()), 13)
            _, hits_sb, _ = real_lib.search("siegel_brauer_2026", "Theorem 1.3")
            self.assertGreaterEqual(len(hits_sb), 1)
            _, hits_gm, _ = real_lib.search("grenie_molteni_2022_classgroup", "4 -")
            self.assertGreaterEqual(len(hits_gm), 1)

    def test_resolve_lifecycle_rechallenge_resurrection_and_deliverable_gate(self):
        # Parsing variants: with/without colon, colons in rationale, aliases, missing rationale rejection
        p_ok = parse_reply(
            "@RESOLVE c3: Verified: denominator matches gan_ichino_2014_ggp offset 53250\n"
            "@CLOSE_DISPUTE #C4 Confirmed equation without colon separator\n"
            "@DISMISS_CHALLENGE c5 Checked lemma 2.1",
            me="RED_TEAM_CHAIR",
            names=("RED_TEAM_CHAIR", "GGP_GL2_ARCHITECT"),
        )
        self.assertEqual(p_ok.issues, [])
        self.assertEqual(len(p_ok.actions), 3)
        self.assertEqual(p_ok.actions[0]["type"], "RESOLVE")
        self.assertEqual(p_ok.actions[0]["target"], "c3")
        self.assertIn("Verified: denominator matches", p_ok.actions[0]["content"])
        self.assertEqual(p_ok.actions[1]["target"], "c4")
        self.assertEqual(p_ok.actions[2]["target"], "c5")

        p_bad = parse_reply("@RESOLVE c3", me="RED_TEAM_CHAIR", names=("RED_TEAM_CHAIR",))
        self.assertEqual(p_bad.actions, [])
        self.assertTrue(any("RESOLVE missing reason" in iss for iss in p_bad.issues))

        valid_doc = (
            "# Monograph\n\n"
            "## [Core Analysis]\n"
            + ("Detailed verified proof derivation of theta-lift divisibility and denominator bounds. " * 6)
            + "\n\n## [Red-Team Referee Report]\n"
            + ("All equations checked character-for-character against primary LaTeX sources with no remaining gaps. " * 10)
        )
        cfg = {
            "name": "resolve_gate_test",
            "topic": "Verify resolve lifecycle and deliverable dispute gate",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "board_visible": True,
            "artifacts": {
                "seed": {"monograph": "# Monograph\nSeed.\n"},
                "visible": ["monograph"],
                "validation": {
                    "artifact": "monograph",
                    "require_modified": True,
                    "min_bytes": 600,
                    "required_sections": ["[Core Analysis]", "[Red-Team Referee Report]"],
                },
            },
            "limits": {"max_turns": 10},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
                {"name": "CHAIR", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            "ALPHA": [
                # Turn 1: post c1 and complete artifact
                "Posting initial claim and writing deliverable.\n"
                "@CLAIM: Automorphic seesaw denominator controls theta-lift divisibility across modular forms\n"
                f"@REPLACE monograph: {valid_doc}",
                # Turn 4: after c1 is disputed -> uncritical mention triggers resurrection_count; vote END_RUN prematurely
                "The automorphic seesaw denominator controls modular forms completely.\n"
                "@END_RUN: Deliverable complete and all disputes settled",
                # Turn 7: resolved c1 -> mention same topic -> MUST NOT trigger resurrection!
                "Building on the resolved automorphic seesaw denominator modular forms argument.",
                # Turn 10: after second RESOLVE -> END_RUN achieves majority and succeeds!
                "Final concurrence after second resolution.\n@END_RUN: All disputes genuinely resolved",
            ],
            "BETA": [
                # Turn 2: challenge c1 (status -> disputed)
                "Challenging c1 for missing offset.\n@CHALLENGE c1: Verify denominator against primary source",
                # Turn 5: majority votes END_RUN while c1 is STILL disputed -> MUST BE BLOCKED despite artifact passing!
                "Claiming completion.\n@END_RUN: All disputes resolved",
                # Turn 8: CHAIR re-challenges c1!
                "Re-opening c1 on second check.\n@CHALLENGE c1: Exponent on prime factor needs re-check",
            ],
            "CHAIR": [
                # Turn 3: also vote END_RUN while disputed -> blocked when BETA votes, or stays blocked
                "Reviewing status.\n@END_RUN: Looks done",
                # Turn 6: CHAIR resolves c1 cleanly
                "Checked source; automorphic seesaw denominator confirmed.\n@RESOLVE c1: Confirmed verbatim at gan_ichino_2014_ggp @ 53250",
                # Turn 9: BETA re-challenged on Turn 8; BETA/ALPHA/etc: wait turn 9 mention triggers resurrection!
                "Proceeding with the automorphic seesaw denominator modular forms analysis as drafted.",
            ],
        })
        run_dir = self.root / "resolve_run"
        ch = Chamber(cfg, run_dir=run_dir, backend=backend, quiet=True, copy_dir="").run()
        # Let's check: on Turn 9 CHAIR didn't resolve c1, so Turn 10 BETA hadn't resolved -> max_turns=10 hit!
        # Verify unresolved dispute after re-challenge kept final deliverable_verified=False!
        self.assertFalse(ch.deliverable_verified)
        self.assertTrue(any("Unresolved blackboard dispute(s): c1" in iss for iss in ch.deliverable_issues))
        # On Turn 5, majority (ALPHA, BETA, CHAIR) had voted END_RUN while c1 was disputed -> MUST HAVE BEEN BLOCKED!
        self.assertTrue(any(e.type == "END_RUN" and e.status == "rejected" and "Unresolved blackboard dispute(s): c1" in e.detail for e in ch.events))
        # Turn 4 (while disputed) and Turn 9 (after re-challenge) both triggered resurrection on c1; Turn 7 (while resolved) did NOT!
        c1_res_turns = [r["turn"] for r in ch.resurrection_events if "c1" in r["source"]]
        self.assertIn(4, c1_res_turns)
        self.assertNotIn(7, c1_res_turns)
        self.assertIn(9, c1_res_turns)
        # Re-challenge preserved resolution audit trail while returning status to disputed
        self.assertEqual(ch.board["c1"].status, "disputed")
        self.assertEqual(len(ch.board["c1"].challenges), 2)
        self.assertEqual(len(ch.board["c1"].resolutions), 1)

        # Resume the saved checkpoint to test second RESOLVE -> majority END_RUN -> deliverable_verified=True & checkpoint round-trip
        ckpt = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
        # Verify checkpoint persisted BoardItem resolution history before resume
        self.assertEqual(len(ckpt["state"]["board"]["c1"]["resolutions"]), 1)
        cfg2 = dict(cfg)
        cfg2["limits"] = {"max_turns": 13}
        backend2 = MockBackend({
            "BETA": ["@RESOLVE c1: Verified prime exponent in source text\n@END_RUN: Second dispute resolved"],
            "CHAIR": ["@END_RUN: Verified and resolved"],
        })
        ch2 = Chamber(
            cfg2, run_dir=self.root / "resolve_run_cont", backend=backend2,
            quiet=True, copy_dir="", checkpoint_data=ckpt,
        ).run()
        self.assertTrue(ch2.deliverable_verified)
        self.assertTrue(ch2.end_requested)
        self.assertEqual(ch2.board["c1"].status, "resolved")
        self.assertEqual(ch2.board["c1"].resolved_by, "BETA")
        self.assertEqual(len(ch2.board["c1"].resolutions), 2)
        metrics2 = ch2.compute_metrics()
        self.assertEqual(metrics2["board_by_status"]["resolved"], 1)
        self.assertEqual(metrics2["board_by_status"]["disputed"], 0)
        events_md = (self.root / "resolve_run_cont" / "events.md").read_text(encoding="utf-8")
        self.assertIn("Resolved by `CHAIR` (turn 6)", events_md)
        self.assertIn("Resolved by `BETA` (turn 11)", events_md)

    def test_duplicate_append_prevention_and_conservative_math_allowance(self):
        mgr = ArtifactManager(self.root / "dup_arts")
        seed = (
            "# Monograph\n\n"
            "## [Spectral vs Functorial Divide]\n\n"
            "### The Spectral Constraint\n"
            "Connes extremizes the Weil quadratic form over the noncommutative adele class space so the Riemann Hypothesis holds if and only if the quadratic form is negative semidefinite on compactly supported test functions [connes_2026_letter @ 1200].\n\n"
            "## [Automorphic Core]\n\n"
            "### GGP Period Identity\n"
            "In the Gan-Ichino theorem, $\\kappa = |S_{\\phi}|$ enters as a denominator at [gan_ichino_2014_ggp @ 53250].\n"
        )
        mgr.write("SYSTEM", "mono", "create", seed, turn=0, attribution=False)

        # 1. Exact duplicate ### subheading rejected with consolidation hint
        with self.assertRaises(ArtifactError) as ctx1:
            mgr.write(
                "SPECTRAL", "mono", "append_section",
                "### The Spectral Constraint\nNew commentary on trace formula.",
                section="[Spectral vs Functorial Divide]", turn=1,
            )
        self.assertIn("already exists", str(ctx1.exception))
        self.assertIn("@SECTION mono | [Spectral vs Functorial Divide]", str(ctx1.exception))

        # 2. Renamed-heading near-identical paragraph rejected conservatively
        with self.assertRaises(ArtifactError) as ctx2:
            mgr.write(
                "SPECTRAL", "mono", "append_section",
                "### Final Spectral Synthesis\n"
                "Connes extremizes the Weil quadratic form over the noncommutative adele class space so the Riemann Hypothesis holds if and only if the quadratic form stays negative semidefinite on compactly supported test functions [connes_2026_letter @ 1200].",
                section="[Spectral vs Functorial Divide]", turn=2,
            )
        self.assertIn("append_section rejected", str(ctx2.exception))

        # 3. Same verbatim block repeated under whole-file @APPEND rejected
        mgr.write("A", "notes", "create", "# Notes\n\nInitial line.", turn=1)
        repeated_para = (
            "By Banks Theorem 1.3 [siegel_brauer_2026 @ 10550], for Galois extensions K/k we obtain "
            "an effective lower bound h_K R_K >= c(k) d_K^{1/2} / ((2*pi)^{[K:Q]} log d_K) independent of Landau-Siegel zeros."
        )
        mgr.write("A", "notes", "append", repeated_para, turn=2, attribution=False)
        with self.assertRaises(ArtifactError) as ctx3:
            mgr.write("B", "notes", "append", repeated_para, turn=3, attribution=False)
        self.assertIn("append rejected", str(ctx3.exception))

        # 4. Legitimate complementary math subsection sharing definitions & citations MUST SUCCEED
        legit_addition = (
            "### Bessel Identity & Covering Obstruction\n"
            "In contrast to GL_1 characters, the sixfold GL_5 cover introduces BesselId dependencies after "
            "$\\kappa = |S_{\\phi}|$ [gan_ichino_2014_ggp @ 53250] and shifts the Rank-2 Selmer lifting estimate "
            "documented at [adriazola_2026_dep_map @ 4100] and [siegel_brauer_2026 @ 10550]."
        )
        _, ver_ok = mgr.write(
            "GGP", "mono", "append_section", legit_addition,
            section="[Automorphic Core]", turn=4,
        )
        self.assertEqual(ver_ok, 2)

        # 5. AnomalyLab Detector #10 catches intra-section duplicates, ignores clean shared-citation sections
        lab = AnomalyLaboratory(self.root / "lab10")
        bad_run = self.root / "bad_subdup"
        (bad_run / "artifacts").mkdir(parents=True)
        (bad_run / "artifacts" / "monograph.md").write_text(
            "# Master\n\n## [Section One]\n\n### Repeated Heading\nFirst body of sufficient length to count as real text.\n\n"
            "### Repeated Heading\nSecond body.\n",
            encoding="utf-8",
        )
        (bad_run / "metrics.json").write_text(json.dumps({"turns_completed": 5}), encoding="utf-8")
        (bad_run / "events.jsonl").write_text("", encoding="utf-8")
        (bad_run / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
        (bad_run / "checkpoint.json").write_text(json.dumps({"schema_version": 2, "state": {"config": {"agents": [{"name": "A"}]}}}), encoding="utf-8")
        found_bad = [sp for sp in lab.inspect_run(bad_run) if sp.anomaly_type == "duplicate_artifact_subsection_content"]
        self.assertEqual(len(found_bad), 1)

        clean_run = self.root / "clean_subdup"
        (clean_run / "artifacts").mkdir(parents=True)
        (clean_run / "artifacts" / "monograph.md").write_text(mgr.read("GGP", "mono")[2], encoding="utf-8")
        (clean_run / "metrics.json").write_text(json.dumps({"turns_completed": 5}), encoding="utf-8")
        (clean_run / "events.jsonl").write_text("", encoding="utf-8")
        (clean_run / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
        (clean_run / "checkpoint.json").write_text(json.dumps({"schema_version": 2, "state": {"config": {"agents": [{"name": "A"}]}}}), encoding="utf-8")
        self.assertEqual([sp for sp in lab.inspect_run(clean_run) if sp.anomaly_type == "duplicate_artifact_subsection_content"], [])

    def test_escaped_newline_normalization_preserves_latex_and_code(self):
        raw = (
            "### Bridge\\n\\n"
            "To preserve $\\nu \\neq \\nabla \\notin \\mathbb{N}$, write:\n"
            "$$\\sum_{k} \\nu_k \\neq \\nabla f \\mid n$$\n"
            "And bare \\nu \\neq \\nabla outside math or \\\\n must stay intact.\\n\\n"
            "1. First item.\\n- Bullet item.\\n"
            "\\begin{equation}x \\neq y \\land \\nabla g\\end{equation}\\n\\n"
            "**Directive:** Read ``\\n\\nkeep_inline`` and fence:\n"
            "```python\ns = \"line1\\n\\nline2\"\n```\\n"
        )
        norm = normalize_escaped_newlines(raw)
        self.assertIn("### Bridge\n\nTo preserve", norm)
        self.assertIn("1. First item.\n- Bullet item.", norm)
        self.assertIn("\n\n**Directive:**", norm)
        self.assertTrue(norm.endswith("```\n"))
        # LaTeX commands & escaped backslashes & code spans must remain verbatim
        self.assertIn(r"$\nu \neq \nabla \notin \mathbb{N}$", norm)
        self.assertIn(r"$$\sum_{k} \nu_k \neq \nabla f \mid n$$", norm)
        self.assertIn(r"bare \nu \neq \nabla outside math or \\n", norm)
        self.assertIn(r"\begin{equation}x \neq y \land \nabla g\end{equation}", norm)
        self.assertIn("``\\n\\nkeep_inline``", norm)
        self.assertIn('s = "line1\\n\\nline2"', norm)

        mgr = ArtifactManager(self.root / "nl_arts")
        mgr.write("A", "doc", "create", raw, turn=1, attribution=False)
        stored = mgr.read("A", "doc")[2]
        self.assertNotIn("Bridge\\n\\n", stored)
        self.assertIn(r"\nu \neq \nabla", stored)

    def test_section_completeness_warnings_and_referee_stub_blocking(self):
        # Even when overall size >= min_bytes, a thin/stub referee section blocks validation!
        long_core = "Verified derivation of Estimate C and Banks Theorem 1.3 with full bounds. " * 90  # ~6.7 KB
        thin_exec = "Brief executive overview summarizing main results cleanly with key citations."  # ~83B (< warn_min)
        stub_ref = "The audit of the Multiplicity Engine is underway. I am checking offset 14 next."  # explicit stub + small

        spec = {
            "artifact": "monograph",
            "require_modified": True,
            "min_bytes": 5000,
            "required_sections": [
                "[Executive Overview]",
                "[Core Derivations]",
                "[Red-Team Referee Report & Open 2027 Research Problems]",
            ],
        }
        mgr = ArtifactManager(self.root / "ref_arts")
        mgr.write("SYSTEM", "monograph", "create", "# Mono\n", turn=0, attribution=False)
        mgr.read("CHAIR", "monograph", turn=1)
        mgr.write(
            "CHAIR", "monograph", "replace",
            f"# Mono\n\n## [Executive Overview]\n{thin_exec}\n\n"
            f"## [Core Derivations]\n{long_core}\n\n"
            f"## [Red-Team Referee Report & Open 2027 Research Problems]\n{stub_ref}\n",
            turn=1,
        )
        ok, issues, details = validate_artifact(mgr, spec)
        self.assertFalse(ok)
        self.assertIn("[Red-Team Referee Report & Open 2027 Research Problems]", details["stub_sections"])
        self.assertIn("[Red-Team Referee Report & Open 2027 Research Problems]", details["thin_sections"])
        self.assertTrue(any("underway" in iss for iss in issues))
        self.assertTrue(any("required referee section" in iss for iss in issues))
        self.assertTrue(any("[Executive Overview]" in w for w in details["warnings"]))

        # End-to-end Chamber test: thin warning shown in Session Status, referee stub blocks END_RUN, full referee unblocks
        cfg = {
            "name": "referee_gate_test",
            "topic": "Complete monograph including referee report",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "artifacts": {"seed": {"monograph": "# Mono\n"}, "visible": ["monograph"], "validation": spec},
            "limits": {"max_turns": 6},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full"},
                {"name": "CHAIR", "endpoint": "local", "capabilities": "full"},
            ],
        }
        full_ref = (
            "### Verbatim Source Audit\n"
            "Checked all 13 primary documents character-for-character: Banks Theorem 1.3 keeps d_K^{1/2} and explicit "
            "base-field dependence c(k) separating fixed-base Galois bounds from general non-Galois extensions; "
            "Grenie-Molteni Theorem 1.1 keeps (4 - 1/(2n)) log^2 Delta strictly conditional on Dedekind ERH while "
            "quadratic/abelian specializations fall under Q-GRH; Gan-Ichino seesaw denominator |S_phi| and LinkDIV "
            "dependencies pinpoint the GL_2 rank-jump barrier; Connes Weil quadratic form stays strictly separated from "
            "theta-lift divisibility transport; Lean TCB hashes and 1,484,729 nanoda declaration counts match Waterloo "
            "audit logs; no unverified citations remain. Open 2027 problems cover non-Galois Siegel mass bounds and "
            "explicit sub-linear lift constants."
        )
        backend = MockBackend({
            "A": [
                "@REPLACE monograph: "
                f"# Mono\n\n## [Executive Overview]\n{thin_exec}\n\n"
                f"## [Core Derivations]\n{long_core}\n\n"
                "## [Red-Team Referee Report & Open 2027 Research Problems]\n"
                "Short non-stub referee summary still way too brief.\n@END_RUN: Done",
                "@END_RUN: Re-affirming after referee completion",
            ],
            "CHAIR": [
                # Turn 2: majority vote with brief referee section MUST fail validation!
                "Voting done.\n@END_RUN: Accept",
                # Turn 4: after expanding referee section -> passes validation and concludes cleanly!
                f"Completed referee section.\n@SECTION monograph | [Red-Team Referee Report & Open 2027 Research Problems]: {full_ref}\n@END_RUN: Verified",
            ],
        })
        ch = Chamber(cfg, run_dir=self.root / "ref_gate_run", backend=backend, quiet=True, copy_dir="").run()
        # CHAIR Turn 2 sees thin-section warning in Session Status
        t2_tail = backend.calls[1][1]["messages"][-1]["content"]
        self.assertIn("Warning:", t2_tail)
        self.assertIn("[Executive Overview]", t2_tail)
        # Turn 2 END_RUN was blocked by referee floor; Turn 4 verified cleanly!
        self.assertTrue(any(e.type == "END_RUN" and e.status == "rejected" and "required referee section" in e.detail for e in ch.events))
        self.assertTrue(ch.deliverable_verified)
        self.assertEqual(ch.turn, 4)

    def test_v5_v6_historical_regression_checks(self):
        repo_root = Path(__file__).resolve().parent
        v5_dir = repo_root / "runs" / "rh_2026_deep_proof_v5_8a_long_cont"
        v6_dir = repo_root / "runs" / "rh_2026_deep_proof_v6_8a_long"
        lab = AnomalyLaboratory(self.root / "hist_lab")

        if not (v5_dir.exists() and v6_dir.exists()):
            self.skipTest("Historical v5/v6 run directories not present")

        # v5 monograph and debate_notes remain free of duplicates; Detector #10 catches the 3 real repeated table blocks in v5 frontier_ledger.md
        v5_specs = lab.inspect_run(v5_dir)
        self.assertEqual([s.anomaly_type for s in v5_specs], ["duplicate_artifact_subsection_content"])
        v5_dup_arts = v5_specs[0].evidence["artifacts"]
        self.assertEqual(sorted(v5_dup_arts.keys()), ["frontier_ledger.md"])
        self.assertEqual(v5_dup_arts["frontier_ledger.md"]["[document]"]["repeated_blocks"], 3)

        # v6 triggers Detector #10 across its duplicated monograph sections (plus ledger/notes)
        v6_specs = lab.inspect_run(v6_dir)
        v6_subdup = [s for s in v6_specs if s.anomaly_type == "duplicate_artifact_subsection_content"]
        self.assertEqual(len(v6_subdup), 1)
        v6_mono_dups = v6_subdup[0].evidence["artifacts"].get("rh_2026_post_rh_frontier_monograph.md", {})
        self.assertGreaterEqual(len(v6_mono_dups), 6)

        # v6 monograph copied into temp ArtifactManager fails v6's spec due to Section 11 referee stub
        v6_mono = (v6_dir / "artifacts" / "rh_2026_post_rh_frontier_monograph.md").read_text(encoding="utf-8")
        v6_cfg = json.loads((repo_root / "experiments" / "rh_2026_deep_proof_v6_8a_long.json").read_text(encoding="utf-8"))
        mgr = ArtifactManager(self.root / "v6_check", max_bytes=1048576, max_total_bytes=4194304)
        mgr.write("SYSTEM", "rh_2026_post_rh_frontier_monograph", "create", "# Seed\n", turn=0, attribution=False)
        mgr.read("A", "rh_2026_post_rh_frontier_monograph", turn=1)
        mgr.write("A", "rh_2026_post_rh_frontier_monograph", "replace", v6_mono, turn=1, force=True)
        ok_v6, issues_v6, det_v6 = validate_artifact(mgr, v6_cfg["artifacts"]["validation"])
        self.assertFalse(ok_v6)
        self.assertIn("[Red-Team Referee Report & Open 2027 Research Problems]", det_v6["stub_sections"])
        self.assertIn("[Red-Team Referee Report & Open 2027 Research Problems]", det_v6["thin_sections"])

        # Escaped-newline normalizer eliminates v6's literal \n\n leaks while leaving math untouched
        fixed_v6 = normalize_escaped_newlines(v6_mono)
        self.assertNotIn("\\n\\n", fixed_v6)
        self.assertIn(r"\mathcal{C}_{lift}", fixed_v6)

    def test_api_prefix_caching_and_openrouter_sticky_routing(self):
        # 1. Provider-specific sticky session / cache keys in build_payload
        msgs_dummy = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}]
        or_ep = {"provider": "openrouter", "model": "deepseek/deepseek-chat", "base_url": "https://openrouter.ai/api/v1"}
        oa_ep = {"provider": "openai", "model": "gpt-5", "base_url": "https://api.openai.com/v1"}
        lc_ep = {"provider": "llamacpp", "model": "gemma-4", "base_url": "http://127.0.0.1:8080/v1"}

        p_or, _ = build_payload(or_ep, {}, msgs_dummy, 0.7, 512, session_id="chamber-or-test")
        p_oa, _ = build_payload(oa_ep, {}, msgs_dummy, 0.7, 512, session_id="chamber-oa-test")
        p_lc, _ = build_payload(lc_ep, {}, msgs_dummy, 0.7, 512, session_id="chamber-lc-test")
        self.assertEqual(p_or.get("session_id"), "chamber-or-test")
        self.assertNotIn("prompt_cache_key", p_or)
        self.assertEqual(p_oa.get("prompt_cache_key"), "chamber-oa-test")
        self.assertNotIn("session_id", p_oa)
        self.assertNotIn("session_id", p_lc)
        self.assertNotIn("prompt_cache_key", p_lc)

        # 2. HttpBackend.complete sends x-session-id header & parses cache hit fallbacks
        captured = {}

        class _FakeHTTPResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        def _make_urlopen(resp_obj):
            def _fake_urlopen(req, timeout=120):
                captured["headers"] = {k.lower(): v for k, v in req.header_items()}
                captured["body"] = json.loads(req.data.decode("utf-8"))
                return _FakeHTTPResp(json.dumps(resp_obj).encode("utf-8"))
            return _fake_urlopen

        backend_http = HttpBackend(allow_paid=True)
        with patch.dict("os.environ", {"TEST_OR_KEY": "sk-test-secret"}), patch(
            "urllib.request.urlopen",
            side_effect=_make_urlopen({
                "model": "deepseek/deepseek-chat",
                "choices": [{"message": {"content": "Reply"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 50, "prompt_cache_hit_tokens": 1800},
            }),
        ):
            res_ds = backend_http.complete(
                "ALPHA",
                {**or_ep, "api_key_env": "TEST_OR_KEY"},
                p_or,
            )
        self.assertEqual(captured["headers"].get("x-session-id"), "chamber-or-test")
        self.assertEqual(captured["body"].get("session_id"), "chamber-or-test")
        self.assertEqual(res_ds.cached_tokens, 1800)
        self.assertEqual(res_ds.cache_source, "usage.prompt_cache_hit_tokens")

        with patch.dict("os.environ", {"TEST_OR_KEY": "sk-test-secret"}), patch(
            "urllib.request.urlopen",
            side_effect=_make_urlopen({
                "model": "anthropic/claude-sonnet",
                "choices": [{"message": {"content": "Reply"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 50, "cache_read_input_tokens": 1600},
            }),
        ):
            res_anth = backend_http.complete(
                "ALPHA",
                {**or_ep, "api_key_env": "TEST_OR_KEY"},
                p_or,
            )
        self.assertEqual(res_anth.cached_tokens, 1600)
        self.assertEqual(res_anth.cache_source, "usage.cache_read_input_tokens")

        # 3. ContextIntelligence api_prefix & auto(openrouter) consolidates static system+fixed prefix + stride eviction
        ctx = ContextIntelligence(default_safety_margin=100)
        sys_sec = {"preamble": "Chamber invariant rules.", "protocol": "Commands go at bottom."}
        fixed_sec = {"topic": "Topic: Dedekind ERH.", "doc_manifest": "Docs: manifest entries."}
        tail_a = {"agent_identity": "You are ALPHA.", "scratchpad": "ALPHA note."}
        tail_b = {"agent_identity": "You are BETA.", "scratchpad": "BETA note."}

        t1, _ = ctx.assemble_and_budget(
            agent_name="ALPHA",
            endpoint={"provider": "openrouter"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_a,
            prompt_layout="auto",
        )
        t2, _ = ctx.assemble_and_budget(
            agent_name="BETA",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[{"id": "m1", "turn": 1, "speaker": "ALPHA", "content": "Initial extraction."}],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_b,
            prompt_layout="api_prefix",
        )
        self.assertEqual([m["role"] for m in t1], ["system", "user"])
        self.assertEqual([m["role"] for m in t2], ["system", "user"])
        self.assertEqual(t1[0]["content"], t2[0]["content"])
        self.assertIn("Chamber invariant rules.", t1[0]["content"])
        self.assertIn("Topic: Dedekind ERH.", t1[0]["content"])
        self.assertTrue(t2[1]["content"].startswith("--- Public Conversation ---\n[m1 | turn 1] ALPHA: Initial extraction."))

        many_msgs = [
            {"id": f"m{i}", "turn": i, "speaker": "ALPHA" if i % 2 else "BETA", "content": f"Msg #{i} " + ("weil form " * 18)}
            for i in range(1, 30)
        ]
        trim_28, acc_28 = ctx.assemble_and_budget(
            agent_name="ALPHA",
            endpoint={"provider": "openrouter"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=many_msgs[:-1],
            effective_context=1200,
            context_source="config",
            reserved_output=400,
            safety_margin=100,
            tail_sections=tail_a,
            prompt_layout="auto",
            kv_evict_stride=8,
        )
        trim_29, acc_29 = ctx.assemble_and_budget(
            agent_name="BETA",
            endpoint={"provider": "openrouter"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=many_msgs,
            effective_context=1200,
            context_source="config",
            reserved_output=400,
            safety_margin=100,
            tail_sections=tail_b,
            prompt_layout="auto",
            kv_evict_stride=8,
        )
        self.assertEqual(acc_28.omitted_messages_count % 8, 0)
        if acc_28.omitted_messages_count == acc_29.omitted_messages_count:
            # Shared conversation prefix stays identical before tail when stride does not advance
            pub_28 = trim_28[1]["content"].split("\n\nYou are ")[0]
            self.assertTrue(trim_29[1]["content"].startswith(pub_28))

        # 4. End-to-end Chamber with OpenRouter endpoint & prompt_layout="auto"
        cfg = {
            "name": "openrouter_prefix_cache_e2e",
            "run_id": "or_prefix_run_01",
            "topic": "Verify API prefix caching & session_id.",
            "scheduler": "round_robin",
            "prompt_layout": "auto",
            "endpoints": {
                "hermes": {
                    "provider": "openrouter",
                    "remote": False,
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "deepseek/deepseek-chat",
                }
            },
            "limits": {"max_turns": 2},
            "agents": [
                {"name": "ALPHA", "endpoint": "hermes", "capabilities": "full", "identity": "You are ALPHA."},
                {"name": "BETA", "endpoint": "hermes", "capabilities": "full", "identity": "You are BETA."},
            ],
        }
        mock_be = MockBackend({"ALPHA": ["Opening theorem statement."], "BETA": ["Confirmed theorem statement."]})
        ch = Chamber(cfg, run_dir=self.root / "or_prefix_e2e", backend=mock_be, quiet=True, copy_dir="").run()
        self.assertEqual(len(mock_be.calls), 2)
        p1, p2 = mock_be.calls[0][1], mock_be.calls[1][1]
        self.assertEqual(p1.get("session_id"), "chamber-or_prefix_run_01")
        self.assertEqual(p2.get("session_id"), "chamber-or_prefix_run_01")
        self.assertEqual([m["role"] for m in p1["messages"]], ["system", "user"])
        self.assertEqual([m["role"] for m in p2["messages"]], ["system", "user"])
        self.assertEqual(p1["messages"][0]["content"], p2["messages"][0]["content"])
        self.assertIn("Topic: Verify API prefix caching", p1["messages"][0]["content"])
        self.assertEqual(ch.compute_metrics()["run_id"], "or_prefix_run_01")

    def test_stop_sequences_exclude_peer_names(self):
        cfg = {
            "name": "stops_test",
            "topic": "Verify stops exclude peer names.",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "PROSE_BRIDGE_LEAD", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            "ALPHA": ["Line one.\nPROSE_BRIDGE_LEAD: please verify Lemma 2.1.\n@NOTE: follow up"],
            "PROSE_BRIDGE_LEAD": ["Verified Lemma 2.1 cleanly."],
        })
        ch = Chamber(cfg, run_dir=self.root / "stops_run", backend=backend, quiet=True, copy_dir="").run()
        stops = backend.calls[0][1].get("stop") or []
        self.assertNotIn("\nPROSE_BRIDGE_LEAD:", stops)
        self.assertNotIn("\nALPHA:", stops)
        self.assertIn("\n--- Public Conversation", stops)
        self.assertIn("\n--- Shared Blackboard", stops)
        self.assertIn("\n--- Session Status", stops)
        self.assertIn("\n--- Workspace Artifacts", stops)
        self.assertEqual(ch.agents["ALPHA"].note, "follow up")

    def test_truncated_trailing_artifact_write_guard(self):
        cfg = {
            "name": "trunc_guard_test",
            "topic": "Reject truncated trailing artifact writes while preserving earlier valid actions.",
            "scheduler": "round_robin",
            "artifacts": {
                "seed": {"notes": "# Notes\n\nSeed line.\n"},
                "visible": ["notes"],
            },
            "limits": {"max_turns": 4},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
            ],
        }
        mid_word = "This theorem statement keeps rolling across lines until it gets sliced mid sentence without any closing mark " * 2
        backend = MockBackend({
            "ALPHA": [
                Result(
                    content=(
                        "Logging note first, then starting append that hits length.\n"
                        "@NOTE: preserved note before truncated write\n"
                        "@APPEND notes: Extracted bound [connes_"
                    ),
                    finish_reason="length",
                    prompt_tokens=100,
                    completion_tokens=400,
                ),
                Result(
                    content="@APPEND notes: Complete interval $x \\in [0, 1)$ verified cleanly.",
                    finish_reason="stop",
                    prompt_tokens=120,
                    completion_tokens=60,
                ),
            ],
            "BETA": [
                Result(
                    content=(
                        "Second turn tests unclosed fence and mid-word ending.\n"
                        "@APPEND notes: ```python\nx = 1"
                    ),
                    finish_reason="length",
                    prompt_tokens=110,
                    completion_tokens=300,
                ),
                Result(
                    content=f"@APPEND notes: {mid_word.rstrip()}",
                    finish_reason="length",
                    prompt_tokens=120,
                    completion_tokens=300,
                ),
            ],
        })
        ch = Chamber(cfg, run_dir=self.root / "trunc_run", backend=backend, quiet=True, copy_dir="").run()
        # Earlier valid @NOTE executed on Turn 1; all three truncated trailing writes were rejected!
        self.assertEqual(ch.agents["ALPHA"].note, "preserved note before truncated write")
        rejected_writes = [
            e for e in ch.events
            if e.type == "ARTIFACT_WRITE" and e.status == "rejected" and "truncated trailing write rejected" in e.detail
        ]
        self.assertEqual(len(rejected_writes), 3)
        # Only Turn 3 complete finish_reason="stop" write (even with [0, 1)) succeeded (notes goes v1 -> v2)
        self.assertEqual(ch.artifacts.versions["notes.md"], 2)
        text = ch.artifacts.read("BETA", "notes")[2]
        self.assertNotIn("[connes_", text)
        self.assertIn("x \\in [0, 1)", text)

    def test_action_divider_and_double_attribution_dedup(self):
        # 1. Standalone --- actions --- divider closes preceding multiline append & is stripped from speech
        p = parse_reply(
            "Opening discussion.\n\n---\n\nKept divider rule above.\n\n--- actions ---\n"
            "@APPEND notes: First line of note\nSecond line of note\n"
            "=== ACTION ===\n"
            "@NOTE: clean scratchpad\n"
            "NEXT: BETA",
            me="ALPHA",
            names=("ALPHA", "BETA"),
        )
        self.assertEqual(p.issues, [])
        self.assertIn("---", p.message)
        self.assertNotIn("actions ---", p.message)
        self.assertEqual(p.next_speaker, "BETA")
        self.assertEqual([a["type"] for a in p.actions], ["ARTIFACT_WRITE", "UPDATE_NOTE"])
        self.assertEqual(p.actions[0]["content"], "First line of note\nSecond line of note")
        self.assertEqual(p.actions[1]["content"], "clean scratchpad")

        # 2. @APPEND double-attribution dedup
        mgr = ArtifactManager(self.root / "attrib_arts")
        mgr.write("ALPHA", "ledger", "create", "# Ledger\n\nInitial row.", turn=0, attribution=False)
        mgr.write(
            "ALPHA",
            "ledger",
            "append",
            "| A | B | C | [doc @ 10] |\n\n— *ALPHA, turn 4*",
            turn=4,
            attribution=True,
        )
        stored = mgr.read("ALPHA", "ledger")[2]
        self.assertEqual(stored.count("— *ALPHA, turn 4*"), 1)

    def test_section_owners_smallest_sections_resolve_hint_and_run_id(self):
        long_a = "Verified primary proof derivation of H0-H12 and Estimate C bounds. " * 10
        peer_only_b = "Peer commentary on sixfold covering obstruction [gan_ichino_2014_ggp @ 53250]."
        ref_ok = (
            "### Audit Findings\n"
            "Verified primary LaTeX theorems, kernel hashes, Banks Theorem 1.3 exponents, and Grenie-Molteni bounds."
        )
        spec_warn = {
            "artifact": "monograph",
            "require_modified": True,
            "min_bytes": 500,
            "section_owners": {
                "[Section Alpha]": "ALPHA",
                "[Section Beta]": "BETA",
                "[Red-Team Referee Report]": "CHAIR",
            },
            "enforce_section_owners": False,
            "required_sections": ["[Section Alpha]", "[Section Beta]", "[Red-Team Referee Report]"],
        }
        mgr = ArtifactManager(self.root / "owner_arts")
        mgr.write(
            "SYSTEM", "monograph", "create",
            "# Mono\n\n## [Section Alpha]\n\n## [Section Beta]\n\n## [Red-Team Referee Report]\n",
            turn=0, attribution=False,
        )
        mgr.read("ALPHA", "monograph", turn=1)
        mgr.write("ALPHA", "monograph", "replace_section", long_a, section="[Section Alpha]", turn=1)
        # Only ALPHA wrote into [Section Beta] via append_section (designated owner BETA has not authored it!)
        mgr.write("ALPHA", "monograph", "append_section", peer_only_b, section="[Section Beta]", turn=1)
        mgr.read("CHAIR", "monograph", turn=1)
        mgr.write("CHAIR", "monograph", "replace_section", ref_ok, section="[Red-Team Referee Report]", turn=1)

        ok_w, _, det_w = validate_artifact(mgr, spec_warn)
        self.assertTrue(ok_w)
        self.assertEqual(det_w["unauthored_by_owner"], {"[Section Beta]": "BETA"})
        self.assertTrue(any("not yet been authored by designated owner BETA" in w for w in det_w["warnings"]))

        spec_enforce = dict(spec_warn, enforce_section_owners=True)
        ok_e, issues_e, det_e = validate_artifact(mgr, spec_enforce)
        self.assertFalse(ok_e)
        self.assertEqual(det_e["unauthored_by_owner"], {"[Section Beta]": "BETA"})
        self.assertTrue(any("has not been authored by designated owner BETA" in iss for iss in issues_e))

        # Once BETA authors [Section Beta] via @SECTION, enforcement passes!
        mgr.read("BETA", "monograph", turn=2)
        mgr.write(
            "BETA", "monograph", "replace_section",
            "Authoritative Gan-Ichino GGP derivation by owner BETA.\n\n" + peer_only_b,
            section="[Section Beta]", turn=2,
        )
        ok_e2, issues_e2, det_e2 = validate_artifact(mgr, spec_enforce)
        self.assertTrue(ok_e2, msg=issues_e2)
        self.assertEqual(det_e2["unauthored_by_owner"], {})

        # End-to-end Chamber: Smallest sections in Session Status, @RESOLVE hint on active claim, and run_id in metrics
        cfg = {
            "name": "owners_and_ux_e2e",
            "run_id": "ux_metrics_run_08",
            "topic": "Test smallest sections summary, resolve UX hint, and run_id.",
            "scheduler": "round_robin",
            "board_visible": True,
            "artifacts": {
                "seed": {
                    "monograph": "# Mono\n\n## [Section Alpha]\n\n## [Section Beta]\n\n## [Red-Team Referee Report]\n",
                },
                "visible": ["monograph"],
                "validation": spec_enforce,
            },
            "limits": {"max_turns": 3},
            "agents": [
                {"name": "ALPHA", "endpoint": "local", "capabilities": "full"},
                {"name": "BETA", "endpoint": "local", "capabilities": "full"},
                {"name": "CHAIR", "endpoint": "local", "capabilities": "full"},
            ],
        }
        e2e_backend = MockBackend({
            "ALPHA": [
                "@CLAIM: Banks effective bound includes d_K^{1/2}\n"
                f"@SECTION monograph | [Section Alpha]: {long_a}"
            ],
            "BETA": [
                "@RESOLVE c1: Calling resolve on active claim\n"
                f"@SECTION monograph | [Section Beta]: Authoritative owner derivation. {peer_only_b}"
            ],
            "CHAIR": [f"@SECTION monograph | [Red-Team Referee Report]: {ref_ok}"],
        })
        run_dir = self.root / "ux_e2e_run"
        ch = Chamber(cfg, run_dir=run_dir, backend=e2e_backend, quiet=True, copy_dir="").run()
        # CHAIR Turn 3 Session Status saw Smallest sections summary (>= 3 sections in deliverable)
        t3_tail = e2e_backend.calls[2][1]["messages"][-1]["content"]
        self.assertIn("Smallest sections:", t3_tail)
        # BETA Turn 2 @RESOLVE on active c1 gave clear @VOTE hint
        self.assertTrue(any(
            e.type == "RESOLVE" and e.status == "rejected" and "use @VOTE c1 yes|no for active claims" in e.detail
            for e in ch.events
        ))
        # run_id present in compute_metrics() and persisted metrics.json
        metrics_json = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics_json["run_id"], "ux_metrics_run_08")

    def test_cross_section_nesting_guard_and_shallowest_span_and_no_newline_bloat(self):
        # 1. Cross-section heading collision guard prevents nesting a sibling section heading
        mgr = ArtifactManager(self.root / "nest_arts")
        seed = (
            "# Monograph\n\n"
            "## [Executive Overview]\nInitial overview.\n\n"
            "## [Red-Team Referee Report]\nInitial referee.\n"
        )
        mgr.write("SYSTEM", "monograph", "create", seed, turn=0, attribution=False)
        mgr.read("CHAIR", "monograph", turn=1)

        with self.assertRaises(ArtifactError) as ctx_nest:
            mgr.write(
                "CHAIR", "monograph", "replace_section",
                "### Synthesis\nOverview body.\n\n## [Red-Team Referee Report]\nEmbedded referee text.",
                section="[Executive Overview]", turn=1,
            )
        self.assertIn("embeds heading", str(ctx_nest.exception))
        self.assertIn("own @SECTION command", str(ctx_nest.exception))

        with self.assertRaises(ArtifactError) as ctx_app_nest:
            mgr.write(
                "CHAIR", "monograph", "append_section",
                "### [Red-Team Referee Report]\nEmbedded inside append_section.",
                section="[Executive Overview]", turn=1,
            )
        self.assertIn("embeds heading", str(ctx_app_nest.exception))

        # 2. Repeated replace_section edits on an earlier section must NOT accumulate trailing blank lines (+1B bloat)
        for t in range(1, 7):
            mgr.read("CHAIR", "monograph", turn=t)
            mgr.write(
                "CHAIR", "monograph", "replace_section",
                f"### Synthesis Pass {t}\nClean overview text without trailing blank line drift.",
                section="[Executive Overview]", turn=t,
            )
        final_text = mgr.read("CHAIR", "monograph", turn=7)[2]
        self.assertTrue(final_text.endswith("\n"))
        self.assertFalse(final_text.endswith("\n\n"))

        # 3. find_section_span selects shallowest non-fenced heading and ignores fenced code headings
        tricky = (
            "# Master\n\n"
            "## [Overview]\n"
            "```md\n"
            "## [Referee]\n"
            "Fenced fake heading.\n"
            "```\n\n"
            "### [Referee]\n"
            "Nested deeper fake.\n\n"
            "## [Referee]\n"
            "Real top-level referee body.\n"
        )
        span = find_section_span(tricky, "[Referee]")
        self.assertIsNotNone(span)
        hs, he, be, lvl = span
        self.assertEqual(lvl, 2)
        self.assertEqual(tricky[hs:he], "## [Referee]")
        self.assertEqual(tricky[he:be].strip(), "Real top-level referee body.")

    def test_replace_section_anti_wipe_meta_stub_and_same_turn_guard(self):
        mgr = ArtifactManager(self.root / "wipe_arts")
        long_sec = (
            "### Detailed Automorphic Core\n"
            + ("Verified Gan-Ichino seesaw identity and LinkDIV dependency chain across unitary groups. " * 14)
        )
        mgr.write(
            "SYSTEM", "monograph", "create",
            f"# Mono\n\n## [Automorphic Core]\n\n{long_sec}\n\n## [Other Section]\nKept safe.\n",
            turn=0, attribution=False,
        )
        mgr.read("GGP", "monograph", turn=1)

        # 1. Meta-stub replacement on a substantial section is rejected even with min_replace_ratio=0
        with self.assertRaises(ArtifactError) as ctx_meta:
            mgr.write(
                "GGP", "monograph", "replace_section",
                "(This section is replaced with the high-density content above to resolve stubs and duplicate headings).",
                section="[Automorphic Core]", turn=1,
            )
        self.assertIn("placeholder/meta-stub", str(ctx_meta.exception))

        # 2. Shrinking a >= 1000B section below 25% is rejected when min_replace_ratio=0.25
        with self.assertRaises(ArtifactError) as ctx_shrink:
            mgr.write(
                "GGP", "monograph", "replace_section",
                "Short replacement losing 90% of verified formulas.",
                section="[Automorphic Core]", turn=1, min_replace_ratio=0.25,
            )
        self.assertIn("would shrink [Automorphic Core]", str(ctx_shrink.exception))

        # 3. Legitimate tightening (~60% of size) succeeds
        tight_sec = (
            "### Consolidated Automorphic Core\n"
            + ("Verified Gan-Ichino seesaw identity and LinkDIV dependency chain across unitary groups. " * 9)
        )
        _, v2 = mgr.write(
            "GGP", "monograph", "replace_section",
            tight_sec, section="[Automorphic Core]", turn=1, min_replace_ratio=0.25,
        )
        self.assertEqual(v2, 2)

        # 4. End-to-end Chamber same-turn duplicate @SECTION guard (mimics v8 Turn 199 failure mode)
        cfg = {
            "name": "same_turn_sec_guard",
            "topic": "Protect same-turn section write against duplicate trailing overwrite.",
            "scheduler": "round_robin",
            "artifacts": {
                "seed": {"monograph": "# Mono\n\n## [Automorphic Core]\n\n## [Higher Rank]\n"},
                "visible": ["monograph"],
            },
            "limits": {"max_turns": 1},
            "agents": [{"name": "GGP", "endpoint": "local", "capabilities": "full"}],
        }
        good_auto = "Full seesaw identity and LinkDIV chain derivation. " * 12
        good_hi = "Rank-2 modular obstruction analysis and BSD separation. " * 10
        backend = MockBackend({
            "GGP": [
                f"@SECTION monograph | [Automorphic Core]: {good_auto}\n"
                f"@SECTION monograph | [Higher Rank]: {good_hi}\n"
                "@SECTION monograph | [Automorphic Core]: Consolidated second overwrite in same turn."
            ]
        })
        ch = Chamber(cfg, run_dir=self.root / "same_turn_sec_run", backend=backend, quiet=True, copy_dir="").run()
        stored = ch.artifacts.read("GGP", "monograph")[2]
        self.assertIn("Full seesaw identity", stored)
        self.assertIn("Rank-2 modular obstruction", stored)
        self.assertNotIn("Consolidated second overwrite", stored)
        self.assertTrue(any(
            e.type == "ARTIFACT_WRITE" and e.status == "rejected" and "duplicate @SECTION" in e.detail
            for e in ch.events
        ))

    def test_per_artifact_preview_chars_config(self):
        long_mono = "# Mono\n\n## [Analysis]\n" + ("Verified proof derivation step. " * 80)
        long_notes = "# Notes\n\n" + ("Raw scratch extraction note from LaTeX. " * 100)
        cfg = {
            "name": "per_art_preview_test",
            "topic": "Per-artifact preview limits keep deliverable full while trimming notes.",
            "scheduler": "round_robin",
            "artifacts": {
                "seed": {"monograph": long_mono, "debate_notes": long_notes},
                "visible": ["monograph", "debate_notes"],
                "preview_chars": {"monograph": 20000, "debate_notes": 600},
            },
            "limits": {"max_turns": 1, "artifact_preview_chars": 20000},
            "agents": [{"name": "ALPHA", "endpoint": "local", "capabilities": "full"}],
        }
        backend = MockBackend({"ALPHA": ["Inspected both artifacts cleanly."]})
        Chamber(cfg, run_dir=self.root / "preview_run", backend=backend, quiet=True, copy_dir="").run()
        prompt_tail = backend.calls[0][1]["messages"][-1]["content"]
        # monograph (~2.6KB < 20KB) appears verbatim in full; debate_notes (~4.1KB > 600B) is truncated!
        self.assertIn(long_mono.strip(), prompt_tail)
        self.assertNotIn("[Middle omitted; use @READ monograph", prompt_tail)
        self.assertIn("[Middle omitted; use @READ debate_notes OFFSET", prompt_tail)


if __name__ == "__main__":
    unittest.main()
