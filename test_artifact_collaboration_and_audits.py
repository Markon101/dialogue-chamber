#!/usr/bin/env python3
"""Regression tests for section-level artifact collaboration, replace guards, citation audits, and scheduling hygiene."""
import json
import tempfile
import unittest
from pathlib import Path

from anomaly_lab import AnomalyLaboratory
from artifacts import ArtifactError, ArtifactManager, find_section_span, validate_artifact
from chamber import Chamber
from context import ContextIntelligence
from documents import DocumentLibrary
from inference import MockBackend
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
        self.assertIn("Blackboard: 1 active, 1 disputed, 0 retracted (1 challenge(s))", t3_tail)
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


if __name__ == "__main__":
    unittest.main()
