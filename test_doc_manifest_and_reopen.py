#!/usr/bin/env python3
"""Tests for Reference Document Manifest / Library and Reopened Continuation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chamber import Chamber, main
from documents import DocumentError, DocumentLibrary
from inference import DemoBackend, MockBackend
from protocol import parse_reply


class DocManifestAndReopenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir()
        (self.docs_dir / "paper1.txt").write_text(
            "Introduction to Chained Theta Lifts.\n"
            "Theorem 1.2: For a von Mangoldt sum of length X, the chain of theta lifts between "
            "unitary groups U(X_0) -> ... -> U(X_m) scales as m ~ sqrt(X).\n"
            "Remark: ERH is not yet formalized.\n",
            encoding="utf-8",
        )
        (self.docs_dir / "paper2.txt").write_text(
            "CONDITIONAL BOUNDS FOR THE LEAST QUADRATIC\n"
            "NON-RESIDUE AND RELATED PROBLEMS.\n"
            "Theorem 1 (Lamzouri-Li-Soundararajan): Assuming GRH, the least quadratic non-residue "
            "satisfies n_p < (ln p)^2 for all primes p >= 5.\n"
            "Explicit constants for L(1, chi) follow via Littlewood bounds.\n",
            encoding="utf-8",
        )
        (self.docs_dir / "paper3.txt").write_text(
            "Breaking the 4 barrier for class group generators.\n"
            "Under GRH the generating bound improves to (4 - 1/(2n)) log 2 ∆.\n"
            "No cyclic hypothesis is needed for the conductor bound.\n",
            encoding="utf-8",
        )
        self.manifest_path = self.docs_dir / "manifest.json"
        self.manifest_path.write_text(
            json.dumps({
                "documents": [
                    {
                        "id": "alpoge_2026_remark",
                        "title": "Remark on RH & GRH Proof",
                        "authors": "Levent Alpoge",
                        "year": "2026",
                        "summary": "Iterated unitary theta lifts of length sqrt(X).",
                        "tags": ["theta-lifts", "lean4"],
                        "path": "paper1.txt",
                    },
                    {
                        "id": "lamzouri_2013_grh",
                        "title": "Conditional bounds for least quadratic non-residue",
                        "authors": "Lamzouri, Li, Soundararajan",
                        "year": "2013",
                        "summary": "Explicit GRH bounds n_p < (ln p)^2.",
                        "tags": ["grh", "algorithms"],
                        "path": "paper2.txt",
                    },
                    {
                        "id": "grenie_molteni_2022_classgroup",
                        "title": "Breaking the 4 barrier",
                        "authors": "Grenie, Molteni",
                        "year": "2025",
                        "summary": "Class group bound below 4 log^2 Delta.",
                        "tags": ["class-group", "grh"],
                        "path": "paper3.txt",
                    },
                ]
            }),
            encoding="utf-8",
        )
        self.addCleanup(patch.stopall)
        patch("chamber.COPY_DIR", "").start()

    def test_document_library_manifest_read_search_and_errors(self):
        lib = DocumentLibrary(manifest_path=self.manifest_path)
        self.assertTrue(lib.has_docs())
        self.assertEqual(lib.ids(), ["alpoge_2026_remark", "lamzouri_2013_grh", "grenie_molteni_2022_classgroup"])
        manifest_txt = lib.render_manifest()
        self.assertIn("alpoge_2026_remark", manifest_txt)
        self.assertIn("lamzouri_2013_grh", manifest_txt)

        doc_id, title, excerpt, off, total = lib.read("alpoge_2026_remark", offset=0, max_chars=500)
        self.assertEqual(doc_id, "alpoge_2026_remark")
        self.assertEqual(off, 0)
        self.assertGreater(total, 100)
        self.assertIn("sqrt(X)", excerpt)

        scope, hits, formatted = lib.search("all", "Lamzouri-Li-Soundararajan")
        self.assertEqual(scope, "all")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["doc_id"], "lamzouri_2013_grh")
        self.assertIn("n_p < (ln p)^2", formatted)

        # Quoted phrase spanning a PDF line break
        scope_q, hits_q, _ = lib.search("lamzouri_2013_grh", '"least quadratic non-residue"')
        self.assertEqual(scope_q, "lamzouri_2013_grh")
        self.assertGreaterEqual(len(hits_q), 1)

        # Redundant "all <doc_id> <query>" prefix unwrapping
        scope_d, hits_d, fmt_d = lib.search("all", 'lamzouri_2013_grh "explicit constants"')
        self.assertEqual(scope_d, "lamzouri_2013_grh")
        self.assertGreaterEqual(len(hits_d), 1)
        self.assertIn("Explicit constants", fmt_d)

        # Unicode math symbol normalization (log^2 Delta vs log 2 ∆)
        scope_u, hits_u, _ = lib.search("grenie_molteni_2022_classgroup", '"4 log^2 Delta"')
        self.assertEqual(scope_u, "grenie_molteni_2022_classgroup")
        self.assertGreaterEqual(len(hits_u), 1)

        # Ranked partial-match fallback when only a subset of tokens appears
        scope_p, hits_p, _ = lib.search("grenie_molteni_2022_classgroup", '"conductor" "totally_absent_token"')
        self.assertEqual(scope_p, "grenie_molteni_2022_classgroup")
        self.assertGreaterEqual(len(hits_p), 1)

        with self.assertRaises(DocumentError):
            lib.read("nonexistent_doc", offset=0)
        with self.assertRaises(DocumentError):
            lib.read("alpoge_2026_remark", offset=999999)

    def test_protocol_parses_doc_commands_and_json(self):
        p = parse_reply(
            "Checking the papers.\n@DOC_LIST\n@DOC_SEARCH all theta lifts\n@DOC_READ alpoge_2026_remark 35",
            me="A",
            names=("A", "B"),
        )
        self.assertEqual([a["type"] for a in p.actions], ["DOC_LIST", "DOC_SEARCH", "DOC_READ"])
        self.assertEqual(p.actions[1]["target"], "all")
        self.assertEqual(p.actions[1]["query"], "theta lifts")
        self.assertEqual(p.actions[2]["doc_id"], "alpoge_2026_remark")
        self.assertEqual(p.actions[2]["offset"], 35)

        p_json = parse_reply(
            json.dumps({
                "message": "Reading paper 2",
                "actions": [
                    {"type": "DOC_SEARCH", "target": "lamzouri_2013_grh", "query": "non-residue"},
                    {"type": "DOC_READ", "doc_id": "lamzouri_2013_grh", "offset": 0},
                ],
            }),
            me="A",
            names=("A", "B"),
        )
        self.assertEqual(p_json.actions[0]["type"], "DOC_SEARCH")
        self.assertEqual(p_json.actions[1]["type"], "DOC_READ")

    def test_chamber_doc_search_and_read_injects_next_context_then_clears(self):
        cfg = {
            "name": "doc_test",
            "topic": "Analyze papers",
            "scheduler": "round_robin",
            "doc_manifest": str(self.manifest_path),
            "limits": {"max_turns": 3},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full"},
                {"name": "B", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            "A": [
                "@DOC_SEARCH all sqrt(X)\n@DOC_READ lamzouri_2013_grh 0\nSearching now.",
                "Got the excerpts!",
            ],
            "B": ["Waiting for A."],
        })
        ch = Chamber(cfg, run_dir=self.root / "doc_run", backend=backend, quiet=True, copy_dir="")
        ch.run()

        self.assertEqual(len(backend.calls), 3)
        # First call for A sees the manifest
        a_prompt1 = json.dumps(backend.calls[0][1]["messages"])
        self.assertIn("Reference Document Library", a_prompt1)
        self.assertIn("alpoge_2026_remark", a_prompt1)
        # Third call (A's turn 3) sees the exact requested search & read excerpts
        a_prompt2 = json.dumps(backend.calls[2][1]["messages"])
        self.assertIn("Requested Reference Document Excerpts", a_prompt2)
        self.assertIn("U(X_0) -> ... -> U(X_m)", a_prompt2)
        self.assertIn("n_p < (ln p)^2", a_prompt2)
        self.assertEqual(ch.agents["A"].pending_doc_reads, [])
        metrics = ch.compute_metrics()
        self.assertEqual(metrics["documents_mounted"], 3)
        self.assertEqual(metrics["doc_reads_count"], 1)
        self.assertEqual(metrics["doc_searches_count"], 1)

    def test_resume_reopen_completed_agent_decided_run_with_doc_manifest(self):
        cfg = {
            "name": "phase1_end",
            "topic": "Phase 1 debate",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "artifacts": {
                "seed": {"synthesis": "# Title\n[Part 1]\nInitial placeholder.\n"},
                "visible": ["synthesis"],
                "validation": {
                    "artifact": "synthesis",
                    "require_modified": True,
                    "min_bytes": 40,
                    "required_sections": ["[Part 1]"],
                },
            },
            "limits": {"max_turns": 6, "max_requests": 20},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full"},
                {"name": "B", "endpoint": "local", "capabilities": "full"},
            ],
        }
        src_dir = self.root / "phase1_run"
        b1 = MockBackend({
            "A": ["@REPLACE synthesis: # Phase 1\n[Part 1]\nSolid initial synthesis content here.\n@END_RUN: phase 1 done"],
            "B": ["Concur.\n@END_RUN: done"],
        })
        ch1 = Chamber(cfg, run_dir=src_dir, backend=b1, quiet=True, copy_dir="")
        ch1.run()
        self.assertTrue(ch1.end_requested)
        self.assertEqual(ch1.turn, 2)

        before_hashes = {
            str(p.relative_to(src_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in src_dir.rglob("*") if p.is_file()
        }

        cont_dir = self.root / "phase2_run"
        topic_file = self.root / "phase2_topic.txt"
        topic_file.write_text("Phase 2: Re-evaluate using newly mounted papers!", encoding="utf-8")
        mode_file = self.root / "phase2_mode.txt"
        mode_file.write_text("Use @DOC_SEARCH and update [Part 2] in synthesis.", encoding="utf-8")
        val_file = self.root / "phase2_val.json"
        val_file.write_text(json.dumps({
            "artifact": "synthesis",
            "require_modified": True,
            "min_bytes": 80,
            "required_sections": ["[Part 1]", "[Part 2]"],
        }), encoding="utf-8")

        b2 = MockBackend({
            "A": [
                "@DOC_SEARCH all theta lifts\n@END_RUN: premature try",
                "Still verifying.",
            ],
            "B": [
                "Reading.\n@END_RUN: premature majority",
                "@REPLACE synthesis: # Phase 2\n[Part 1]\nSolid initial synthesis content here.\n[Part 2]\nAudited against alpoge_2026_remark!\n@END_RUN: phase 2 verified",
            ],
        })

        def _mock_demo_complete(self_demo, *args, **kwargs):
            return b2.complete(*args, **kwargs)

        with patch.object(DemoBackend, "complete", _mock_demo_complete):
            rc = main([
                "--resume", str(src_dir),
                "--run-dir", str(cont_dir),
                "--reopen",
                "--doc-manifest", str(self.manifest_path),
                "--topic-file", str(topic_file),
                "--mode-text-file", str(mode_file),
                "--validation-file", str(val_file),
                "--turns", "6",
                "--min-turns-before-end", "4",
                "--quiet",
            ])
        self.assertEqual(rc, 0)

        after_hashes = {
            str(p.relative_to(src_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in src_dir.rglob("*") if p.is_file()
        }
        self.assertEqual(before_hashes, after_hashes)

        cont_metrics = json.loads((cont_dir / "metrics.json").read_text())
        self.assertEqual(cont_metrics["documents_mounted"], 3)
        self.assertEqual(cont_metrics["doc_searches_count"], 1)
        self.assertTrue(cont_metrics["deliverable_verified"])
        self.assertEqual(sum(cont_metrics["turns_by_agent"].values()), 6)
        self.assertEqual(cont_metrics["requests_used"], 6)
        events = [json.loads(line) for line in (cont_dir / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["type"] == "END_RUN" and e["status"] == "rejected" for e in events))
        self.assertTrue(any(e["type"] == "DELIVERABLE_VERIFIED" for e in events))
        self.assertIn("Audited against alpoge_2026_remark!", (cont_dir / "artifacts" / "synthesis.md").read_text())


if __name__ == "__main__":
    unittest.main()
