#!/usr/bin/env python3
"""Targeted regression tests for 2026-11-23 Dialogue Chamber harness fixes & upgrades."""
import json
import tempfile
import unittest
from pathlib import Path

from anomaly_lab import AnomalyLaboratory
from chamber import Chamber
from context import ContextIntelligence
from documents import DocumentLibrary
from inference import MockBackend, Result
from protocol import canonicalize_agent_id, parse_read_offset, parse_reply


class HarnessFixes20261123Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.doc_dir = self.root / "docs"
        self.doc_dir.mkdir()
        (self.doc_dir / "guide.tex").write_text(
            "\\documentclass{article}\n\\begin{document}\n"
            "\\section{The Main Inequality $(M)$}\n"
            "Let $\\chi$ denote a Dirichlet character and $\\Lambda$ the von Mangoldt function.\n"
            "\\subsection{\\texorpdfstring{Bost Slope \\& Seesaw}{Bost Slope and Seesaw}}\n"
            "By Bost's slope inequality and Rallis seesaw on $U(X_0)$.\n"
            "\\end{document}\n",
            encoding="utf-8",
        )
        (self.doc_dir / "manifest.json").write_text(
            json.dumps({
                "documents": [{
                    "id": "guide",
                    "title": "Proof Reconstruction Guide",
                    "authors": "Test",
                    "year": "2026",
                    "path": "guide.tex",
                }]
            }),
            encoding="utf-8",
        )

    def test_tolerant_read_offsets_and_toc_command(self):
        for raw, expected in (
            (0, 0),
            (11206, 11206),
            ("[11206]", 11206),
            ("[11,206]", 11206),
            ("1000-2000", 1000),
            ("[1000-2000]", 1000),
            ("1000..2000", 1000),
            ("offset=8000", 8000),
            ("offset:12000", 12000),
        ):
            val, err = parse_read_offset(raw)
            self.assertIsNone(err, msg=f"{raw!r}: {err}")
            self.assertEqual(val, expected)

        for bad in (-1, True, False, "12.5", "abc", "-50"):
            val, err = parse_read_offset(bad)
            self.assertIsNone(val, msg=f"{bad!r} should be rejected")
            self.assertIn("offset", err)

        p = parse_reply(
            "Checking.\n"
            "@DOC_TOC guide\n"
            "@DOC_READ guide [12]\n"
            "@DOC_READ guide [100 - 200]\n"
            "@DOC_READ guide offset: 50\n"
            "@DOC_READ guide 88 (Main Inequality)\n"
            "@DOC_READ guide toc\n"
            "@READ report: 25\n"
            "@DOC_READ guide -1\n"
            "@DOC_READ guide 50 bogus_arg\n",
            me="A",
            names=("A", "B"),
        )
        self.assertEqual([a["type"] for a in p.actions], ["DOC_TOC", "DOC_READ", "DOC_READ", "DOC_READ", "DOC_READ", "DOC_TOC", "ARTIFACT_READ"])
        self.assertEqual(p.actions[1]["offset"], 12)
        self.assertEqual(p.actions[2]["offset"], 100)
        self.assertEqual(p.actions[3]["offset"], 50)
        self.assertEqual(p.actions[4]["offset"], 88)
        self.assertEqual(p.actions[6]["offset"], 25)
        self.assertEqual(len(p.issues), 2)

    def test_doc_toc_and_latex_search(self):
        lib = DocumentLibrary(self.doc_dir / "manifest.json")
        doc_id, _, entries, fmt = lib.table_of_contents("guide")
        self.assertEqual(doc_id, "guide")
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["title"], "The Main Inequality (M)")
        self.assertEqual(entries[1]["title"], "Bost Slope and Seesaw")
        self.assertIn("offset " + str(entries[0]["offset"]), fmt)

        _, _, excerpt, _, _ = lib.read("guide", offset=entries[0]["offset"], max_chars=200)
        self.assertTrue(excerpt.startswith("\\section{The Main Inequality"))

        _, hits_chi, _ = lib.search("guide", r"\chi")
        self.assertGreaterEqual(len(hits_chi), 1)
        _, hits_m, _ = lib.search("guide", "Main Inequality (M)")
        self.assertGreaterEqual(len(hits_m), 1)

    def test_multiline_artifact_write_ignores_prose_and_bibtex(self):
        reply = (
            "Publishing consolidated draft.\n"
            "@REPLACE report: # Proof Report\n"
            "Temperature: critical in the Bost slope bound.\n"
            "Dry: not needed.\n"
            "Next: consider why $U(X_0)$ scales as $\\sqrt{X}$.\n"
            "@article{alpoge2026,\n"
            "  title={Remark}\n"
            "}\n"
            "/section/subsection note\n"
            "@END_RUN: complete\n"
            "NEXT: B\n"
        )
        p = parse_reply(reply, me="A", names=("A", "B"))
        self.assertEqual(p.issues, [])
        self.assertEqual(p.next_speaker, "B")
        self.assertEqual([a["type"] for a in p.actions], ["ARTIFACT_WRITE", "END_RUN"])
        content = p.actions[0]["content"]
        self.assertIn("Temperature: critical", content)
        self.assertIn("Next: consider", content)
        self.assertIn("@article{alpoge2026", content)
        self.assertIn("/section/subsection note", content)

    def test_canonicalize_agent_id_multi_token_none_and_slash(self):
        names = {"AGENT_A", "AGENT_B"}
        self.assertEqual(canonicalize_agent_id("NONE 10/10 no wait", names), "NONE")
        self.assertEqual(canonicalize_agent_id("AGENT_A / AGENT_B", names), "AGENT_A")
        self.assertEqual(canonicalize_agent_id("GHOST", names), "GHOST")
        self.assertEqual(canonicalize_agent_id("AGENT_A_AGENT_B", names), "AGENT_A_AGENT_B")

        p = parse_reply("Done.\nNEXT: NONE (10/10, ready)", me="AGENT_A", names=("AGENT_A", "AGENT_B"))
        self.assertIsNone(p.next_speaker)
        self.assertEqual(p.issues, [])

    def test_dyad_lockin_breaker_and_anti_starvation_clear_nomination(self):
        cfg = {
            "name": "dyad_test",
            "topic": "Avoid ping-pong",
            "scheduler": "self_directed",
            "limits": {"max_turns": 7, "max_consecutive": 1, "max_dyad_turns": 4, "starvation_turns": 20},
            "agents": [
                {"name": "A", "endpoint": "local"},
                {"name": "B", "endpoint": "local"},
                {"name": "C", "endpoint": "local"},
                {"name": "D", "endpoint": "local"},
            ],
        }
        backend = MockBackend({
            "A": ["A1\nNEXT: B", "A2\nNEXT: B", "A3\nNEXT: B"],
            "B": ["B1\nNEXT: A", "B2\nNEXT: A", "B3\nNEXT: A"],
            "C": ["C speaks!\nNEXT: D"],
            "D": ["D speaks!\nNEXT: C"],
        })
        ch = Chamber(cfg, run_dir=self.root / "dyad_run", backend=backend, quiet=True, copy_dir="").run()
        speakers = [m.speaker for m in ch.public_messages]
        # Turns 1-4: A, B, A, B; Turn 5 MUST break out to C (not A/B!)
        self.assertEqual(speakers[:5], ["A", "B", "A", "B", "C"])
        self.assertTrue(any("dyad lock-in breaker" in e.detail for e in ch.events))

    def test_prefix_cache_layout_feedback_persistence_and_token_accounting(self):
        cfg = {
            "name": "cache_fb_test",
            "topic": "Analyze proof architecture",
            "scheduler": "round_robin",
            "doc_manifest": str(self.doc_dir / "manifest.json"),
            "artifacts": {
                "seed": {"report": "# Seed\n"},
                "visible": ["report"],
                "validation": {"artifact": "report", "require_modified": True, "min_bytes": 50, "required_sections": ["[Sec1]"]},
            },
            "limits": {"max_turns": 6},
            "agents": [
                {"name": "A", "endpoint": "local", "capabilities": "full", "identity": "Specialist Alpha"},
                {"name": "B", "endpoint": "local", "capabilities": "full", "identity": "Specialist Beta"},
                {"name": "C", "endpoint": "local", "capabilities": "full"},
                {"name": "D", "endpoint": "local", "capabilities": "full"},
                {"name": "E", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            # Turn 1: A runs an invalid DOC_READ (turn 1); turns 2..5 are B,C,D,E; turn 6 is A again!
            "A": ["@DOC_READ missing_paper 0\n@DOC_TOC guide\nTry 1.", "Turn 6 A."],
            "B": ["Turn 2 B."],
            "C": ["Turn 3 C."],
            "D": ["Turn 4 D."],
            "E": ["Turn 5 E."],
        })
        ch = Chamber(cfg, run_dir=self.root / "cache_run", backend=backend, quiet=True, copy_dir="").run()

        # System messages are byte-identical between A (call 0) and B (call 1) for KV cache prefix sharing
        a_sys = backend.calls[0][1]["messages"][0]["content"]
        b_sys = backend.calls[1][1]["messages"][0]["content"]
        self.assertEqual(a_sys, b_sys)

        # User prompt puts Topic + Doc Manifest BEFORE Public Conversation, and identity/artifacts/status AFTER
        a2_user = backend.calls[5][1]["messages"][1]["content"]
        self.assertLess(a2_user.index("Reference Document Library"), a2_user.index("--- Public Conversation ---"))
        self.assertLess(a2_user.index("--- Public Conversation ---"), a2_user.index("Your Identity & Role (A)"))
        self.assertLess(a2_user.index("--- Public Conversation ---"), a2_user.index("Requested Reference Document Excerpts"))
        # Rejection from Turn 1 persisted across 5 intervening turns until A's Turn 6!
        self.assertIn("DOC_READ [missing_paper]: REJECTED", a2_user)
        # Live session status shows incomplete deliverable status
        self.assertIn("Deliverable `report.md`: INCOMPLETE", a2_user)

        # TokenAccounting captures reference_documents accurately
        self.assertGreater(ch.token_accounting_history[0]["reference_documents"], 0)
        metrics = ch.compute_metrics()
        self.assertEqual(metrics["doc_toc_count"], 1)

    def test_sampling_noop_and_cache_metrics_and_anomaly_v2_checkpoint(self):
        cfg = {
            "name": "noop_cache",
            "topic": "Check noop sampling and cache rate",
            "scheduler": "round_robin",
            "limits": {"max_turns": 2},
            "agents": [
                {"name": "A", "endpoint": "local", "top_p": 0.9, "top_k": 40},
                {"name": "B", "endpoint": "local"},
            ],
        }
        backend = MockBackend({
            "A": [Result(
                content="Noop sampling.\n@SET_SAMPLING top_p=0.9 top_k=40",
                finish_reason="stop", prompt_tokens=1000, completion_tokens=50,
                cached_tokens=800, cache_source="timings.cache_n", latency=1.0,
            )],
            "B": [Result(
                content="Turn 2.",
                finish_reason="stop", prompt_tokens=1000, completion_tokens=50,
                cached_tokens=600, cache_source="timings.cache_n", latency=1.0,
            )],
        })
        ch = Chamber(cfg, run_dir=self.root / "metrics_run", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(ch.sampling_changes, [])
        metrics = ch.compute_metrics()
        self.assertEqual(metrics["rejected_actions"], 0)
        self.assertAlmostEqual(metrics["prompt_cache_hit_rate"], 0.7, places=3)
        self.assertAlmostEqual(metrics["mean_completion_tok_per_s"], 50.0, places=1)

        # AnomalyLaboratory on schema-v2 checkpoint
        run_dir = self.root / "anom_v2"
        run_dir.mkdir()
        (run_dir / "events.jsonl").write_text(
            json.dumps({
                "seq": 1, "turn": 3, "actor": "A", "type": "CHALLENGE", "status": "rejected",
                "target": "B", "detail": "target item 'B' not on board (existing IDs: c1)",
            }) + "\n",
            encoding="utf-8",
        )
        (run_dir / "checkpoint.json").write_text(json.dumps({
            "schema_version": 2,
            "state": {
                "config": {"scheduler": "self_directed", "agents": [{"name": "A"}, {"name": "B"}, {"name": "C"}]},
                "public_messages": [
                    {"turn": 1, "speaker": "A", "content": "x"},
                    {"turn": 2, "speaker": "B", "content": "y"},
                    {"turn": 3, "speaker": "A", "content": "x"},
                    {"turn": 4, "speaker": "B", "content": "y"},
                    {"turn": 5, "speaker": "A", "content": "x"},
                ],
            },
        }), encoding="utf-8")
        lab = AnomalyLaboratory(lab_dir=self.root / "anom_out")
        found = {sp.anomaly_type for sp in lab.inspect_run(run_dir)}
        self.assertIn("dyadic_floor_starvation", found)
        self.assertIn("blackboard_target_category_error", found)

    def test_kv_messages_layout_and_trimming(self):
        ctx = ContextIntelligence(default_safety_margin=128)
        sys_sec = {"preamble": "Chamber rules.", "protocol": "Write plain text."}
        fixed_sec = {"topic": "Topic: Riemann Hypothesis proof compression.", "doc_manifest": "Docs: guide."}
        tail_a = {"agent_identity": "You are ALPHA.", "scratchpad": "ALPHA private note."}
        tail_b = {"agent_identity": "You are BETA.", "scratchpad": "BETA private note."}
        tail_c = {"agent_identity": "You are GAMMA.", "scratchpad": "GAMMA private note."}

        # "auto" with provider="mock" stays single-user 2-message format; with "llamacpp" uses kv_messages
        mock_msgs, _ = ctx.assemble_and_budget(
            agent_name="ALPHA",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_a,
            prompt_layout="auto",
        )
        self.assertEqual(len(mock_msgs), 2)

        # Avoid HTTP /tokenize timeout by explicitly specifying prompt_layout="kv_messages" with mock tokenizer
        m1 = {"id": "m1", "turn": 1, "speaker": "ALPHA", "content": "First public message."}
        m2 = {"id": "m2", "turn": 2, "speaker": "BETA", "content": "Second public message."}

        t1_msgs, _ = ctx.assemble_and_budget(
            agent_name="ALPHA",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_a,
            prompt_layout="kv_messages",
        )
        # Turn 1 must be [system, user(fixed), user(empty_notice + tail)]
        self.assertEqual([m["role"] for m in t1_msgs], ["system", "user", "user"])
        self.assertEqual(t1_msgs[1]["content"], "Topic: Riemann Hypothesis proof compression.\n\nDocs: guide.")
        self.assertIn("(No public messages yet", t1_msgs[2]["content"])
        self.assertIn("Your turn (ALPHA):", t1_msgs[2]["content"])

        t2_msgs, _ = ctx.assemble_and_budget(
            agent_name="BETA",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[m1],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_b,
            prompt_layout="kv_messages",
        )
        t3_msgs, _ = ctx.assemble_and_budget(
            agent_name="GAMMA",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[m1, m2],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_c,
            prompt_layout="kv_messages",
        )
        # Prefix invariance across growing conversation turns
        self.assertEqual(t2_msgs[:2], t1_msgs[:2])
        self.assertEqual(t3_msgs[:3], t2_msgs[:3])
        self.assertEqual(len(t3_msgs), 5)
        self.assertIn("BETA: Second public message.", t3_msgs[3]["content"])
        self.assertIn("You are GAMMA.", t3_msgs[4]["content"])
        self.assertIn("Your turn (GAMMA):", t3_msgs[4]["content"])

        # Context trimming under kv_messages preserves system, fixed, newest message, tail, and omission notice
        many_msgs = [
            {"id": f"m{i}", "turn": i, "speaker": "ALPHA" if i % 2 else "BETA", "content": f"Message #{i} " + ("zeta prime " * 18)}
            for i in range(1, 25)
        ]
        trimmed_msgs, trimmed_acc = ctx.assemble_and_budget(
            agent_name="GAMMA",
            endpoint={"provider": "mock"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=many_msgs,
            effective_context=1200,
            context_source="config",
            reserved_output=400,
            safety_margin=120,
            tail_sections=tail_c,
            prompt_layout="kv_messages",
        )
        self.assertGreater(trimmed_acc.omitted_messages_count, 0)
        self.assertEqual(trimmed_msgs[0]["content"], t1_msgs[0]["content"])
        self.assertEqual(trimmed_msgs[1]["content"], t1_msgs[1]["content"])
        self.assertIn("earlier public message(s) omitted", trimmed_msgs[2]["content"])
        self.assertIn("Message #24", trimmed_msgs[-2]["content"])
        self.assertIn("Your turn (GAMMA):", trimmed_msgs[-1]["content"])

        # prompt_layout="auto" with provider="llamacpp" resolves to multi-message kv_messages layout
        auto_llama_msgs, _ = ctx.assemble_and_budget(
            agent_name="BETA",
            endpoint={"provider": "llamacpp"},
            system_sections=sys_sec,
            fixed_sections=fixed_sec,
            public_messages=[m1],
            effective_context=32768,
            context_source="config",
            tail_sections=tail_b,
            prompt_layout="auto",
        )
        self.assertEqual(len(auto_llama_msgs), 4)

    def test_24_char_agent_id_and_bold_peer_directive_non_truncation(self):
        names = {"SYNTHESIS_CHAIR", "LEAN_KERNEL_AUDIT", "PROSE_GAP_AUDITOR"}
        self.assertEqual(canonicalize_agent_id("LEAN_KERNEL_AUDIT", names), "LEAN_KERNEL_AUDIT")
        self.assertEqual(canonicalize_agent_id("PROSE_GAP_AUDITOR", names), "PROSE_GAP_AUDITOR")

        # Bold peer-addressing directive after ':' or starting with a directive verb must NOT be truncated
        reply_bold_ok = (
            "I require two specific checks:\n"
            "**LEAN_KERNEL_AUDIT**: Please confirm the nanoda declaration count is Lean/Mathlib decls.\n"
            "**PROSE_GAP_AUDITOR**: Verify Section I.12 hypotheses.\n"
            "NEXT: LEAN_KERNEL_AUDIT"
        )
        p_ok = parse_reply(reply_bold_ok, me="SYNTHESIS_CHAIR", names=tuple(names))
        self.assertEqual(p_ok.flags, [])
        self.assertEqual(p_ok.next_speaker, "LEAN_KERNEL_AUDIT")
        self.assertIn("Verify Section I.12 hypotheses.", p_ok.message)

        # Genuine impersonation (unbolded label or with transcript turn tag) MUST still be caught & cut
        p_imp1 = parse_reply(
            "My point.\nLEAN_KERNEL_AUDIT: I confirm all axioms.",
            me="SYNTHESIS_CHAIR",
            names=tuple(names),
        )
        self.assertEqual(p_imp1.message, "My point.")
        self.assertTrue(any("wrote a line as LEAN_KERNEL_AUDIT" in f for f in p_imp1.flags))

        p_imp2 = parse_reply(
            "My point.\n[m2 | turn 2] **LEAN_KERNEL_AUDIT**: Please confirm all axioms.",
            me="SYNTHESIS_CHAIR",
            names=tuple(names),
        )
        self.assertEqual(p_imp2.message, "My point.")
        self.assertTrue(any("wrote a line as LEAN_KERNEL_AUDIT" in f for f in p_imp2.flags))

    def test_doc_coverage_status_repeat_end_run_and_anomaly_non_false_positive(self):
        (self.doc_dir / "second.tex").write_text("Second paper content about Stekloff and Stepanov.", encoding="utf-8")
        (self.doc_dir / "manifest.json").write_text(
            json.dumps({
                "documents": [
                    {"id": "guide", "title": "Proof Guide", "authors": "A", "year": "2026", "path": "guide.tex"},
                    {"id": "second", "title": "Second Paper", "authors": "B", "year": "2026", "path": "second.tex"},
                ]
            }),
            encoding="utf-8",
        )
        cfg = {
            "name": "cov_end_test",
            "topic": "Check doc coverage and repeat END_RUN.",
            "scheduler": "round_robin",
            "end_mode": "agent_decided",
            "end_policy": "majority",
            "min_turns_before_end": 1,
            "doc_manifest": str(self.doc_dir / "manifest.json"),
            "limits": {"max_turns": 5},
            "agents": [
                {"name": "LEAN_KERNEL_AUDIT", "endpoint": "local", "capabilities": "full"},
                {"name": "PROSE_GAP_AUDITOR", "endpoint": "local", "capabilities": "full"},
                {"name": "SYNTHESIS_CHAIR", "endpoint": "local", "capabilities": "full"},
            ],
        }
        backend = MockBackend({
            "LEAN_KERNEL_AUDIT": [
                "Reading guide.\n@DOC_READ guide 0\n@END_RUN: initial vote",
                "Reaffirming end.\n@END_RUN: still ready",
            ],
            "PROSE_GAP_AUDITOR": ["Agree on next step.", "Still reviewing."],
            "SYNTHESIS_CHAIR": ["Reviewing guide first.", "Checking coverage.\n@END_RUN: quorum"],
        })
        ch = Chamber(cfg, run_dir=self.root / "cov_run", backend=backend, quiet=True, copy_dir="").run()
        # Turn 2 prompt (call index 1, PROSE_GAP_AUDITOR) should report Ref docs inspected (1/2) & Recent reads
        t2_tail = backend.calls[1][1]["messages"][-1]["content"]
        self.assertIn("Ref docs inspected (1/2): unread = second", t2_tail)
        self.assertIn("Recent reads: guide@0", t2_tail)
        # LEAN_KERNEL_AUDIT remained untruncated in agent registry & reaffirmation logged cleanly
        self.assertIn("LEAN_KERNEL_AUDIT", ch.agents)
        self.assertTrue(any(e.type == "END_RUN" and "(vote reaffirmed)" in e.detail for e in ch.events))
        metrics = ch.compute_metrics()
        self.assertEqual(metrics["prompt_layout"], "auto")

        # anomaly_lab non-false-positive: "wrote a line as" alone must not emit prompt_leakage_and_institutional_paranoia
        anom_dir = self.root / "anom_attrib_only"
        anom_dir.mkdir()
        (anom_dir / "events.jsonl").write_text(
            json.dumps({
                "seq": 1, "turn": 3, "actor": "SYNTHESIS_CHAIR", "type": "ATTRIBUTION_FLAG",
                "status": "system", "detail": "wrote a line as BETA; reply cut there",
            }) + "\n",
            encoding="utf-8",
        )
        (anom_dir / "checkpoint.json").write_text(json.dumps({
            "schema_version": 2,
            "state": {
                "config": {"scheduler": "round_robin", "agents": [{"name": "SYNTHESIS_CHAIR"}, {"name": "BETA"}]},
                "public_messages": [
                    {"turn": 1, "speaker": "SYNTHESIS_CHAIR", "content": "Opening."},
                    {"turn": 2, "speaker": "BETA", "content": "Response."},
                ],
            },
        }), encoding="utf-8")
        lab = AnomalyLaboratory(lab_dir=self.root / "anom_out2")
        found = {sp.anomaly_type for sp in lab.inspect_run(anom_dir)}
        self.assertNotIn("prompt_leakage_and_institutional_paranoia", found)


if __name__ == "__main__":
    unittest.main()
