#!/usr/bin/env python3
"""Verification tests for Tokenizer-Aware Context Intelligence."""
import math
import unittest
from context import ContextIntelligence, TokenAccounting, TokenizerEngine


class TestContextIntelligence(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerEngine()
        self.ctx = ContextIntelligence(default_safety_margin=512, tokenizer=self.tokenizer)

    def test_conservative_token_estimation_labeling(self):
        """Verify that when exact tokenizer is unavailable, estimation is explicitly labeled."""
        text = "Hello multi-agent world! Let us negotiate a constitution."
        cnt, method = self.tokenizer.count_tokens(text, endpoint={"provider": "mock"})
        self.assertEqual(method, "estimated")
        self.assertGreater(cnt, 0)
        # Verify conservative property: tokens should never be zero or severely undercounted
        self.assertGreaterEqual(cnt, math.ceil(len(text) / 4))

    def test_context_negotiation_sources(self):
        """Verify context negotiation distinguishes server-reported, config, and manual sources."""
        # 1. Server reported
        limit, src, conf = self.ctx.negotiate_context_limit(server_reported_ctx=32768, user_configured_ctx=None)
        self.assertEqual(limit, 32768)
        self.assertEqual(src, "server-reported")
        self.assertEqual(conf, "exact")

        # 2. User configured override
        limit, src, conf = self.ctx.negotiate_context_limit(server_reported_ctx=32768, user_configured_ctx=16384)
        self.assertEqual(limit, 16384)
        self.assertEqual(src, "config")

        # 3. Server reported is smaller than user configured -> clamp to server limit
        limit, src, conf = self.ctx.negotiate_context_limit(server_reported_ctx=8192, user_configured_ctx=32768)
        self.assertEqual(limit, 8192)
        self.assertEqual(src, "server-reported")

        # 4. Unknown server limit falls back to manual model default
        limit, src, conf = self.ctx.negotiate_context_limit(server_reported_ctx=None, user_configured_ctx=None, model_default_ctx=65536)
        self.assertEqual(limit, 65536)
        self.assertEqual(src, "manual")

    def test_context_pressure_trimming_protects_instructions(self):
        """Under context pressure, older public messages are trimmed while essential instructions remain intact."""
        sys_sections = {
            "preamble": "You are AGENT_ALPHA in a bounded laboratory.",
            "protocol": "RULES: No external execution. Speak truthfully.",
        }
        fixed_sections = {
            "topic": "Topic: What is the nature of high-dimensional space?",
            "scratchpad": "Scratchpad: Remember hypothesis H1.",
        }

        # Create 40 dialogue messages of sufficient length to exceed the 700 token budget
        pub_msgs = [
            {"id": f"m{i}", "turn": i, "speaker": "BETA" if i % 2 == 0 else "ALPHA", "content": f"Turn {i} discussion sentence about geometry and activation vectors in high-dimensional manifolds and topological embedding spaces."}
            for i in range(1, 41)
        ]

        # Use a tight effective context (e.g. 1500 tokens)
        messages, accounting = self.ctx.assemble_and_budget(
            agent_name="ALPHA",
            endpoint={"provider": "mock"},
            system_sections=sys_sections,
            fixed_sections=fixed_sections,
            public_messages=pub_msgs,
            effective_context=1500,
            context_source="config",
            reserved_output=500,
            safety_margin=200,
        )

        user_content = messages[1]["content"]
        # System instructions and fixed sections must be present
        self.assertIn("You are AGENT_ALPHA", messages[0]["content"])
        self.assertIn("RULES: No external execution", messages[0]["content"])
        self.assertIn("Topic: What is the nature of high-dimensional space?", user_content)
        self.assertIn("Scratchpad: Remember hypothesis H1", user_content)

        # Older messages should be omitted
        self.assertGreater(accounting.omitted_messages_count, 0)
        self.assertIn("earlier public message(s) omitted due to context limit", user_content)
        # Newest message must be present
        self.assertIn("Turn 40 discussion sentence", user_content)

    def test_oversized_prompt_detection(self):
        """When prompt + reserved output exceeds effective context, flag diagnostic."""
        sys_sections = {"preamble": "A" * 8000}
        fixed_sections = {"topic": "B" * 8000}
        messages, accounting = self.ctx.assemble_and_budget(
            agent_name="ALPHA",
            endpoint={"provider": "mock"},
            system_sections=sys_sections,
            fixed_sections=fixed_sections,
            public_messages=[],
            effective_context=2048,
            context_source="manual",
            reserved_output=1024,
            safety_margin=256,
        )
        self.assertFalse(accounting.fits_in_context)
        self.assertGreater(len(accounting.diagnostics), 0)
        self.assertIn("Oversized prompt", accounting.diagnostics[0])


if __name__ == "__main__":
    unittest.main()
