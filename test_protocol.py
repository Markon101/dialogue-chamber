#!/usr/bin/env python3
"""Verification tests for Protocol Parsing, Canonical Agent IDs, and Formatting Repair."""
import json
import unittest

from protocol import (
    canonicalize_agent_id,
    parse_reply,
    repair_unambiguous_json,
    validate_action_schema,
)


class TestProtocolAndParsing(unittest.TestCase):
    def test_canonicalize_agent_id(self):
        """Verify normalization of punctuation, markdown wrappers, casing, and hyphens."""
        names = {"EMPEROR_NULL", "HIGH_CHANCELLOR", "JANITOR_ZERO"}
        self.assertEqual(canonicalize_agent_id("@EMPEROR_NULL:", names), "EMPEROR_NULL")
        self.assertEqual(canonicalize_agent_id("<janitor-zero>", names), "JANITOR_ZERO")
        self.assertEqual(canonicalize_agent_id("`high_chancellor`", names), "HIGH_CHANCELLOR")
        self.assertEqual(canonicalize_agent_id("emperor_null", names), "EMPEROR_NULL")
        self.assertEqual(canonicalize_agent_id("  *JANITOR_ZERO*  ", names), "JANITOR_ZERO")
        self.assertEqual(canonicalize_agent_id("EMPEROR_NULL (EMPEROR_NULL, HIGH_CHANCELLOR)", names), "EMPEROR_NULL")
        # Custom unknown agent
        self.assertEqual(canonicalize_agent_id("@NEW-AGENT_42"), "NEW_AGENT_42")

    def test_repair_unambiguous_json(self):
        """Verify repairing trailing commas and unclosed brackets without guessing semantics."""
        # 1. Trailing comma before closing brace/bracket
        broken = '{"message": "Hello", "actions": [{"type": "CLAIM", "content": "test",},],}'
        repaired = repair_unambiguous_json(broken)
        obj = json.loads(repaired)
        self.assertEqual(obj["message"], "Hello")
        self.assertEqual(len(obj["actions"]), 1)

        # 2. Markdown fence wrapping
        fenced = '```json\n{"message": "Inside fences"}\n```'
        repaired_fenced = repair_unambiguous_json(fenced)
        obj2 = json.loads(repaired_fenced)
        self.assertEqual(obj2["message"], "Inside fences")

        # 3. Truncated unclosed braces
        truncated = '{"message": "Cut off mid sentence", "actions": [{"type": "VOTE", "target": "c1"'
        repaired_trunc = repair_unambiguous_json(truncated)
        obj3 = json.loads(repaired_trunc)
        self.assertEqual(obj3["message"], "Cut off mid sentence")
        self.assertEqual(obj3["actions"][0]["target"], "c1")

    def test_action_schema_validation(self):
        """Verify action schema validation rejects incomplete/ambiguous actions with clear issues."""
        issues = []
        names = {"ALPHA", "BETA"}

        # 1. Valid whisper
        act_whisper = {"type": "WHISPER", "recipient": "@BETA", "content": "Secret"}
        v1 = validate_action_schema(act_whisper, issues, names)
        self.assertIsNotNone(v1)
        self.assertEqual(v1["recipient"], "BETA")

        # 2. Invalid whisper missing content
        act_empty_whisper = {"type": "WHISPER", "recipient": "BETA", "content": ""}
        v2 = validate_action_schema(act_empty_whisper, issues, names)
        self.assertIsNone(v2)
        self.assertTrue(any("missing message content" in iss for iss in issues))

        # 3. Invalid challenge missing target
        act_chal = {"type": "CHALLENGE", "content": "Objection"}
        v3 = validate_action_schema(act_chal, issues, names)
        self.assertIsNone(v3)
        self.assertTrue(any("missing target claim ID" in iss for iss in issues))

    def test_parse_reply_with_repaired_json(self):
        """Verify parse_reply recovers structured fields from repaired JSON."""
        reply = """
```json
{
  "message": "We must vote on the proposal.",
  "actions": [
    {"type": "VOTE", "target": "c1", "value": "yes",},
  ],
  "next_speaker": "@JANITOR-ZERO",
}
```
"""
        parsed = parse_reply(reply, names=("EMPEROR_NULL", "JANITOR_ZERO"))
        self.assertEqual(parsed.format, "json")
        self.assertEqual(parsed.message, "We must vote on the proposal.")
        self.assertEqual(parsed.next_speaker, "JANITOR_ZERO")
        self.assertEqual(len(parsed.actions), 1)
        self.assertEqual(parsed.actions[0]["value"], "yes")


if __name__ == "__main__":
    unittest.main()
