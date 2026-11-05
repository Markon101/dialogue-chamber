#!/usr/bin/env python3
"""Verification tests for Dynamic OpenRouter Multi-Model Support, Caching, and Fallback."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from models import ModelMetadata, OpenRouterCatalog
from inference import HttpBackend, InferenceError, MockBackend, Result
from chamber import Chamber


class TestOpenRouterModels(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_models_"))
        self.cache_file = self.temp_dir / "test_models.json"
        self.catalog = OpenRouterCatalog(cache_file=self.cache_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_catalog_caching_and_load(self):
        """Verify model catalog writes to cache and loads without network call."""
        # Seed test catalog
        sample_models = [
            ModelMetadata(
                id="meta-llama/llama-3.2-3b-instruct:free",
                name="Llama 3.2 3B (free)",
                context_length=131072,
                is_free=True,
                has_reasoning=False,
            ),
            ModelMetadata(
                id="deepseek/deepseek-r1",
                name="DeepSeek R1",
                context_length=65536,
                prompt_price_per_m=0.55,
                completion_price_per_m=2.19,
                is_free=False,
                has_reasoning=True,
            ),
        ]
        self.catalog._models = {m.id: m for m in sample_models}
        self.catalog._last_fetched = 1000.0
        self.catalog._save_cache([])

        # Reload in a new instance
        reloaded = OpenRouterCatalog(cache_file=self.cache_file)
        self.assertEqual(len(reloaded._models), 2)
        m_free = reloaded.get_model("meta-llama/llama-3.2-3b-instruct:free")
        self.assertIsNotNone(m_free)
        self.assertTrue(m_free.is_free)
        self.assertEqual(m_free.context_length, 131072)

        m_paid = reloaded.get_model("deepseek/deepseek-r1")
        self.assertIsNotNone(m_paid)
        self.assertFalse(m_paid.is_free)
        self.assertTrue(m_paid.has_reasoning)

    def test_search_and_filter_free_models(self):
        """Verify searching and filtering for free models only."""
        self.catalog._models = {
            "google/gemma-2-9b-it:free": ModelMetadata(id="google/gemma-2-9b-it:free", name="Gemma 2 9B Free", is_free=True, context_length=8192),
            "google/gemma-2-27b-it": ModelMetadata(id="google/gemma-2-27b-it", name="Gemma 2 27B Paid", is_free=False, context_length=8192),
            "qwen/qwen-2.5-72b-instruct": ModelMetadata(id="qwen/qwen-2.5-72b-instruct", name="Qwen 2.5 72B", is_free=False, context_length=32768),
        }
        # Search "gemma" free only
        res = self.catalog.search_models("gemma", free_only=True)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].id, "google/gemma-2-9b-it:free")

        # List free only
        free_list = self.catalog.list_models(free_only=True)
        self.assertEqual(len(free_list), 1)
        self.assertEqual(free_list[0].id, "google/gemma-2-9b-it:free")

    def test_free_tier_guard_blocks_paid_models(self):
        """Verify HttpBackend with free_only=True blocks requests for paid models."""
        backend = HttpBackend(free_only=True, allow_paid=False)
        ep = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-3-opus"}
        with self.assertRaises(InferenceError) as ctx:
            backend.complete("TEST_AGENT", ep, {"model": "anthropic/claude-3-opus", "messages": []})
        self.assertIn("not in free tier", str(ctx.exception))

    def test_per_agent_fallback_chain_execution(self):
        """Verify that when an agent's primary model fails, it falls back to the configured backup model."""
        calls = []

        class FailingPrimaryMock:
            def complete(self, agent_name, endpoint, payload, timeout=120, reasoning_mode_label="none"):
                calls.append((agent_name, endpoint.get("model")))
                if endpoint.get("model") == "failing/primary-model":
                    raise InferenceError("HTTP 503 Provider unavailable", transient=True)
                return Result(content="Fallback model succeeded!", actual_model=endpoint.get("model"))

        cfg = {
            "name": "fallback_test",
            "topic": "Test fallback mechanism",
            "scheduler": "round_robin",
            "limits": {"max_turns": 1, "max_retries_per_call": 1},
            "agents": [
                {
                    "name": "AGENT_A",
                    "endpoint": {"provider": "openrouter", "name": "primary", "model": "failing/primary-model"},
                    "fallback_models": [
                        {"provider": "openrouter", "name": "backup", "model": "working/backup-model"}
                    ],
                }
            ],
        }

        ch = Chamber(cfg, run_dir=self.temp_dir / "run_fb", backend=FailingPrimaryMock(), quiet=True, copy_dir="")
        ch.run()

        self.assertEqual(len(calls), 3)  # initial try + 1 retry on primary, then fallback to backup
        self.assertEqual(calls[-1][1], "working/backup-model")
        self.assertEqual(len(ch.fallbacks_triggered), 1)
        self.assertEqual(ch.fallbacks_triggered[0]["to_endpoint"], "backup")


if __name__ == "__main__":
    unittest.main()
