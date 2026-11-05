"""Dory routing and resource boundaries, without loading or contacting a model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chamber import Chamber, main
from inference import MockBackend, Result, HttpBackend, InferenceError
from presets import ENDPOINTS


class DoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"name": "dory_test", "topic": "A short test", "scheduler": "round_robin",
                       "limits": {"max_turns": 1}, "agents": [
                           {"name": "A", "endpoint": "dory", "capabilities": "baseline"}]}

    def test_dory_budget_and_payload_offline(self):
        backend = MockBackend({"A": ["A short reply."]})
        ch = Chamber(self.config, run_dir=self.root / "run", backend=backend, quiet=True, copy_dir="").run()
        agent = ch.agents["A"]
        self.assertEqual(agent.endpoint["base_url"], ENDPOINTS["dory"]["base_url"])
        self.assertEqual(agent.effective_context_tokens, 3072)
        self.assertEqual(agent.max_tokens, 256)
        self.assertEqual(agent.context_chars, 6000)
        payload = backend.calls[0][1]
        self.assertEqual(payload["model"], "dory")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["top_p"], 0.95)

    def test_endpoint_preflight_and_request_never_start_server(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        calls = []
        def complete(_backend, _agent, endpoint, payload, timeout, reasoning_mode_label):
            calls.append((endpoint, payload, timeout))
            return Result(content="A short reply.", prompt_tokens=20, completion_tokens=4, actual_model="dory")
        with patch("chamber.LlamaServerManager.ensure_server") as start, \
             patch("chamber.LlamaServerManager.query_server_info", return_value={"online": True, "model_id": "dory", "allocated_context": 2048}), \
             patch.object(HttpBackend, "complete", complete), \
             patch("context.TokenizerEngine.count_tokens", return_value=(10, "estimated")):
            rc = main(["--config", str(config_path), "--run-dir", str(self.root / "run"), "--quiet"])
        self.assertEqual(rc, 0)
        start.assert_not_called()
        self.assertEqual(calls[0][0]["model"], "dory")
        self.assertEqual(calls[0][1]["max_tokens"], 256)
        self.assertEqual(calls[0][2], 600)
        manifest = json.loads((self.root / "run" / "manifest.json").read_text())
        self.assertEqual(manifest["agent_roster"][0]["context_window"], 2048)

    def test_wrong_server_and_port_collision_fail_before_start(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        with patch("chamber.LlamaServerManager.ensure_server") as start, \
             patch("chamber.LlamaServerManager.query_server_info", return_value={"online": True, "model_id": "other"}), \
             self.assertRaisesRegex(RuntimeError, "alias"):
            main(["--config", str(config_path), "--run-dir", str(self.root / "wrong"), "--quiet"])
        start.assert_not_called()
        mixed = dict(self.config)
        mixed["agents"] = self.config["agents"] + [{"name": "B", "endpoint": "local"}]
        config_path.write_text(json.dumps(mixed))
        with patch("chamber.LlamaServerManager.ensure_server") as start, \
             patch("chamber.LlamaServerManager.query_server_info") as query, \
             self.assertRaisesRegex(ValueError, "only one llama-server"):
            main(["--config", str(config_path), "--run-dir", str(self.root / "collision"), "--quiet"])
        start.assert_not_called()
        query.assert_not_called()

    def test_cli_endpoint_override_replaces_previous_model(self):
        config_path = self.root / "config.json"
        config = dict(self.config)
        config["agents"] = [{"name": "A", "endpoint": "hermes", "model": "old/model:free"}]
        config_path.write_text(json.dumps(config))
        run_dir = self.root / "override"
        self.assertEqual(main(["--config", str(config_path), "--agent-endpoint", "A=dory",
                               "--run-dir", str(run_dir), "--mock", "--quiet"]), 0)
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["agent_roster"][0]["model"], "dory")
        self.assertEqual(manifest["agent_roster"][0]["context_window"], 3072)

    def test_oversized_dory_prompt_is_not_sent(self):
        cfg = dict(self.config)
        cfg["topic"] = "Long premise. " * 2000
        backend = MockBackend({"A": ["Should never be sent"]})
        ch = Chamber(cfg, run_dir=self.root / "oversize", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(backend.calls, [])
        self.assertTrue(any(e.type == "ERROR" and "context cap" in e.detail for e in ch.events))

    def test_dory_fallback_cannot_bypass_preflight_and_limits(self):
        cfg = dict(self.config)
        cfg["agents"] = [{"name": "A", "endpoint": "hermes", "fallback_models": ["dory"]}]
        backend = MockBackend({"A": [InferenceError("offline")]})
        ch = Chamber(cfg, run_dir=self.root / "fallback", backend=backend, quiet=True, copy_dir="").run()
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(any(e.type == "MODEL_FALLBACK" and e.status == "rejected" for e in ch.events))


if __name__ == "__main__":
    unittest.main()
