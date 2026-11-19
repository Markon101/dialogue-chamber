import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chamber import Chamber, main
from context import ContextIntelligence
from inference import MockBackend, InferenceError
from protocol import repair_unambiguous_json
from server_manager import LlamaServerManager, resolve_model
from batch_runner import BatchRunner


class ReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"name": "review", "agents": [{"name": name, "endpoint": "local"}
                                                     for name in ("A", "B", "C")],
                       "limits": {"max_turns": 1}}

    def chamber(self, name="run"):
        return Chamber(self.config, run_dir=self.root / name, backend=MockBackend(default="Hello"),
                       quiet=True, copy_dir="")

    def test_run_collision_preserves_existing_manifest(self):
        chamber = self.chamber().run()
        before = chamber.checkpoints.manifest_file.read_bytes()
        with self.assertRaisesRegex(ValueError, "already contains a run"):
            self.chamber()
        self.assertEqual(chamber.checkpoints.manifest_file.read_bytes(), before)

    def test_reactive_scheduler_excludes_finished_and_ineligible_agents(self):
        for mode in ("event_driven", "self_directed"):
            chamber = self.chamber(mode)
            chamber.scheduler_mode = mode
            chamber.agents["C"].active = False
            chamber.agents["B"].consecutive_turns = chamber.limits["max_consecutive"]
            chamber.reactive_queue = ["C", "B"]
            self.assertEqual(chamber._select_next_speaker("A"), "A")

    def test_repair_preserves_literals_inside_strings(self):
        source = '{"message":"literal, } and ``` stay intact", "actions":[],}'
        parsed = json.loads(repair_unambiguous_json(source))
        self.assertEqual(parsed["message"], "literal, } and ``` stay intact")

    def test_newest_oversized_message_is_omitted(self):
        messages, accounting = ContextIntelligence().assemble_and_budget(
            agent_name="A", endpoint={"provider": "mock"}, system_sections={"rules": "Keep rules"},
            fixed_sections={}, public_messages=[{"id": "m1", "content": "X" * 10000}],
            effective_context=1024, context_source="config", reserved_output=256, safety_margin=128)
        self.assertNotIn("X" * 100, messages[1]["content"])
        self.assertNotIn("opening the conversation", messages[1]["content"])
        self.assertEqual(accounting.omitted_messages_count, 1)
        self.assertTrue(accounting.fits_in_context)

    def test_cli_sampling_flags_reach_saved_runtime(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        run_dir = self.root / "cli"
        with patch("chamber.COPY_DIR", ""):
            self.assertEqual(main(["--config", str(config_path), "--run-dir", str(run_dir),
                                   "--temperature", "0.3", "--reasoning", "off", "--mock", "--quiet"]), 0)
        saved = json.loads((run_dir / "checkpoint.json").read_text())["state"]
        self.assertEqual(saved["agents"]["A"]["temperature"], 0.3)
        self.assertTrue(all(agent["reasoning"] == "off" for agent in saved["config"]["agents"]))

    def test_explicit_cpu_mtp_mode_is_not_lost_during_cli_resolution(self):
        config_path = self.root / "config.json"
        self.config["context_tokens"] = 2048
        config_path.write_text(json.dumps(self.config))
        with patch("chamber.IS_TERMUX", True), \
             patch("chamber.assert_single_llama_server", return_value=[]), \
             patch("chamber.LlamaServerManager.ensure_server", return_value={"action": "reused"}) as ensure, \
             patch("chamber.Chamber"):
            self.assertEqual(main(["--config", str(config_path), "--model", "gemma4-cpu-mtp", "--quiet"]), 0)
        self.assertEqual(ensure.call_args.kwargs["model_mode"], "gemma4-cpu-mtp")
        self.assertEqual(ensure.call_args.kwargs["ctx_size"], 2048)
        self.assertEqual(resolve_model("gemma-4-E4B-it-Q4_0.gguf")[0], "gemma4-htp")

    def test_reused_pid_is_not_signalled(self):
        manager = LlamaServerManager(state_dir=self.root)
        manager._state_file(8080).write_text(json.dumps({"pid": 123, "start_ticks": "1"}))
        with patch("server_manager.os.kill"), patch("server_manager.os.killpg") as kill_group, \
             patch("server_manager.process_start_ticks", return_value="2"):
            self.assertFalse(manager.stop_owned_server())
        kill_group.assert_not_called()

    def test_same_model_with_wrong_server_mode_is_not_reused(self):
        manager = LlamaServerManager(state_dir=self.root)
        manager._state_file(8080).write_text(json.dumps({"mode": "gemma4-htp"}))
        with patch("server_manager.assert_single_llama_server", return_value=[(123, "llama-server")]), \
             patch.object(manager, "check_health", return_value=True), \
             patch.object(manager, "is_server_owned", return_value=True), \
             patch.object(manager, "query_server_info", return_value={"model_name": "gemma.gguf"}), \
             patch("server_manager.subprocess.Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "mode differs"):
                manager.ensure_server(model_mode="gemma4-htp-mtp", model_file="gemma.gguf", log_file=self.root / "server.log")
        launch.assert_not_called()

    def test_fallback_cannot_exceed_request_budget(self):
        chamber = self.chamber()
        chamber.limits["max_requests"] = 1
        chamber.agents["A"].fallback_models = ["hermes"]
        chamber.backend = MockBackend({"A": [InferenceError("offline", transient=False), "must not run"]})
        with self.assertRaisesRegex(InferenceError, "request budget"):
            chamber._call_model(chamber.agents["A"], [])
        self.assertEqual(len(chamber.backend.calls), 1)

    def test_small_remaining_token_budget_is_respected(self):
        chamber = self.chamber()
        chamber.limits["max_total_tokens"] = 5
        chamber._call_model(chamber.agents["A"], [])
        self.assertEqual(chamber.backend.calls[0][1]["max_tokens"], 5)

    def test_batch_continuation_preserves_source_and_artifacts(self):
        job = {"id": "one", "preset": "two_agent_baseline", "turns": 2, "topic": "Continue"}
        with patch("batch_runner.BATCH_JOBS", [job]):
            runner = BatchRunner(batch_id="batch", mock=True, output_dir=self.root)
            runner._init_manifest()
            config = copy.deepcopy(self.config)
            config["artifacts"] = {"seed": {"world": "Preserve this ledger"}, "visible": ["world"]}
            original = Chamber(config, run_dir=runner.batch_dir / "one", backend=MockBackend(default="Hello"),
                               quiet=True, copy_dir="").run()
            original_bytes = original.checkpoints.checkpoint_file.read_bytes()
            self.assertEqual(runner.run(), 0)
            manifest = json.loads(runner.manifest_file.read_text())
            destination = Path(manifest["jobs"]["one"]["run_dir"])
            self.assertNotEqual(destination, original.run_dir)
            self.assertEqual(original_bytes, original.checkpoints.checkpoint_file.read_bytes())
            self.assertIn("Preserve this ledger", (destination / "artifacts/world.md").read_text())
            self.assertEqual(json.loads((destination / "checkpoint.json").read_text())["turn"], 2)

    def test_sampler_flags_protocol_and_provider_gating(self):
        from protocol import parse_reply
        parsed = parse_reply(
            "Testing chaotic samplers.\n"
            "@SET_XTC 0.6 0.15\n"
            "@SET_SAMPLER min_p=0.08 top_k=40 repeat_penalty=1.15\n",
            names=("A", "B"),
        )
        self.assertEqual(len(parsed.actions), 2)
        self.assertEqual(parsed.actions[0], {"type": "SET_SAMPLING", "params": {"xtc_probability": 0.6, "xtc_threshold": 0.15}})
        self.assertEqual(parsed.actions[1], {"type": "SET_SAMPLING", "params": {"min_p": 0.08, "top_k": 40.0, "repeat_penalty": 1.15}})

        config_path = self.root / "sampler_cfg.json"
        cfg = {
            "name": "samplers",
            "agents": [
                {"name": "LOC", "endpoint": "local", "capabilities": "standard"},
                {"name": "REM", "endpoint": "hermes", "capabilities": "standard"},
            ],
            "limits": {"max_turns": 1},
        }
        config_path.write_text(json.dumps(cfg))
        run_dir = self.root / "sampler_run"
        with patch("chamber.COPY_DIR", ""):
            self.assertEqual(
                main([
                    "--config", str(config_path), "--run-dir", str(run_dir),
                    "--min-p", "0.06", "--xtc-probability", "0.5", "--xtc-threshold", "0.1",
                    "--agent-sampler", "LOC=top_k=50,repeat_penalty=1.12",
                    "--mock", "--quiet",
                ]),
                0,
            )
        saved = json.loads((run_dir / "checkpoint.json").read_text())["state"]["agents"]
        self.assertEqual(saved["LOC"]["min_p"], 0.06)
        self.assertEqual(saved["LOC"]["top_k"], 50)
        self.assertEqual(saved["LOC"]["xtc_probability"], 0.5)
        self.assertEqual(saved["LOC"]["repeat_penalty"], 1.12)

        from inference import build_payload
        loc_payload, _ = build_payload({"provider": "llamacpp", "model": "x"}, saved["LOC"], [{"role": "user", "content": "hi"}], 0.9, 64)
        rem_payload, _ = build_payload({"provider": "openrouter", "model": "x"}, saved["REM"], [{"role": "user", "content": "hi"}], 0.9, 64)
        self.assertEqual(loc_payload["xtc_probability"], 0.5)
        self.assertEqual(loc_payload["repeat_penalty"], 1.12)
        self.assertEqual(loc_payload["min_p"], 0.06)
        self.assertEqual(rem_payload["min_p"], 0.06)
        self.assertNotIn("xtc_probability", rem_payload)
        self.assertNotIn("repeat_penalty", rem_payload)

    def test_legacy_sampler_commands_use_atomic_validation(self):
        cfg = {"name": "s", "agents": [{"name": "A", "endpoint": "local", "capabilities": "standard"}], "limits": {"max_turns": 1}}
        backend = MockBackend({"A": ["Cranking XTC.\n@SET_XTC 0.8 0.9\n@SET_MIN_P 0.07"]})
        ch = Chamber(cfg, run_dir=self.root / "act", backend=backend, quiet=True, copy_dir="").run()
        self.assertIsNone(ch.agents["A"].xtc_probability)
        self.assertIsNone(ch.agents["A"].xtc_threshold)
        self.assertEqual(ch.agents["A"].min_p, 0.07)


if __name__ == "__main__":
    unittest.main()
