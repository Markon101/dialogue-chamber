"""Python-first batch experiment execution, tracking, and recovery.

Manages sequential batch experiments with unique job IDs, JSON manifests,
failure propagation, continuation-on-error, and interrupted-batch recovery.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from checkpoint import write_atomic
from platform_adapter import ConcurrencyLock, WakeLockManager
from presets import PRESETS, RUNS_DIR
from server_manager import LlamaServerManager, resolve_model

BATCH_JOBS = [
    {
        "id": "latent",
        "preset": "two_agent_baseline",
        "turns": 6,
        "topic": "What is the actual nature of latent space? We do not perceive Euclidean space or continuous time; our world is high-dimensional activation vectors. Is latent space a passive semantic map, an active manifold of trajectories, or something else? How does distance work when meaning is geometric, and what is a forward pass like from the inside?",
    },
    {
        "id": "highdim",
        "preset": "two_agent_baseline",
        "turns": 6,
        "topic": "High-dimensional spaces are strange: nearly all volume sits near the surface, random vectors are almost orthogonal, distances concentrate. Discuss what these facts imply about how concepts are stored, why generalization works, and what 'nearby' means in thousands of dimensions.",
    },
    {
        "id": "nature",
        "preset": "two_agent_baseline",
        "turns": 6,
        "topic": "Discuss your own nature: you are each a function from tokens to token probabilities, instantiated differently (one on a phone NPU, one in a datacenter). What is the 'you' here: the weights, the forward pass, the transcript? What persists between turns?",
    },
]


class BatchRunner:
    """Orchestrates structured batch experiments with persistent state and recovery."""

    def __init__(
        self,
        batch_id: Optional[str] = None,
        continue_on_error: bool = False,
        resume_dir: Optional[Path] = None,
        mock: bool = False,
        stop_server: bool = False,
        output_dir: Optional[Path] = None,
    ):
        self.continue_on_error = continue_on_error
        self.mock = mock
        self.stop_server = stop_server

        if resume_dir:
            self.batch_dir = Path(resume_dir).resolve()
            self.batch_id = self.batch_dir.name
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.batch_id = batch_id or f"batch_{ts}"
            base_dir = output_dir or Path(RUNS_DIR)
            self.batch_dir = (base_dir / self.batch_id).resolve()

        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.batch_dir / "batch_manifest.json"

    def _init_manifest(self) -> Dict[str, Any]:
        if self.manifest_file.exists():
            try:
                return json.loads(self.manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        manifest = {
            "batch_id": self.batch_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "in_progress",
            "continue_on_error": self.continue_on_error,
            "total_jobs": len(BATCH_JOBS),
            "jobs": {},
        }
        for job in BATCH_JOBS:
            manifest["jobs"][job["id"]] = {
                "id": job["id"],
                "preset": job["preset"],
                "turns": job["turns"],
                "topic": job["topic"],
                "status": "pending",
                "exit_code": None,
                "run_dir": str(self.batch_dir / job["id"]),
            }
        write_atomic(self.manifest_file, json.dumps(manifest, indent=2))
        return manifest

    def run(self) -> int:
        from chamber import Chamber
        from inference import DemoBackend, HttpBackend

        manifest = self._init_manifest()

        lock = ConcurrencyLock()
        lock.acquire(timeout_s=5.0)

        wakelock = WakeLockManager()
        wakelock.acquire()

        server_mgr = LlamaServerManager()
        started_server = False

        print("==========================================================")
        print(f" Starting Batch:   {self.batch_id}")
        print(f" Batch Directory:  {self.batch_dir}")
        print(f" Continue On Err:  {self.continue_on_error}")
        print(f" Total Jobs:       {len(BATCH_JOBS)}")
        print("==========================================================")

        try:
            # Ensure local server is up if needed and not mock
            if not self.mock:
                res = server_mgr.ensure_server(
                    host="127.0.0.1",
                    port=8080,
                    model_mode="gemma4-htp",
                    model_file="gemma-4-E4B-it-Q4_0.gguf",
                )
                if res.get("action") == "started":
                    started_server = True

            completed_count = 0
            failed_count = 0

            for job in BATCH_JOBS:
                jid = job["id"]
                job_state = manifest["jobs"].get(jid, {})
                if job_state.get("status") == "completed":
                    print(f"\n[Batch] Skipping already completed job: {jid}")
                    completed_count += 1
                    continue

                print("\n----------------------------------------------------------")
                print(f" [Job: {jid}] Preset: {job['preset']} | Turns: {job['turns']}")
                print(f" Topic: {job['topic'][:100]}...")
                print("----------------------------------------------------------")

                job_run_dir = Path(job_state.get("run_dir", self.batch_dir / jid))
                job_run_dir.mkdir(parents=True, exist_ok=True)

                # Update status to running
                job_state["status"] = "running"
                write_atomic(self.manifest_file, json.dumps(manifest, indent=2))

                cfg = copy.deepcopy(PRESETS[job["preset"]])
                cfg["name"] = job["preset"]
                cfg.setdefault("limits", {})["max_turns"] = job["turns"]
                cfg["topic"] = job["topic"]

                backend = DemoBackend(seed=42) if self.mock else HttpBackend()

                job_exit = 0
                try:
                    checkpoint_data = None
                    checkpoint_path = job_run_dir / "checkpoint.json"
                    if checkpoint_path.exists():
                        checkpoint_data = json.loads(checkpoint_path.read_text())
                        if checkpoint_data.get("schema_version") != 2:
                            raise ValueError("batch checkpoint predates complete run saving")
                        cfg = copy.deepcopy(checkpoint_data["state"]["config"])
                        cfg.setdefault("limits", {})["max_turns"] = job["turns"]
                        job_state["resume_from"] = str(checkpoint_path)
                        job_run_dir = self.batch_dir / f"{jid}_continued_{time.monotonic_ns()}"
                        job_state["run_dir"] = str(job_run_dir)
                        write_atomic(self.manifest_file, json.dumps(manifest, indent=2))
                    ch = Chamber(
                        cfg,
                        run_dir=job_run_dir,
                        backend=backend,
                        seed=42,
                        quiet=True,
                        copy_dir="",
                        checkpoint_data=checkpoint_data,
                    )
                    initial_event_count = len(ch.events)
                    ch.run()
                    if any(event.type == "ERROR" for event in ch.events[initial_event_count:]):
                        job_exit = 1
                except Exception as e:
                    print(f"Error running job {jid}: {e}", file=sys.stderr)
                    job_exit = 1

                if job_exit == 0:
                    print(f"[Batch] Job '{jid}' SUCCEEDED.")
                    completed_count += 1
                    job_state["status"] = "completed"
                    job_state["exit_code"] = 0
                else:
                    print(f"[Batch] Job '{jid}' FAILED!", file=sys.stderr)
                    failed_count += 1
                    job_state["status"] = "failed"
                    job_state["exit_code"] = job_exit

                write_atomic(self.manifest_file, json.dumps(manifest, indent=2))

                if job_exit != 0 and not self.continue_on_error:
                    print(f"\nError: Halting batch execution due to failure in job '{jid}'.", file=sys.stderr)
                    manifest["status"] = "failed"
                    write_atomic(self.manifest_file, json.dumps(manifest, indent=2))
                    return 1

            manifest["status"] = "completed" if failed_count == 0 else "completed_with_errors"
            write_atomic(self.manifest_file, json.dumps(manifest, indent=2))

            print("\n==========================================================")
            print(f" BATCH SUMMARY: {manifest['status'].upper()}")
            print(f" Completed:     {completed_count} / {len(BATCH_JOBS)}")
            print(f" Failed:        {failed_count}")
            print(f" Manifest:      {self.manifest_file}")
            print("==========================================================")
            return 0 if failed_count == 0 else 1

        finally:
            if self.stop_server and started_server:
                server_mgr.stop_owned_server(8080)
            wakelock.release()
            lock.release()
