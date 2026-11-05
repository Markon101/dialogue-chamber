"""Atomic checkpointing, run manifests, state recovery, and deterministic event replay.

Provides immutable effective configuration snapshots, incremental per-turn checkpoints,
atomic file writes, interrupted run resumption, and event log replay.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def write_atomic(file_path: Path, content: str) -> None:
    """Writes content to a file atomically via a temporary file and os.replace."""
    file_path = Path(file_path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(f"{file_path.name}.tmp.{os.getpid()}_{time.monotonic_ns()}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@dataclass
class RunManifest:
    """Immutable snapshot of an experiment run's metadata and effective configuration."""
    run_id: str
    preset_name: str
    topic: str
    created_at: str
    updated_at: str
    status: str = "in_progress"               # in_progress | completed | interrupted | failed
    turns_completed: int = 0
    max_turns: int = 16
    seed: int = 42
    stop_reason: str = "not started"
    effective_config: Dict[str, Any] = field(default_factory=dict)
    agent_roster: List[Dict[str, Any]] = field(default_factory=list)
    token_budgets: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """Manages manifest creation, incremental event logging, and state checkpointing."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.run_dir / "manifest.json"
        self.checkpoint_file = self.run_dir / "checkpoint.json"
        self.events_jsonl = self.run_dir / "events.jsonl"

    def init_manifest(
        self,
        run_id: str,
        preset_name: str,
        topic: str,
        effective_config: Dict[str, Any],
        agent_roster: List[Dict[str, Any]],
        token_budgets: Dict[str, Any],
        max_turns: int,
        seed: int,
    ) -> RunManifest:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # Sanitize effective_config so secrets / API keys are never stored
        clean_cfg = self._sanitize_config(effective_config)
        clean_roster = [self._sanitize_dict(a) for a in agent_roster]

        manifest = RunManifest(
            run_id=run_id,
            preset_name=preset_name,
            topic=topic,
            created_at=now,
            updated_at=now,
            status="in_progress",
            turns_completed=0,
            max_turns=max_turns,
            seed=seed,
            effective_config=clean_cfg,
            agent_roster=clean_roster,
            token_budgets=token_budgets,
        )
        self.save_manifest(manifest)
        return manifest

    def save_manifest(self, manifest: RunManifest) -> None:
        manifest.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        content = json.dumps(manifest.to_dict(), indent=2)
        write_atomic(self.manifest_file, content)

    def load_manifest(self) -> Optional[RunManifest]:
        if not self.manifest_file.exists():
            return None
        try:
            data = json.loads(self.manifest_file.read_text(encoding="utf-8"))
            return RunManifest(**data)
        except Exception:
            return None

    def append_event(self, event_dict: Dict[str, Any]) -> None:
        """Appends an event to the events.jsonl stream."""
        line = json.dumps(event_dict, ensure_ascii=False) + "\n"
        with open(self.events_jsonl, "a", encoding="utf-8") as f:
            f.write(line)

    def save_checkpoint(self, turn: int, state_snapshot: Dict[str, Any]) -> None:
        """Atomically saves full Chamber turn state and updates manifest."""
        data = {
            "schema_version": 2,
            "turn": turn,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": state_snapshot,
        }
        write_atomic(self.checkpoint_file, json.dumps(data, indent=2))

        # Update manifest turns_completed
        manifest = self.load_manifest()
        if manifest:
            manifest.turns_completed = turn
            self.save_manifest(manifest)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Loads latest saved checkpoint state."""
        if not self.checkpoint_file.exists():
            return None
        try:
            return json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def mark_completed(self, stop_reason: str, status: str = "completed") -> None:
        manifest = self.load_manifest()
        if manifest:
            manifest.status = status
            manifest.stop_reason = stop_reason
            self.save_manifest(manifest)

    def mark_failed(self, error_message: str) -> None:
        manifest = self.load_manifest()
        if manifest:
            manifest.status = "failed"
            manifest.stop_reason = error_message
            manifest.diagnostics.append(error_message)
            self.save_manifest(manifest)

    @staticmethod
    def replay_events(events_path: Path) -> List[Dict[str, Any]]:
        """Deterministically reads previously recorded event stream without re-running inference."""
        events: List[Dict[str, Any]] = []
        if not events_path.exists():
            return events
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def _sanitize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(cfg, dict):
            return {key: self._sanitize_config(value) for key, value in cfg.items()
                    if str(key).lower() not in {"api_key", "token", "authorization"}}
        if isinstance(cfg, list):
            return [self._sanitize_config(value) for value in cfg]
        return cfg

    def _sanitize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return self._sanitize_config(d)
