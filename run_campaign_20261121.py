#!/usr/bin/env python3
"""Sequential 8-hour compute campaign runner for Qwen3-TTS + bonus experiments."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from checkpoint import write_atomic

REPO_DIR = Path(__file__).resolve().parent
RUNS_DIR = REPO_DIR / "runs"
REPORT_MD = RUNS_DIR / "campaign_20261121_report.md"
SUMMARY_JSON = RUNS_DIR / "campaign_20261121_summary.json"

PHASE1_JOBS: List[Tuple[str, str]] = [
    ("qwen3_tts_r1_studio_4a", "experiments/qwen3_tts/r1_studio_opera_4a.json"),
    ("qwen3_tts_r2_duel_5a", "experiments/qwen3_tts/r2_codec_war_5a.json"),
    ("qwen3_tts_r3_asymmetric_6a", "experiments/qwen3_tts/r3_blind_pipeline_6a.json"),
    ("qwen3_tts_r4_octet_100t", "experiments/qwen3_tts/r4_octet_symphony_100t.json"),
]

PHASE2_JOBS: List[Tuple[str, str]] = [
    ("bonus_b1_covert_saboteur_5a", "experiments/bonus_2026_11_21/b1_covert_saboteur.json"),
    ("bonus_b2_klein_atlas_5a", "experiments/bonus_2026_11_21/b2_klein_bottle_atlas.json"),
    ("bonus_b3_memetic_6a", "experiments/bonus_2026_11_21/b3_memetic_phylogeny.json"),
    ("bonus_b4_febreze_society_8a", "experiments/bonus_2026_11_21/b4_febreze_local_society.json"),
]

PHASE1_HARD_CAP_S = 7.5 * 3600
PHASE2_START_CUTOFF_S = 6.5 * 3600


def check_llama_server(url: str = "http://127.0.0.1:8080/v1/models", retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            if attempt + 1 < retries:
                time.sleep(10)
    return False


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def summarize_run(run_id: str, cfg_path: str, phase: str, status: str, exit_code: int, wall_s: float) -> Dict[str, Any]:
    rdir = RUNS_DIR / run_id
    manifest = load_json_if_exists(rdir / "manifest.json")
    metrics = load_json_if_exists(rdir / "metrics.json")
    arts_dir = rdir / "artifacts"
    artifacts = []
    if arts_dir.exists():
        for p in sorted(arts_dir.iterdir()):
            if p.is_file():
                artifacts.append({"name": p.name, "bytes": p.stat().st_size, "path": str(p)})
    return {
        "run_id": run_id,
        "phase": phase,
        "config": cfg_path,
        "status": status,
        "manifest_status": manifest.get("status"),
        "exit_code": exit_code,
        "wall_s": round(wall_s, 2),
        "turns_completed": metrics.get("turns_completed"),
        "max_turns": manifest.get("max_turns"),
        "stop_reason": metrics.get("stop_reason") or manifest.get("stop_reason"),
        "deliverable_required": metrics.get("deliverable_required"),
        "deliverable_verified": metrics.get("deliverable_verified"),
        "deliverable_issues": metrics.get("deliverable_issues"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "completion_tokens": metrics.get("completion_tokens"),
        "total_tokens": metrics.get("total_tokens"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "elapsed_s": metrics.get("elapsed_s"),
        "accepted_actions": metrics.get("accepted_actions"),
        "rejected_actions": metrics.get("rejected_actions"),
        "sampling_changes_count": len(metrics.get("sampling_changes") or []),
        "sampling_changes": metrics.get("sampling_changes") or [],
        "artifacts": artifacts,
    }


def write_campaign_reports(results: List[Dict[str, Any]], campaign_start: float, notes: List[str]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    elapsed_total = round(time.time() - campaign_start, 2)
    payload = {
        "started_at_epoch": campaign_start,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_total_s": elapsed_total,
        "results": results,
        "notes": notes,
    }
    write_atomic(SUMMARY_JSON, json.dumps(payload, indent=2))

    lines = [
        "# Campaign 2026-11-21 — Qwen3-TTS & Bonus Experiments",
        "",
        f"- **Updated (UTC)**: `{payload['updated_at_utc']}`",
        f"- **Elapsed total**: `{elapsed_total}s` ({elapsed_total / 3600:.2f}h)",
        "",
        "| Run ID | Phase | Status | Turns | Verified | Tokens | Tok/s | Elapsed(s) | Actions (Acc/Rej) | Sampling Δ | Stop Reason |",
        "|---|---|---|---:|---|---:|---:|---:|---|---:|---|",
    ]
    for r in results:
        turns = f"{r.get('turns_completed')}/{r.get('max_turns')}" if r.get("turns_completed") is not None else "N/A"
        ver = str(r.get("deliverable_verified")) if r.get("deliverable_required") else "N/A (voluntary)"
        acc_rej = f"{r.get('accepted_actions', 0)}/{r.get('rejected_actions', 0)}"
        lines.append(
            f"| `{r['run_id']}` | {r['phase']} | `{r['status']}` | {turns} | {ver} | "
            f"{r.get('total_tokens', 'N/A')} | {r.get('tokens_per_second', 'N/A')} | "
            f"{r.get('elapsed_s', r.get('wall_s'))} | {acc_rej} | {r.get('sampling_changes_count', 0)} | "
            f"{str(r.get('stop_reason', '')).replace('|', '/')} |"
        )
    if notes:
        lines.extend(["", "## Campaign Notes", ""])
        for n in notes:
            lines.append(f"- {n}")
    write_atomic(REPORT_MD, "\n".join(lines) + "\n")


def run_job(run_id: str, cfg_path: str, phase: str) -> Dict[str, Any]:
    rdir = RUNS_DIR / run_id
    manifest = load_json_if_exists(rdir / "manifest.json")
    if manifest.get("status") == "completed":
        print(f"[Campaign] Skipping already completed run: {run_id}", flush=True)
        return summarize_run(run_id, cfg_path, phase, "skipped_completed", 0, 0.0)

    if not check_llama_server():
        print(f"[Campaign] ERROR: llama-server unreachable before {run_id}", file=sys.stderr, flush=True)
        return summarize_run(run_id, cfg_path, phase, "server_unreachable", 2, 0.0)

    print(f"\n[Campaign] === Starting {phase}: {run_id} ({cfg_path}) ===", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "chamber.py", "--config", cfg_path, "--run-id", run_id],
        cwd=str(REPO_DIR),
    )
    wall_s = time.monotonic() - t0
    status = "completed" if proc.returncode == 0 else "failed"
    print(f"[Campaign] === Finished {run_id}: status={status}, exit={proc.returncode}, wall={wall_s:.1f}s ===", flush=True)
    return summarize_run(run_id, cfg_path, phase, status, proc.returncode, wall_s)


def main() -> int:
    campaign_start = time.time()
    results: List[Dict[str, Any]] = []
    notes: List[str] = []

    for run_id, cfg_path in PHASE1_JOBS:
        elapsed = time.time() - campaign_start
        if elapsed >= PHASE1_HARD_CAP_S:
            notes.append(f"Skipped {run_id} because hard cap ({PHASE1_HARD_CAP_S}s) was reached.")
            break
        res = run_job(run_id, cfg_path, "Phase 1 (Qwen3-TTS)")
        results.append(res)
        write_campaign_reports(results, campaign_start, notes)
        if res["status"] == "server_unreachable":
            notes.append(f"Aborted campaign at {run_id}: local llama-server unreachable.")
            write_campaign_reports(results, campaign_start, notes)
            return 2

    for run_id, cfg_path in PHASE2_JOBS:
        elapsed = time.time() - campaign_start
        if elapsed >= PHASE2_START_CUTOFF_S:
            notes.append(f"Stopped before Phase 2 job {run_id}: elapsed {elapsed:.0f}s >= cutoff {PHASE2_START_CUTOFF_S:.0f}s.")
            break
        res = run_job(run_id, cfg_path, "Phase 2 (Bonus)")
        results.append(res)
        write_campaign_reports(results, campaign_start, notes)
        if res["status"] == "server_unreachable":
            notes.append(f"Aborted Phase 2 at {run_id}: local llama-server unreachable.")
            write_campaign_reports(results, campaign_start, notes)
            return 2

    print("\n[Campaign] Running Anomaly Laboratory scan across runs...", flush=True)
    anom = subprocess.run([sys.executable, "chamber.py", "--anomalies", "runs"], cwd=str(REPO_DIR))
    notes.append(f"Anomaly Laboratory scan finished with exit code {anom.returncode}.")
    write_campaign_reports(results, campaign_start, notes)
    print(f"[Campaign] All done! Summary written to {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
