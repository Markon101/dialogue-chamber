#!/usr/bin/env python3
"""R1 benchmark: Lantern mystery, round_robin vs self_directed scheduling, homogeneous local model.

solved_turn comes from the preset's regex answer_key, which is a text-matching PROXY for
"the agents reached the answer" (it can false-positive on a mention of the culprit).
Results are written to runs/bench_lantern_<ts>/results.json (+ per-run directories).
"""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

from chamber import Chamber
from inference import DemoBackend, HttpBackend
from platform_adapter import ConcurrencyLock, WakeLockManager
from presets import PRESETS, RUNS_DIR
from server_manager import LlamaServerManager, resolve_model

SCHEDULERS = ("round_robin", "self_directed")


def build_cfg(scheduler, turns, max_tokens, reasoning, run_id, model_name=None):
    cfg = copy.deepcopy(PRESETS["information_asymmetry"])
    cfg["name"] = "bench_lantern"
    cfg["run_id"] = run_id
    cfg["scheduler"] = scheduler
    cfg.setdefault("limits", {}).update(max_turns=turns, max_requests=turns * 3, max_duration_s=1800)
    for a in cfg["agents"]:  # homogeneous: every agent on the same local model, same protocol
        a.update(endpoint="local", profile="standard", capabilities="standard",
                 max_tokens=max_tokens, reasoning=reasoning)
        if model_name:
            a["model"] = model_name
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--reasoning", default="off")
    ap.add_argument("--model", default="gemma4-htp")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop the server if we started it")
    args = ap.parse_args()

    out = Path(RUNS_DIR) / f"bench_lantern_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    lock, wake, sm, started = ConcurrencyLock(), WakeLockManager(), LlamaServerManager(), False
    lock.acquire(timeout_s=5)
    wake.acquire()
    results = []
    try:
        active_model_name = None
        if not args.mock:
            mode, mfile, _ = resolve_model(args.model)
            res = sm.ensure_server(model_mode=mode, model_file=mfile)
            started = res["action"] == "started"
            active_model_name = res.get("model_name") or mfile
            print(f"[bench] server {res['action']}: {active_model_name}")
        for trial in range(args.trials):
            for sched in SCHEDULERS:
                rid = f"t{trial}_{sched}"
                cfg = build_cfg(sched, args.turns, args.max_tokens, args.reasoning, rid, model_name=active_model_name)
                backend = DemoBackend(seed=trial) if args.mock else HttpBackend()
                t0 = time.monotonic()
                try:
                    ch = Chamber(cfg, run_dir=out / rid, backend=backend, seed=1000 + trial,
                                 quiet=True, copy_dir="")
                    ch.run()
                    m = ch.compute_metrics()
                    row = dict(trial=trial, scheduler=sched, ok=True,
                               solved_turn=m["solved_turn"],
                               verified_synthesis_turn=m.get("verified_synthesis_turn"),
                               turns=m["turns_completed"], completion_tokens=m["completion_tokens"],
                               prompt_tokens=m["prompt_tokens"], rejected_actions=m["rejected_actions"],
                               attribution_flags=m.get("attribution_flag_count", 0),
                               truncated=m["truncated_generations"], stop_reason=m["stop_reason"],
                               wall_s=round(time.monotonic() - t0, 1))
                except Exception as e:  # record, don't hide
                    row = dict(trial=trial, scheduler=sched, ok=False, error=f"{type(e).__name__}: {e}")
                results.append(row)
                print("[bench]", json.dumps(row), flush=True)
                (out / "results.json").write_text(json.dumps(results, indent=2))
    finally:
        if args.stop and started:
            sm.stop_owned_server(8080)
        wake.release()
        lock.release()

    print("\nSUMMARY (solved_turn is regex proxy; verified_synthesis_turn requires all clue-holders + non-list match)")
    for sched in SCHEDULERS:
        rows = [r for r in results if r["scheduler"] == sched and r["ok"]]
        solved = [r["solved_turn"] for r in rows if r["solved_turn"] is not None]
        synth = [r["verified_synthesis_turn"] for r in rows if r.get("verified_synthesis_turn") is not None]
        med = statistics.median(solved) if solved else None
        med_synth = statistics.median(synth) if synth else None
        print(f"  {sched:14s} runs={len(rows)} regex_solved={len(solved)}/{len(rows)} (med={med}) verified_synth={len(synth)}/{len(rows)} (med={med_synth})")
    print(f"results: {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
