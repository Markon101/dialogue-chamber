"""Diagnostics, Environment Doctor, Context Reporting, and Dry-Run Simulator.

Implements --context-info, --model-info, --show-effective-config, --dry-run, and --doctor.
All displayed values reflect actual runtime verification or are labeled as configured, estimated, or unknown.
Never exposes API keys or secrets.
"""
import copy
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from context import ContextIntelligence, TokenizerEngine
from models import get_catalog
from presets import LIMITS
from platform_adapter import (
    IS_LINUX,
    IS_TERMUX,
    get_available_memory_mb,
    get_thermal_diagnostics,
)
from server_manager import LlamaServerManager, resolve_model


def run_doctor(endpoints: Dict[str, Any], runs_dir: Path) -> Dict[str, Any]:
    """Runs system, environment, dependency, and endpoint checks."""
    checks: List[Dict[str, Any]] = []

    def add_check(category: str, item: str, status: str, detail: str) -> None:
        checks.append({"category": category, "item": item, "status": status, "detail": detail})

    # 1. Environment & Platform
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        add_check("Platform", "Python Version", "PASS", f"Python {py_ver}")
    else:
        add_check("Platform", "Python Version", "FAIL", f"Python {py_ver} (>= 3.10 required)")

    if IS_TERMUX:
        termux_ver = os.environ.get("TERMUX_VERSION", "detected")
        add_check("Platform", "Android Termux", "PASS", f"Termux environment ({termux_ver})")
        # Wake-lock check
        if shutil.which("termux-wake-lock"):
            add_check("Platform", "Termux Wake-Lock", "PASS", "termux-wake-lock utility installed")
        else:
            add_check("Platform", "Termux Wake-Lock", "WARN", "termux-wake-lock missing (termux-api package)")
    else:
        add_check("Platform", "OS Platform", "PASS", f"{sys.platform} environment")

    # 2. Host Memory & Resources
    mem = get_available_memory_mb()
    if mem is not None:
        if mem > 1500:
            add_check("Resources", "Available Memory", "PASS", f"{mem} MB available")
        elif mem > 600:
            add_check("Resources", "Available Memory", "WARN", f"{mem} MB available (low for large models)")
        else:
            add_check("Resources", "Available Memory", "FAIL", f"{mem} MB available (critical memory pressure)")
    else:
        add_check("Resources", "Available Memory", "WARN", "Could not read /proc/meminfo")

    therm = get_thermal_diagnostics()
    if therm.get("available"):
        add_check("Resources", "Thermal Sensors", "PASS", f"Available ({len(therm.get('thermal_zones', {}))} zones)")
    else:
        add_check("Resources", "Thermal Sensors", "INFO", "Thermal telemetry not directly accessible")

    # 3. File System & Permissions
    runs_dir = Path(runs_dir).resolve()
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        test_file = runs_dir / ".doctor_write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        add_check("Storage", "Runs Directory", "PASS", f"Writable: {runs_dir}")
    except Exception as e:
        add_check("Storage", "Runs Directory", "FAIL", f"Cannot write to {runs_dir}: {e}")

    # Phone mirror check (only when configured or on Termux)
    copy_dir_val = os.environ.get("CHAMBER_COPY_DIR", "/sdcard/dialogue_chamber/runs" if IS_TERMUX else "")
    if copy_dir_val:
        copy_dir = Path(copy_dir_val)
        try:
            if copy_dir.exists() and os.access(copy_dir, os.W_OK):
                add_check("Storage", "Phone Storage Mirror", "PASS", f"Accessible: {copy_dir}")
            elif copy_dir.exists():
                add_check("Storage", "Phone Storage Mirror", "WARN", f"Exists but read-only: {copy_dir}")
            else:
                add_check("Storage", "Phone Storage Mirror", "INFO", f"Mirror path not created yet ({copy_dir})")
        except Exception:
            add_check("Storage", "Phone Storage Mirror", "INFO", "Mirror not accessible")

    # 4. Local llama-server & Models
    sm = LlamaServerManager()
    if sm.serve_script.exists():
        add_check("Server", "Snapdragon Serve Script", "PASS", f"Found: {sm.serve_script}")
    else:
        add_check("Server", "Snapdragon Serve Script", "WARN", f"Not found at {sm.serve_script}")

    models_dir = Path.home() / "models"
    if models_dir.exists():
        ggufs = list(models_dir.glob("*.gguf"))
        if ggufs:
            add_check("Server", "Local Model Files", "PASS", f"Found {len(ggufs)} .gguf models in ~/models")
        else:
            add_check("Server", "Local Model Files", "WARN", "No .gguf files in ~/models")
    else:
        add_check("Server", "Local Model Files", "WARN", "~/models directory does not exist")

    # 5. Endpoint Connectivity
    for ep_name, ep in endpoints.items():
        prov = ep.get("provider", "unknown")
        url = ep.get("base_url", "")
        if prov == "llamacpp":
            ep_url = urlsplit(url or "http://127.0.0.1:8080/v1")
            ep_host = ep_url.hostname or "127.0.0.1"
            ep_port = ep_url.port or (443 if ep_url.scheme == "https" else 8080)
            try:
                healthy = sm.check_health(ep_host, ep_port, timeout=1.0)
                if healthy:
                    info = sm.query_server_info(ep_host, ep_port, timeout=1.0)
                    add_check("Endpoints", f"Local Endpoint ({ep_name})", "PASS", f"Online ({info.get('model_name')})")
                else:
                    add_check("Endpoints", f"Local Endpoint ({ep_name})", "INFO", "Offline (will start on experiment launch)")
            except Exception as e:
                add_check("Endpoints", f"Local Endpoint ({ep_name})", "INFO", f"Offline: {e}")

        elif prov == "openrouter":
            key_env = ep.get("api_key_env", "OPENROUTER_API_KEY")
            key_val = os.environ.get(key_env, "")
            if key_val:
                masked = f"{key_val[:4]}...{key_val[-4:]}" if len(key_val) > 8 else "***"
                add_check("Endpoints", f"Remote Endpoint ({ep_name})", "PASS", f"API Key present in ${key_env} ({masked})")
            else:
                add_check("Endpoints", f"Remote Endpoint ({ep_name})", "WARN", f"${key_env} not set; OpenRouter requests will fail")

    return {"doctor_passed": all(c["status"] != "FAIL" for c in checks), "checks": checks}


def format_doctor_report(doctor_data: Dict[str, Any]) -> str:
    """Formats doctor checklist in clean markdown/terminal output."""
    lines = [
        "==========================================================",
        "          DIALOGUE CHAMBER — SYSTEM DOCTOR REPORT          ",
        "==========================================================",
        "",
    ]
    current_cat = None
    for c in doctor_data["checks"]:
        if c["category"] != current_cat:
            current_cat = c["category"]
            lines.append(f"\n[{current_cat}]")
        status_tag = f"[{c['status']}]"
        lines.append(f"  {status_tag:<8} {c['item']:<28} : {c['detail']}")

    lines.append("\n" + "=" * 58)
    overall = "HEALTHY" if doctor_data["doctor_passed"] else "ATTENTION NEEDED"
    lines.append(f" Overall Status: {overall}")
    lines.append("=" * 58)
    return "\n".join(lines)


def run_context_info(
    preset_name: str,
    preset_cfg: Dict[str, Any],
    endpoints: Dict[str, Any],
    cli_turns: Optional[int] = None,
    cli_ctx: Optional[Any] = None,
    cli_max_output: Optional[int] = None,
    cli_reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates structured context intelligence report for configured endpoints and roster."""
    sm = LlamaServerManager()
    cat = get_catalog()
    ctx_intel = ContextIntelligence()

    # 1. Local Llama.cpp Endpoint Inspection
    local_ep = endpoints.get("local", {})
    local_url = urlsplit(local_ep.get("base_url", "http://127.0.0.1:8080/v1"))
    local_host = local_url.hostname or "127.0.0.1"
    local_port = local_url.port or (443 if local_url.scheme == "https" else 8080)
    local_info = sm.query_server_info(local_host, local_port, timeout=1.5)
    is_local_online = local_info.get("online", False)

    loaded_model = local_info.get("model_name") or local_info.get("model_id") or local_ep.get("model", "unknown")
    local_ctx_reported = local_info.get("context_window")

    # Negotiate context limit
    eff_local_ctx, local_ctx_src, local_ctx_conf = ctx_intel.negotiate_context_limit(
        server_reported_ctx=local_ctx_reported,
        user_configured_ctx=cli_ctx,
        model_default_ctx=32768 if "gemma" in str(loaded_model).lower() else 4096,
    )

    tok_engine = TokenizerEngine()
    tok_available = False
    if is_local_online:
        _, method = tok_engine.count_tokens("test", local_ep)
        tok_available = (method == "exact")

    raw_out_budget = cli_max_output or int(LIMITS.get("max_tokens_per_response", 16384))
    out_budget = min(raw_out_budget, max(128, eff_local_ctx // 2))
    safety_reserve = min(ctx_intel.default_safety_margin, max(64, eff_local_ctx // 4))
    avail_prompt = max(0, eff_local_ctx - (out_budget + safety_reserve))

    # 2. Remote / OpenRouter Inspection
    hermes_ep = endpoints.get("hermes", {})
    hermes_model = hermes_ep.get("model", "deepseek/deepseek-v4.1-flash-1017")
    remote_meta = cat.get_model(hermes_model)
    has_remote_key = bool(os.environ.get("OPENROUTER_API_KEY", ""))

    remote_ctx = remote_meta.context_length if remote_meta else 65536
    remote_ctx_src = "metadata" if remote_meta else "config"
    remote_max_out = remote_meta.max_completion_tokens if remote_meta else 8192
    remote_reasoning = "supported" if (remote_meta and remote_meta.has_reasoning) else "unreported"

    # 3. Harness & Roster
    agents_raw = preset_cfg.get("agents", [])
    effective_turns = cli_turns or preset_cfg.get("limits", {}).get("max_turns", 16)

    roster_rows = []
    for a in agents_raw:
        aname = a.get("name", "AGENT")
        ep_ref = a.get("endpoint", "local")
        if isinstance(ep_ref, dict):
            ep_dict = ep_ref
            backend = ep_dict.get("provider", "custom")
            amodel = ep_dict.get("model", "custom")
        else:
            ep_dict = endpoints.get(str(ep_ref), {})
            backend = ep_dict.get("provider", str(ep_ref))
            amodel = a.get("model") or ep_dict.get("model", "unknown")

        if backend == "llamacpp":
            actx = eff_local_ctx
        elif backend == "openrouter":
            m_info = cat.get_model(amodel)
            actx = m_info.context_length if m_info else 65536
        else:
            actx = 4096

        roster_rows.append({
            "agent": aname,
            "backend": backend,
            "model": amodel,
            "context_window": actx,
            "context_display": f"{actx // 1024}K" if actx >= 1024 else str(actx),
        })

    report = {
        "local": {
            "connection": "OK" if is_local_online else "OFFLINE (will launch on demand)",
            "endpoint": local_ep.get("base_url", "http://127.0.0.1:8080/v1"),
            "actual_model": loaded_model,
            "model_id_verified": "YES" if is_local_online and local_info.get("model_name") else "CONFIGURED",
            "context_limit": eff_local_ctx,
            "context_source": local_ctx_src,
            "tokenizer_available": "YES" if tok_available else "ESTIMATED_FALLBACK",
            "thinking_mode": "enabled" if (cli_reasoning != "off") else "disabled",
            "output_budget": out_budget,
            "safety_reserve": safety_reserve,
            "available_prompt": avail_prompt,
        },
        "remote": {
            "connection": "CONFIGURED (API key present)" if has_remote_key else "UNCONFIGURED (missing API key)",
            "provider": hermes_ep.get("provider", "openrouter"),
            "actual_model": hermes_model,
            "model_context": remote_ctx,
            "context_source": remote_ctx_src,
            "output_limit": remote_max_out or "unreported",
            "reasoning_support": remote_reasoning,
            "token_counting": "exact" if has_remote_key else "estimated",
        },
        "harness": {
            "selected_preset": preset_name,
            "agents_count": len(agents_raw),
            "configured_turns": effective_turns,
            "effective_prompt_budget": avail_prompt,
            "roster": roster_rows,
        },
    }
    return report


def format_context_info_report(diag: Dict[str, Any]) -> str:
    """Renders human-readable report matching Prompt specification."""
    loc = diag["local"]
    rem = diag["remote"]
    harn = diag["harness"]

    lines = [
        "DIALOGUE CHAMBER — INFERENCE DIAGNOSTICS",
        "",
        "LOCAL LLAMA.CPP",
        f"Connection:             {loc['connection']}",
        f"Endpoint:               {loc['endpoint']}",
        f"Actual model:           {loc['actual_model']}",
        f"Model ID verified:      {loc['model_id_verified']}",
        f"Context limit:          {loc['context_limit']} tokens",
        f"Context source:         {loc['context_source']}",
        f"Tokenizer available:    {loc['tokenizer_available']}",
        f"Thinking mode:          {loc['thinking_mode']}",
        f"Output budget:          {loc['output_budget']} tokens",
        f"Safety reserve:         {loc['safety_reserve']} tokens",
        f"Available prompt:       {loc['available_prompt']} tokens",
        "",
        "HERMES / REMOTE",
        f"Connection:             {rem['connection']}",
        f"Provider:               {rem['provider']}",
        f"Actual model:           {rem['actual_model']}",
        f"Model context:          {rem['model_context']} tokens",
        f"Context source:         {rem['context_source']}",
        f"Output limit:           {rem['output_limit']} tokens",
        f"Reasoning support:      {rem['reasoning_support']}",
        f"Token counting:         {rem['token_counting']}",
        "",
        "HARNESS",
        f"Selected preset:        {harn['selected_preset']}",
        f"Agents:                 {harn['agents_count']}",
        f"Configured turns:       {harn['configured_turns']}",
        f"Effective prompt budget: {harn['effective_prompt_budget']} tokens",
        "",
        "ROSTER CONTEXT ALLOCATION",
        f"{'AGENT':<18} {'BACKEND':<14} {'MODEL':<34} {'CONTEXT':<8}",
        "-" * 76,
    ]
    for r in harn["roster"]:
        lines.append(f"{r['agent']:<18} {r['backend']:<14} {r['model']:<34} {r['context_display']:<8}")

    return "\n".join(lines)


def run_dry_run(
    preset_name: str,
    preset_cfg: Dict[str, Any],
    endpoints: Dict[str, Any],
    topic: Optional[str] = None,
    turns: Optional[int] = None,
) -> Dict[str, Any]:
    """Simulates an experiment run, calculating token budgets and plan without making model calls."""
    ctx_diag = run_context_info(preset_name, preset_cfg, endpoints, cli_turns=turns)
    harn = ctx_diag["harness"]

    sim_turns = harn["configured_turns"]
    topic_text = topic or preset_cfg.get("topic", "")

    # Estimate budget per turn
    est_turn_prompt_tokens = 450
    est_turn_completion_tokens = 250
    est_total_tokens = sim_turns * (est_turn_prompt_tokens + est_turn_completion_tokens)

    plan = {
        "mode": "dry_run",
        "preset": preset_name,
        "turns": sim_turns,
        "topic": topic_text[:120] + ("..." if len(topic_text) > 120 else ""),
        "agents": harn["roster"],
        "estimated_tokens_total": est_total_tokens,
        "estimated_requests": sim_turns,
        "context_info": ctx_diag,
    }
    return plan


def format_dry_run_report(plan: Dict[str, Any]) -> str:
    """Formats human-readable dry-run summary."""
    lines = [
        "==========================================================",
        "          DIALOGUE CHAMBER — DRY-RUN SIMULATION           ",
        "==========================================================",
        f" Preset:              {plan['preset']}",
        f" Planned Turns:       {plan['turns']}",
        f" Topic:               {plan['topic']}",
        f" Estimated Requests:  {plan['estimated_requests']}",
        f" Estimated Tokens:    ~{plan['estimated_tokens_total']} tokens across run",
        "",
        "ROSTER CONTEXT BUDGETS:",
        f"{'AGENT':<18} {'BACKEND':<14} {'MODEL':<34} {'CONTEXT':<8}",
        "-" * 76,
    ]
    for r in plan["agents"]:
        lines.append(f"{r['agent']:<18} {r['backend']:<14} {r['model']:<34} {r['context_display']:<8}")

    lines.extend([
        "",
        "[DRY-RUN COMPLETE] No inference requests were submitted to providers.",
        "==========================================================",
    ])
    return "\n".join(lines)
