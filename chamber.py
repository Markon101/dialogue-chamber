#!/usr/bin/env python3
"""THE DIALOGUE CHAMBER — Evolution Edition.

Canonical in-memory state, validated event dispatcher, context intelligence engineer,
speaker scheduler, exploratory heuristics, Markdown & JSON report writer,
and Python-first orchestration engine.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlsplit

from artifacts import ArtifactError, ArtifactManager
from checkpoint import CheckpointManager, RunManifest, write_atomic
from context import ContextIntelligence, TokenAccounting, TokenizerEngine
from diagnostics import (
    format_context_info_report,
    format_doctor_report,
    format_dry_run_report,
    run_context_info,
    run_doctor,
    run_dry_run,
)
from inference import DemoBackend, HttpBackend, InferenceError, MockBackend, build_payload
from models import OpenRouterCatalog, get_catalog, handle_models_cli
from platform_adapter import (
    IS_TERMUX,
    ConcurrencyLock,
    WakeLockManager,
    get_available_memory_mb,
    get_thermal_diagnostics,
)
from presets import (
    CAPABILITY_SETS,
    COPY_DIR,
    DEFAULT_AGENT,
    ENDPOINTS,
    FREEFORM_TOPIC,
    LIMITS,
    PRESETS,
    RUNS_DIR,
    SAMPLING_PARAMS,
    normalize_sampling_param,
)
from protocol import (
    ACTION_TYPES, ALIASES, ITEM_KINDS, canonicalize_agent_id, canonicalize_claim_id,
    parse_reply, parse_sampling_spec,
)
from server_manager import LlamaServerManager, assert_single_llama_server, resolve_model
from society import Society, SOCIAL_ACTIONS
from autonomy import validate_policy, validate_settings
from communication import Communication
from context_edits import render_context_view, update_context_edits, validate_context_edits

STOP_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "are", "was", "were", "been",
    "into", "about", "what", "when", "where", "which", "while", "would", "could", "should",
    "their", "there", "they", "them", "then", "than", "your", "ours", "just", "more", "some",
    "such", "only", "also", "very", "much", "like", "make", "made", "does", "each", "other",
}
AGREE_RE = re.compile(
    r"^\s*(?:ah\b|yes\b|indeed\b|exactly\b|absolutely\b|spot on|you(?:'re| are) (?:right|absolutely right|on the right track)|"
    r"that(?:'s| is) (?:a )?(?:great|fantastic|fascinating|brilliant|wonderful|compelling|spot-on))",
    re.I,
)


@dataclass
class AgentState:
    name: str
    endpoint_name: str
    endpoint: dict
    model: str
    identity: str = ""
    focus: str = ""
    knowledge: str = ""
    temperature: float = 0.8
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    xtc_probability: Optional[float] = None
    xtc_threshold: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    dry_multiplier: Optional[float] = None
    dynatemp_range: Optional[float] = None
    max_tokens: int = 1200
    context_chars: int = 10000
    effective_context_tokens: int = 32768
    context_source: str = "config"
    reasoning: Optional[str] = None
    weight: float = 1.0
    priority: int = 0
    profile: str = "standard"
    autonomy: dict = field(default_factory=dict)
    self_settings: dict = field(default_factory=dict)
    context_edits: dict = field(default_factory=dict)
    capabilities: set = field(default_factory=set)
    subscriptions: list = field(default_factory=list)
    structured: bool = False
    note: str = ""
    style: str = ""
    pending_reads: list = field(default_factory=list)   # list of (fname, version, content)
    active: bool = True
    finished_reason: str = ""
    turns_taken: int = 0
    consecutive_turns: int = 0
    last_turn: int = 0
    failures: int = 0
    empty_count: int = 0
    fallback_models: list = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


@dataclass
class Message:
    id: str
    turn: int
    speaker: str
    channel: str
    content: str
    reply_to: Optional[str] = None
    reasoning: str = ""


@dataclass
class DirectMessage:
    id: str
    turn: int
    sender: str
    recipient: str
    content: str
    answered: bool = False


@dataclass
class BoardItem:
    id: str
    turn: int
    author: str
    kind: str
    content: str
    status: str = "active"   # active | disputed | retracted
    assignee: Optional[str] = None
    challenges: list = field(default_factory=list)   # list of dict(turn, author, content)
    retracted_turn: Optional[int] = None
    retracted_reason: str = ""
    votes: dict = field(default_factory=dict)        # agent -> vote string


@dataclass
class Event:
    seq: int
    turn: int
    actor: str
    type: str
    status: str   # accepted | rejected | system
    target: str = ""
    detail: str = ""
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


SAVED_RUNTIME_FIELDS = (
    "topic", "initial_topic", "reactive_queue", "nominated_next", "next_speaker",
    "requests_used", "retries_used", "prompt_tokens", "completion_tokens",
    "reasoning_tokens_total", "visible_tokens_total", "tracked_tokens", "remote_cost_usd",
    "remote_calls_without_cost", "empty_streak", "format_counts", "attribution_flags",
    "temp_changes", "sampling_changes", "end_votes", "end_requested",
    "speaker_transitions", "resurrection_events", "fallbacks_triggered",
    "solved_turn", "verified_synthesis_turn", "token_accounting_history",
)
SAVED_AGENT_FIELDS = (
    *SAMPLING_PARAMS, "self_settings", "context_edits",
    "note", "style", "pending_reads", "active", "finished_reason",
    "turns_taken", "consecutive_turns", "last_turn", "failures", "empty_count",
    "total_prompt_tokens", "total_completion_tokens", "subscriptions",
)


def preview_artifact(name: str, content: str, limit: int = 1200) -> str:
    if len(content) <= limit:
        return content
    marker = f"\n[Middle omitted; use @READ {name.removesuffix('.md')} OFFSET to inspect it.]\n"
    prefix = min(400, (limit - len(marker)) // 2)
    suffix = limit - len(marker) - prefix
    return content[:prefix] + marker + content[-suffix:]


class Chamber:
    """One self-contained multi-agent experiment run with context intelligence and checkpointing."""

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Optional[Any] = None,
        backend: Optional[Any] = None,
        seed: int = 42,
        quiet: bool = False,
        copy_dir: Optional[str] = COPY_DIR,
        context_mode: str = "tokens",     # "tokens" | "chars"
        safety_margin: int = 1024,
        free_only: bool = False,
        allow_paid: bool = False,
        checkpoint_data: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = copy.deepcopy(config)
        self.limits = {**LIMITS, **(self.cfg.get("limits") or {})}
        self.seed = seed
        self.rng = random.Random(seed)
        self.quiet = quiet
        self.copy_dir = copy_dir if copy_dir is not None else COPY_DIR
        self.backend = backend or HttpBackend(free_only=free_only, allow_paid=allow_paid)
        self.context_mode = context_mode
        self.safety_margin = safety_margin
        self.free_only = free_only
        self.allow_paid = allow_paid

        self.topic = str(self.cfg.get("topic") or FREEFORM_TOPIC).strip()
        self.initial_topic = self.topic
        self.mode_text = str(self.cfg.get("mode_text") or "").strip()
        self.scheduler_mode = str(self.cfg.get("scheduler") or "self_directed").lower()
        self.end_mode = str(self.cfg.get("end_mode") or "fixed_turns").lower()
        if self.end_mode not in ("fixed_turns", "agent_decided"):
            raise ValueError(f"unknown end_mode {self.end_mode!r}; expected 'fixed_turns' or 'agent_decided'")
        self.end_policy = str(self.cfg.get("end_policy") or "majority").lower()
        if self.end_policy not in ("majority", "unanimous", "any"):
            raise ValueError(f"unknown end_policy {self.end_policy!r}; expected 'majority', 'unanimous', or 'any'")
        self.min_turns_before_end = max(0, int(self.cfg.get("min_turns_before_end", 0)))
        self.board_visible = bool(self.cfg.get("board_visible", True))
        self.dm_visibility = str(self.cfg.get("dm_visibility") or "private")
        self.watch_terms = [t.lower() for t in (self.cfg.get("watch_terms") or [])]
        self.answer_key = self.cfg.get("answer_key")

        # Workspace & Artifact manager
        ts = time.strftime("%Y%m%d_%H%M%S")
        preset_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.cfg.get("name") or "experiment"))
        self.run_id = str(self.cfg.get("run_id") or f"{preset_name}_{ts}")
        self.run_dir = Path(run_dir or os.path.join(RUNS_DIR, self.run_id)).resolve()
        if any((self.run_dir / filename).exists() for filename in ("manifest.json", "checkpoint.json", "events.jsonl")):
            raise ValueError("run directory already contains a run; continue into a new directory")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactManager(
            self.run_dir / "artifacts",
            max_count=int(self.limits["max_artifacts"]),
            max_bytes=int(self.limits["max_artifact_bytes"]),
            max_total_bytes=int(self.limits["max_workspace_bytes"]),
        )
        self.visible_artifacts: List[str] = []

        # Checkpointing & Context Intelligence
        self.checkpoints = CheckpointManager(self.run_dir)
        self.tokenizer = TokenizerEngine()
        self.ctx_intelligence = ContextIntelligence(
            default_safety_margin=self.safety_margin,
            tokenizer=self.tokenizer,
        )

        # Build roster & models
        raw_agents = self.cfg.get("agents") or []
        if not raw_agents or len(raw_agents) > int(self.limits["max_agents"]):
            raise ValueError(f"agent count must be between 1 and {self.limits['max_agents']}")

        self.agents: Dict[str, AgentState] = {}
        self.agent_order: List[str] = []
        self.channels: Dict[str, List[Message]] = {"general": []}
        for ch in (self.cfg.get("channels") or {}):
            cname = re.sub(r"[^a-z0-9_-]", "", str(ch).lower())[:24]
            if cname:
                self.channels.setdefault(cname, [])

        endpoints_map = {**ENDPOINTS, **(self.cfg.get("endpoints") or {})}
        self.cfg["endpoints"] = copy.deepcopy(endpoints_map)
        resolved_agents = []
        server_mgr = LlamaServerManager()
        catalog = get_catalog()

        # Query local server once to avoid repeating HTTP queries per agent
        server_reported_local_ctx = None
        # Offline and remote-only runs must not contact the local model server.
        if isinstance(self.backend, HttpBackend) and any(a.get("endpoint", "local") == "local" for a in raw_agents):
            url = urlsplit(endpoints_map["local"]["base_url"])
            if url.scheme == "http" and url.hostname:
                info = server_mgr.query_server_info(url.hostname, url.port or 80, timeout=0.8)
                if info.get("online"):
                    server_reported_local_ctx = info.get("allocated_context") or info.get("context_window")

        for raw in raw_agents:
            merged = {**DEFAULT_AGENT, **raw}
            name = canonicalize_agent_id(merged.get("name"))
            if not name or name in self.agents:
                raise ValueError(f"invalid or duplicate agent name: {merged.get('name')!r}")

            # Endpoint and model resolution
            ep_ref = merged.get("endpoint", "local")
            if isinstance(ep_ref, dict):
                ep_name, ep = ep_ref.get("name", "custom"), dict(ep_ref)
            else:
                ep_name = str(ep_ref)
                if ep_name not in endpoints_map:
                    # Support dynamic "openrouter" or "local" endpoint directly
                    if ep_name == "openrouter":
                        ep = dict(endpoints_map["hermes"])
                    else:
                        raise ValueError(f"unknown endpoint {ep_name!r}")
                else:
                    ep = dict(endpoints_map[ep_name])

            if merged.get("model"):
                ep["model"] = str(merged["model"])
            # Backend-specific context discovery and negotiation
            ctx_source = "config"
            if ep.get("provider") == "llamacpp":
                eff_ctx, ctx_source, _ = self.ctx_intelligence.negotiate_context_limit(
                    server_reported_ctx=server_reported_local_ctx if ep_name == "local" else None,
                    user_configured_ctx=self.cfg.get("context_tokens"),
                    model_default_ctx=ep.get("context_tokens", 32768 if "gemma" in ep["model"].lower() else 4096),
                )
                if ep.get("context_tokens"):
                    eff_ctx = min(eff_ctx, int(ep["context_tokens"]))
                    ctx_source = "endpoint-cap"
            elif ep.get("provider") == "openrouter":
                m_meta = catalog.get_model(ep["model"]) if isinstance(self.backend, HttpBackend) else None
                reported_m_ctx = m_meta.context_length if m_meta else None
                eff_ctx, ctx_source, _ = self.ctx_intelligence.negotiate_context_limit(
                    server_reported_ctx=reported_m_ctx,
                    user_configured_ctx=self.cfg.get("context_tokens"),
                    model_default_ctx=reported_m_ctx or 65536,
                )
            else:
                eff_ctx = 4096
                ctx_source = "manual"

            caps_spec = merged.get("capabilities", "standard")
            if isinstance(caps_spec, str):
                caps = set(CAPABILITY_SETS.get(caps_spec, CAPABILITY_SETS["standard"]))
            else:
                caps = {ALIASES.get(str(c).upper(), str(c).upper()) for c in caps_spec}

            subs = []
            for ch in (merged.get("subscriptions") or []):
                cname = re.sub(r"[^a-z0-9_-]", "", str(ch).lower())[:24]
                if cname and cname not in subs and len(subs) < int(self.limits["max_subscriptions"]):
                    subs.append(cname)
                    self.channels.setdefault(cname, [])

            if ep_name == "dory" and "temperature" not in raw:
                merged["temperature"] = 1.0
            temp_bounds = (float(self.limits["temp_min"]), float(self.limits["temp_max"]))
            _, temp = normalize_sampling_param("temperature", merged["temperature"], temp_bounds=temp_bounds)
            max_tok = min(int(self.limits["max_tokens_per_response"]), max(32, int(merged["max_tokens"])))
            ctx_chars = min(int(self.limits["max_context_chars"]), max(1000, int(merged["context_chars"])))
            if ep.get("max_output_tokens"):
                max_tok = min(max_tok, int(ep["max_output_tokens"]))
            if ep.get("max_context_chars"):
                ctx_chars = min(ctx_chars, int(ep["max_context_chars"]))
            if ep_name == "dory":
                for key, value in (("top_p", 0.95), ("top_k", 0), ("min_p", 0.0), ("reasoning", "off")):
                    if key not in raw:
                        merged[key] = value

            fallbacks = list(merged.get("fallback_models") or [])
            sampling_kwargs = {}
            for sk in SAMPLING_PARAMS:
                if sk == "temperature":
                    continue
                if merged.get(sk) is not None:
                    _, sampling_kwargs[sk] = normalize_sampling_param(sk, merged[sk])
                    merged[sk] = sampling_kwargs[sk]

            autonomy = merged.get("autonomy", {})
            validate_policy(autonomy, max_tok)
            if autonomy.keys() - {"context_edit"}:
                caps.add("CONFIGURE_SELF")
            if "context_edit" in autonomy:
                caps.add("EDIT_CONTEXT")

            ag = AgentState(
                name=name,
                endpoint_name=ep_name,
                endpoint=ep,
                model=ep["model"],
                identity=str(merged.get("identity") or ""),
                focus=str(merged.get("focus") or ""),
                knowledge=str(merged.get("knowledge") or ""),
                temperature=temp,
                **sampling_kwargs,
                max_tokens=max_tok,
                context_chars=ctx_chars,
                effective_context_tokens=eff_ctx,
                context_source=ctx_source,
                reasoning=merged.get("reasoning"),
                weight=max(0.05, float(merged.get("weight", 1.0))),
                priority=int(merged.get("priority", 0)),
                profile=str(merged.get("profile") or "standard").lower(),
                capabilities=caps,
                autonomy=autonomy,
                subscriptions=subs,
                structured=bool(merged.get("structured", False)),
                fallback_models=fallbacks,
            )
            self.agents[name] = ag
            self.agent_order.append(name)
            resolved_agents.append({**merged, "name": name, "model": ag.model})

        self.cfg["agents"] = resolved_agents

        self.agent_order.sort(key=lambda n: -self.agents[n].priority)

        self.society = Society(self, self.cfg["society"]) if "society" in self.cfg else None
        if self.society:
            for agent in self.agents.values():
                agent.capabilities.update(SOCIAL_ACTIONS)
        self.communication = Communication(self, self.cfg.get("communication", {}))

        # Seed initial Markdown artifacts if configured
        art_cfg = self.cfg.get("artifacts") or {}
        for aname, content in ((art_cfg.get("seed") or {}) if checkpoint_data is None else {}).items():
            fname, _ = self.artifacts.write("SYSTEM", aname, "create", str(content), turn=0, attribution=False)
            for n in self.agent_order:
                self.artifacts.seen[(n, fname)] = 1
        for aname in (art_cfg.get("visible") or []):
            try:
                self.visible_artifacts.append(self.artifacts.normalize(aname))
            except ArtifactError:
                pass

        # In-memory state collections
        self.public_messages: List[Message] = []
        self.dms: List[DirectMessage] = []
        self.board: Dict[str, BoardItem] = {}
        self.events: List[Event] = []
        self.reactive_queue: List[str] = []
        self.nominated_next: Optional[str] = None
        self.next_speaker: Optional[str] = None
        self.turn = 0
        self.stop_reason = "not started"
        self.start_time = time.monotonic()
        self.elapsed_s = 0.0

        # Counters & telemetry
        self.requests_used = 0
        self.retries_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens_total = 0
        self.visible_tokens_total = 0
        self.tracked_tokens = 0
        self.remote_cost_usd = 0.0
        self.remote_calls_without_cost = 0
        self.empty_streak = 0
        self.format_counts = {"json": 0, "partial_json": 0, "commands": 0, "plain": 0, "empty": 0}
        self.attribution_flags: List[Tuple[int, str, str]] = []
        self.temp_changes: List[Tuple[int, str, float, float]] = []
        self.sampling_changes: List[Tuple[int, str, dict]] = []
        self.end_votes: Dict[str, str] = {}
        self.end_requested: bool = False
        self.speaker_transitions: List[Tuple[int, str, str]] = []
        self.resurrection_events: List[dict] = []
        self.fallbacks_triggered: List[dict] = []
        self.solved_turn: Optional[int] = None
        self.verified_synthesis_turn: Optional[int] = None
        self.start_memory_mb = get_available_memory_mb()
        self.token_accounting_history: List[Dict[str, Any]] = []

        if checkpoint_data is not None:
            self.resume_from_checkpoint(checkpoint_data)

        # Initialize Run Manifest
        roster_meta = [
            {
                "name": a.name,
                "model": a.model,
                "backend": a.endpoint.get("provider", "custom"),
                "context_window": a.effective_context_tokens,
                "context_source": a.context_source,
                "max_tokens": a.max_tokens,
                "temperature": a.temperature,
            }
            for a in self.agents.values()
        ]
        self.checkpoints.init_manifest(
            run_id=self.run_id,
            preset_name=preset_name,
            topic=self.topic,
            effective_config=self.cfg,
            agent_roster=roster_meta,
            token_budgets={
                "max_total_tokens": self.limits["max_total_tokens"],
                "max_tokens_per_response": self.limits["max_tokens_per_response"],
                "safety_margin": self.safety_margin,
            },
            max_turns=int(self.limits["max_turns"]),
            seed=self.seed,
        )

    # ── logging & events ──
    def _emit(self, line: str = ""):
        if not self.quiet:
            print(line, flush=True)

    def _record(self, actor: str, etype: str, status: str, target: str = "", detail: str = "", **data) -> Event:
        ev = Event(
            seq=len(self.events) + 1,
            turn=self.turn,
            actor=actor,
            type=etype,
            status=status,
            target=str(target)[:80],
            detail=str(detail)[:240],
            data=data,
        )
        self.events.append(ev)
        self.checkpoints.append_event({
            "seq": ev.seq,
            "turn": ev.turn,
            "actor": ev.actor,
            "type": ev.type,
            "status": ev.status,
            "target": ev.target,
            "detail": ev.detail,
            "data": ev.data,
            "timestamp": ev.timestamp,
        })
        return ev

    # ── context construction with intelligence ──
    def _visible_context_records(self, agent, source):
        if source == "public":
            return [m for m in self.public_messages if m.speaker == agent.name or
                    self.communication.allows("public", m.speaker, agent.name)]
        if source == "whispers":
            return [m for m in self.dms if m.sender == agent.name or m.recipient == agent.name][-8:]
        if isinstance(source, str) and source.startswith("#"):
            channel = source[1:]
            if channel not in agent.subscriptions or not self.communication.channel_allows(channel, agent.name, "readers"):
                return None
            if self.society and channel in self.society.groups and agent.name not in self.society.groups[channel]["members"]:
                return None
            return self.channels.get(channel, [])[-6:]
        return None

    def build_context(self, agent: AgentState) -> Tuple[List[Dict[str, str]], TokenAccounting]:
        others = [n for n in self.agent_order if n != agent.name]
        all_names = ", ".join(self.agent_order)
        other_names = ", ".join(others) if others else agent.name
        turn = max(1, self.turn)
        turn_limit = int(self.limits["max_turns"])
        remaining = max(0, turn_limit - turn)
        turn_budget = f"Turn {turn} of {turn_limit}. Scheduled turns remaining after this reply: {remaining}."
        if remaining == 0:
            turn_budget += " Submit any final actions in this reply."
        if self.end_mode == "agent_decided" and "END_RUN" in agent.capabilities:
            if turn >= self.min_turns_before_end:
                turn_budget += f" End voting is open ({self.end_policy}). Use @END_RUN: <reason> to vote."
            elif self.min_turns_before_end <= turn_limit:
                turn_budget += f" End voting opens on turn {self.min_turns_before_end} ({self.end_policy})."
            else:
                turn_budget += " The turn cap arrives before end voting opens."

        sys_sections = {
            "preamble": f"You are {agent.name}, participating in an open-ended multi-agent experiment.\n"
                        f"Participants: {all_names}. Other participants: {other_names}.",
            "turn_budget": turn_budget,
            "identity": agent.identity.strip() if agent.identity else "",
            "role": f"Your current angle or role: {agent.focus.strip()}." if agent.focus else "",
            "style": f"Your chosen speaking style: {agent.style.strip()}." if agent.style else "",
            "mode": self.mode_text if self.mode_text else "",
            "boundary": "You operate inside a bounded chamber: no external tools, shell, files, or web. "
                        "Respond naturally to the conversation — express curiosity, disagreement, humor, or new ideas. "
                        "Write only your own response; never write lines for another participant.",
            "protocol": self._protocol_instructions(agent),
        }

        fixed_sections = {
            "topic": f"Topic: {self.topic}",
            "knowledge": f"--- Your Private Initial Knowledge (visible only to {agent.name}) ---\n{agent.knowledge}" if agent.knowledge else "",
            "scratchpad": f"--- Your Private Scratchpad (visible only to {agent.name}) ---\n{agent.note}" if agent.note else "",
        }
        self_policy = {key: value for key, value in agent.autonomy.items() if key != "context_edit"}
        if self_policy:
            policy = json.dumps(self_policy, ensure_ascii=False)
            sys_sections["protocol"] += (
                "\nUse @CONFIGURE_SELF followed by a one-line JSON object, or a CONFIGURE_SELF action "
                "with a settings object. Change any permitted subset for your next turns. "
                f"Your permissions: {policy}. focus_chars allows a private focus string up to that length. "
                "history_messages selects the latest visible public messages. reply_tokens caps your reply."
            )
            current = {
                "focus": agent.self_settings.get("focus", ""),
                "reply_tokens": agent.self_settings.get("reply_tokens", agent.max_tokens),
                "history_messages": agent.self_settings.get("history_messages", "all that fit"),
            }
            fixed_sections["self_settings"] = "Your working focus and turn settings: " + json.dumps(current, ensure_ascii=False)
            last = next((e for e in reversed(self.events) if e.actor == agent.name and e.type == "CONFIGURE_SELF"), None)
            if last:
                fixed_sections["self_result"] = f"Last self-configuration: {last.status}. {last.detail}"
        if "EDIT_CONTEXT" in agent.capabilities:
            edit_policy = agent.autonomy.get("context_edit", {})
            sys_sections["protocol"] += (
                '\nUse @EDIT_CONTEXT {"source":"public","op":"hide","target":"m1"} '
                'or a JSON action with type EDIT_CONTEXT and an edit object. '
                'Edits affect your own context. Use the message IDs already shown to you. '
                '"replace" takes a target and text, "summarize" takes text for all visible prior messages '
                'in the source, and "restore" takes a target or restores the whole source. '
                f"Your source permissions: {json.dumps(edit_policy)}."
            )
            last = next((e for e in reversed(self.events) if e.actor == agent.name and e.type == "EDIT_CONTEXT"), None)
            if last:
                fixed_sections["context_edit_result"] = f"Last private context edit: {last.status}. {last.detail}"

        if "communication" in self.cfg:
            fixed_sections["communication"] = self.communication.describe(agent.name)
        if self.society:
            fixed_sections["society"] = "--- Society ---\n" + self.society.describe(agent.name)
            my_events = [e for e in self.events if e.actor == agent.name and
                         e.type in SOCIAL_ACTIONS | {"PUBLISH", "ARTIFACT_WRITE"}][-8:]
            if my_events:
                fixed_sections["society_results"] = "--- Results of your recent actions ---\n" + "\n".join(
                    f"{e.type} {e.target}: {e.status}" + (f" ({e.detail})" if e.detail else "") for e in my_events)

        # Artifacts
        art_blocks = []
        shown_files = set()
        for saved_read in agent.pending_reads:
            fname, ver, text = saved_read[:3]
            shown_files.add(fname)
            if len(saved_read) == 5:
                _, _, _, offset, total = saved_read
                label = f"[Artifact {fname} (v{ver}, requested read, chars {offset}-{offset + len(text)} of {total})]"
            else:
                label = f"[Artifact {fname} (v{ver}, requested read)]"
            art_blocks.append(f"{label}\n{text[:1500]}")
        for fname in self.visible_artifacts:
            if fname in self.artifacts.versions and fname not in shown_files:
                try:
                    _, ver, text = self.artifacts.read(agent.name, fname, turn=self.turn)
                    art_blocks.append(f"[Shared Artifact {fname} (v{ver})]\n{preview_artifact(fname, text)}")
                except ArtifactError:
                    pass
        if "ARTIFACT_WRITE" in agent.capabilities or "ARTIFACT_READ" in agent.capabilities:
            listing = ", ".join(f"{f} (v{v}, {sz}B)" for f, v, sz in self.artifacts.listing()) or "(none yet)"
            art_blocks.append(f"Available workspace artifacts: {listing}")
        if art_blocks:
            fixed_sections["artifacts"] = "--- Workspace Artifacts ---\n" + "\n\n".join(art_blocks)

        # Blackboard
        if self.board and (self.board_visible or any(k in agent.capabilities for k in ("CLAIM", "CHALLENGE", "RETRACT"))):
            b_lines = []
            for item in list(self.board.values())[-18:]:
                tag = item.status.upper()
                if item.status == "retracted":
                    tag = f"RETRACTED by {item.author} on turn {item.retracted_turn} — withdrawn, do not treat as active"
                elif item.status == "disputed":
                    tag = f"DISPUTED ({len(item.challenges)} challenge(s))"
                extra = f" [assigned: {item.assignee}]" if item.assignee else ""
                votes = f" [votes: {', '.join(f'{k}={v}' for k, v in item.votes.items())}]" if item.votes else ""
                b_lines.append(f"- {item.id} [{item.kind} by {item.author}, {tag}]{extra}{votes}: {item.content}")
                for ch in item.challenges[-2:]:
                    b_lines.append(f"    ↳ challenge by {ch['author']} (turn {ch['turn']}): {ch['content']}")
            fixed_sections["blackboard"] = "--- Shared Blackboard ---\n" + "\n".join(b_lines)

        # Apply private views only after resolving the current visibility of each source.
        summaries = []

        def viewed(source):
            records = self._visible_context_records(agent, source)
            if records is None:
                return [], 0
            messages, summary = render_context_view(records, agent.context_edits.get(source))
            if summary:
                summaries.append(f"[Your summary of {source} through turn {summary['through_turn']}] {summary['text']}")
            return messages, len(records) - len(messages)

        ch_lines = []
        for cname in agent.subscriptions:
            messages, _ = viewed(f"#{cname}")
            for m in messages:
                ch_lines.append(f"[#{cname} | {m.id} | turn {m.turn}] {m.speaker}: {m.content}")
        if ch_lines:
            fixed_sections["channels"] = "--- Subscribed Channels ---\n" + "\n".join(ch_lines)

        whispers, _ = viewed("whispers")
        if whispers:
            dm_lines = [
                f"[PRIVATE WHISPER {d.id} | turn {d.turn} | {d.sender} → {d.recipient}]: {d.content}"
                for d in whispers
            ]
            fixed_sections["whispers"] = "--- Private Messages (visible only to sender & recipient) ---\n" + "\n".join(dm_lines)

        public, edited_count = viewed("public")
        if summaries:
            fixed_sections["context_summaries"] = "--- Your Private Context Summaries ---\n" + "\n".join(summaries)
        history_count = agent.self_settings.get("history_messages", len(public))
        hidden_count = max(0, len(public) - history_count)
        pub_msgs = [{"id": m.id, "turn": m.turn, "speaker": m.speaker, "content": m.content, "reply_to": m.reply_to}
                    for m in public[hidden_count:]]
        if self.dm_visibility == "notice":
            for d in self.dms[-6:]:
                if d.sender != agent.name and d.recipient != agent.name:
                    pub_msgs.append({"id": f"not_{d.id}", "turn": d.turn, "speaker": "NOTICE", "content": f"(notice: {d.sender} sent a private whisper to {d.recipient})"})

        messages, accounting = self.ctx_intelligence.assemble_and_budget(
            agent_name=agent.name,
            endpoint=agent.endpoint if isinstance(self.backend, HttpBackend) else {**agent.endpoint, "provider": "mock"},
            system_sections=sys_sections,
            fixed_sections=fixed_sections,
            public_messages=pub_msgs,
            effective_context=agent.effective_context_tokens,
            context_source=agent.context_source,
            reserved_output=agent.self_settings.get("reply_tokens", agent.max_tokens),
            hidden_public_count=hidden_count,
            edited_public_count=edited_count,
            safety_margin=int(agent.endpoint.get("safety_margin", self.safety_margin)),
            context_mode=self.context_mode,
            max_context_chars=int(self.limits["max_context_chars"]),
            agent_context_chars=int(agent.context_chars),
        )

        for diag_msg in accounting.diagnostics:
            self._record(agent.name, "CONTEXT_DIAGNOSTIC", "system", detail=diag_msg)

        return messages, accounting

    def _protocol_instructions(self, agent: AgentState) -> str:
        caps = agent.capabilities
        names = ", ".join(self.agent_order)
        lo, hi = self.limits["temp_min"], self.limits["temp_max"]
        if self.society:
            return ("Put any commands first, each on its own line, followed by your public reply. "
                    "Keep the public reply concise (about 100 words) so commands are not cut off.\n"
                    "@CREATE_GROUP name | @INVITE_GROUP name AGENT | @JOIN_GROUP name | @LEAVE_GROUP name\n"
                    "@COMPEL_GROUP name AGENT | @RELEASE_GROUP name AGENT (authorities only)\n"
                    "@PUBLISH name: private group message | @WORLD: one-line addition to the world ledger\n"
                    "@READ world OFFSET: inspect 1,500 characters starting at OFFSET (0-based; optional)\n"
                    "Group names use 1-24 lowercase letters, digits, hyphens or underscores, starting with a letter. "
                    "Follow each group's admission and frozen policies in the society state. "
                    "Group members can read up to six recent earlier posts, subject to channel permissions. "
                    "Only group members can post. Compelled members can leave after release. "
                    "Do not claim you joined unless society state confirms it; failed commands do not change state. "
                    "The world document is shared; distinguish claims from established facts. "
                    "Speech follows configured communication routes; by default everyone receives it.")
        if agent.profile == "minimal":
            lines = [
                "Write your message in plain text (no speaker prefix).",
                f"Optionally, at the very end on separate lines, you may add:\n"
                f"NEXT: <name>   ({names})\n"
                f"TEMP: <number> ({lo} to {hi})",
            ]
            if "CLAIM" in caps:
                lines.append("Optional board lines: @CLAIM: <idea> | @CHALLENGE <id>: <why> | @RETRACT <id>")
            if "WHISPER" in caps:
                lines.append("Optional private line: @WHISPER <name>: <secret message>")
            if self.end_mode == "agent_decided" and "END_RUN" in caps:
                lines.append(f"@END_RUN: <reason> (vote from turn {self.min_turns_before_end}, policy: {self.end_policy})")
            if self.end_mode == "agent_decided" and "FINISH" in caps:
                lines.append("@FINISH: <reason> (leave the conversation)")
            return "\n".join(lines)

        supports_extra_samplers = agent.endpoint.get("provider") == "llamacpp" and not agent.endpoint.get("externally_managed")
        sampling_enabled = "SET_SAMPLING" in caps
        can_end = self.end_mode == "agent_decided" and "END_RUN" in caps
        end_hint = f"Vote to end from turn {self.min_turns_before_end} ({self.end_policy})."
        if agent.profile == "rich":
            allowed = sorted(caps - {"SAY"})
            if self.end_mode != "agent_decided" and "END_RUN" in allowed:
                allowed.remove("END_RUN")
            sampling_example = {"min_p": 0.05, "top_k": 40}
            if supports_extra_samplers:
                sampling_example.update(xtc_probability=0.3, xtc_threshold=0.1)
            extra_json = f'\n  "sampling_request": {json.dumps(sampling_example)},' if sampling_enabled else ""
            extra_note = f"\nUse END_RUN to vote. {end_hint}" if can_end else ""
            return (
                "Respond with a JSON object (or plain text with @commands if you prefer):\n"
                '{\n  "message": "your public speech",\n'
                f'  "next_speaker": "one of {names} (optional)",\n'
                f'  "temperature_request": 0.9,{extra_json}\n'
                '  "actions": [\n'
                '    {"type": "CLAIM", "kind": "hypothesis|claim|question|proposal|decision|task", "content": "..."},\n'
                '    {"type": "CHALLENGE", "target": "c1", "content": "counterargument"},\n'
                '    {"type": "RETRACT", "target": "c1", "content": "reason for withdrawing your own claim"},\n'
                '    {"type": "WHISPER", "recipient": "NAME", "content": "private message"},\n'
                '    {"type": "PUBLISH", "channel": "planning", "content": "channel post"},\n'
                '    {"type": "VOTE", "target": "c1", "value": "yes|no"},\n'
                '    {"type": "UPDATE_NOTE", "content": "your private scratchpad note"},\n'
                '    {"type": "ARTIFACT_WRITE", "name": "world", "mode": "create|append|replace", "content": "markdown"},\n'
                '    {"type": "ARTIFACT_READ", "name": "world"}\n'
                "  ]\n}\n"
                f"Permitted action types for you: {', '.join(allowed)}.{extra_note}"
            )

        cmds = [
            f"NEXT: <name> ({names}" + ("; omit or write NONE when ending)" if self.end_mode == "agent_decided" else ")"),
            f"TEMP: <{lo}..{hi}>",
        ]
        if sampling_enabled:
            sampler_hint = "@SET_SAMPLING top_p=0.9 top_k=40 min_p=0.05"
            if supports_extra_samplers:
                sampler_hint += " xtc_prob=0.1 xtc_thresh=0.1 repeat_penalty=1.1"
            cmds.append(sampler_hint)
        if "WHISPER" in caps:
            cmds.append("@WHISPER <name>: <private message>")
        if "PUBLISH" in caps and len(self.channels) > 1:
            cmds.append(f"@PUBLISH <{'|'.join(sorted(self.channels))}>: <message>")
        if "CLAIM" in caps:
            cmds.append("@CLAIM hypothesis: <statement>  (or @CLAIM claim: ...)")
        if "CHALLENGE" in caps:
            cmds.append("@CHALLENGE <c1|c2|...>: <objection>  (use exact board ID, e.g. c1)")
        if "RETRACT" in caps:
            cmds.append("@RETRACT <c1|c2|...>: <why you withdraw your claim>")
        if "VOTE" in caps:
            cmds.append("@VOTE <c1|c2|...> yes|no  (do not re-vote unchanged votes)")
        if "UPDATE_NOTE" in caps:
            cmds.append("@NOTE: <private scratchpad note>")
        if "ARTIFACT_WRITE" in caps:
            cmds.append("@APPEND <artifact_name>: <markdown to append>  (or @READ <artifact_name> [offset])")
        if can_end:
            cmds.append(f"@END_RUN: <reason> ({end_hint})")
        if self.end_mode == "agent_decided" and "FINISH" in caps:
            cmds.append("@FINISH: <reason> (leave the conversation)")
        max_act = int(self.limits["max_actions_per_response"])
        return f"Write your public reply, then optionally add up to {max_act} control/action lines at the end:\n" + "\n".join(cmds)

    # ── budget & model inference with fallback chain ──
    def _check_budgets_before_call(self, endpoint: dict) -> Optional[str]:
        if (time.monotonic() - self.start_time) >= float(self.limits["max_duration_s"]):
            return "duration limit reached"
        if self.requests_used >= int(self.limits["max_requests"]):
            return f"request budget reached ({self.limits['max_requests']})"
        if self.tracked_tokens >= int(self.limits["max_total_tokens"]):
            return f"total token budget reached ({self.limits['max_total_tokens']})"
        max_spend = self.limits.get("max_spend_usd")
        if max_spend is not None and (endpoint.get("remote") or endpoint.get("provider") == "openrouter"):
            if self.remote_cost_blocked(float(max_spend)):
                return f"remote spend limit or unpriced remote usage ({max_spend} USD)"
        return None

    def remote_cost_blocked(self, max_spend: float) -> bool:
        if self.remote_cost_usd >= max_spend:
            return True
        if self.remote_calls_without_cost > 0 and max_spend > 0:
            return True
        return False

    def _call_model(
        self,
        agent: AgentState,
        messages: list,
        max_tokens: Optional[int] = None,
        stops=None,
        reasoning=None,
    ):
        budget_err = self._check_budgets_before_call(agent.endpoint)
        if budget_err:
            raise InferenceError(budget_err, transient=False)

        mtok = min(int(self.limits["max_tokens_per_response"]), int(max_tokens or agent.max_tokens))
        mtok = min(mtok, agent.self_settings.get("reply_tokens", mtok))
        rem_tok = int(self.limits["max_total_tokens"]) - self.tracked_tokens
        mtok = max(1, min(mtok, rem_tok))

        # Model routing & fallback candidate list
        candidates = [(agent.endpoint_name, agent.endpoint)]
        for fb in agent.fallback_models:
            if isinstance(fb, dict):
                candidates.append((fb.get("name", "fallback"), fb))
            elif isinstance(fb, str):
                ep_map = {**ENDPOINTS, **(self.cfg.get("endpoints") or {})}
                if fb in ep_map:
                    candidates.append((fb, ep_map[fb]))
                elif "/" in fb:
                    # OpenRouter model fallback
                    fb_ep = dict(ep_map.get("hermes", ENDPOINTS["hermes"]))
                    fb_ep["model"] = fb
                    candidates.append((f"openrouter:{fb}", fb_ep))

        last_error = None
        for cand_idx, (cand_name, cand_ep) in enumerate(candidates):
            if cand_idx > 0 and cand_ep.get("externally_managed"):
                last_error = InferenceError("externally managed endpoints must be selected directly; fallback would bypass their context and identity checks", transient=False)
                self._record(agent.name, "MODEL_FALLBACK", "rejected", target=cand_name, detail=str(last_error))
                continue
            if cand_idx > 0:
                self.fallbacks_triggered.append({
                    "turn": self.turn,
                    "agent": agent.name,
                    "from_endpoint": candidates[cand_idx - 1][0],
                    "to_endpoint": cand_name,
                    "reason": str(last_error),
                })
                self._record(agent.name, "MODEL_FALLBACK", "system", target=cand_name, detail=f"Fell back from {candidates[cand_idx-1][0]}: {last_error}")
                self._emit(f"  ↪ [{agent.name}] falling back to model {cand_ep.get('model')} ({cand_name})")

            payload, reasoning_label = build_payload(
                cand_ep,
                agent.__dict__,
                messages,
                agent.temperature,
                mtok,
                stops=stops,
                seed=self.seed + self.turn,
                reasoning=reasoning,
            )

            attempts = 0
            max_attempts = 1 + int(cand_ep.get("max_retries_per_call", self.limits["max_retries_per_call"]))
            backoff_s = 1.0

            while True:
                budget_err = self._check_budgets_before_call(cand_ep)
                if budget_err:
                    raise InferenceError(budget_err, transient=False)
                attempts += 1
                self.requests_used += 1
                try:
                    if isinstance(self.backend, HttpBackend) and cand_ep.get("provider") == "llamacpp":
                        assert_single_llama_server()
                    res = self.backend.complete(
                        agent.name,
                        cand_ep,
                        payload,
                        timeout=float(cand_ep.get("request_timeout_s", 120)),
                        reasoning_mode_label=reasoning_label,
                    )
                    # Successful completion
                    ptok = res.prompt_tokens if res.prompt_tokens is not None else max(1, len(json.dumps(messages)) // 4)
                    ctok = res.completion_tokens if res.completion_tokens is not None else max(1, len(res.content + res.reasoning) // 4)
                    self.prompt_tokens += ptok
                    self.completion_tokens += ctok
                    self.tracked_tokens += ctok
                    agent.total_prompt_tokens += ptok
                    agent.total_completion_tokens += ctok

                    if res.reasoning_tokens:
                        self.reasoning_tokens_total += res.reasoning_tokens
                    if res.visible_tokens:
                        self.visible_tokens_total += res.visible_tokens

                    if cand_ep.get("remote") or cand_ep.get("provider") == "openrouter":
                        if res.cost is not None:
                            self.remote_cost_usd += float(res.cost)
                        else:
                            self.remote_calls_without_cost += 1
                    return res

                except InferenceError as e:
                    last_error = e
                    if (not e.transient or attempts >= max_attempts
                            or self.retries_used >= int(self.limits["max_retries_total"])
                            or self.requests_used >= int(self.limits["max_requests"])):
                        break
                    self.retries_used += 1
                    # Rate limit backoff
                    time.sleep(min(backoff_s, 15.0))
                    backoff_s *= 2.0

        if last_error:
            raise last_error
        raise InferenceError("All candidate model endpoints exhausted", transient=False)

    # ── action validation & dispatch ──
    def _dispatch_actions(self, agent: AgentState, parsed):
        for issue in parsed.issues:
            self._record(agent.name, "PROTOCOL", "rejected", detail=issue)
        for flag in parsed.flags:
            self.attribution_flags.append((self.turn, agent.name, flag))
            self._record(agent.name, "ATTRIBUTION_FLAG", "system", detail=flag)

        agent.pending_reads.clear()

        # Public message
        if parsed.message:
            mid = f"m{len(self.public_messages) + 1}"
            msg = Message(
                id=mid,
                turn=self.turn,
                speaker=agent.name,
                channel="public",
                content=parsed.message,
                reply_to=parsed.reply_to,
                reasoning=parsed.reasoning,
            )
            self.public_messages.append(msg)
            self._record(agent.name, "SAY", "accepted", target=mid, detail=parsed.message[:120])
            self._check_dm_answered(agent.name, parsed.message)
            self._check_resurrection(agent.name, parsed.message)
            self._check_answer_key(parsed.message)
            for other in self.agent_order:
                if other != agent.name and self.communication.allows("public", agent.name, other) and re.search(rf"\b{re.escape(other)}\b", parsed.message):
                    self._queue_reactive(other)

        # Temperature request
        if parsed.temperature is not None:
            self._apply_action(agent, {"type": "SET_TEMPERATURE", "value": parsed.temperature})

        # Next speaker nomination
        if parsed.next_speaker:
            self._apply_action(agent, {"type": "PROPOSE_NEXT", "target": parsed.next_speaker})

        # Skip unchanged votes while preserving deliberate vote changes and repeated writes.
        deduped_actions = []
        votes = {}
        for act in parsed.actions:
            if act["type"] == "VOTE" and "VOTE" in agent.capabilities and act["target"] in self.board:
                target, value = act["target"], act["value"]
                if votes.get(target, self.board[target].votes.get(agent.name)) == value:
                    continue
                votes[target] = value
            deduped_actions.append(act)

        max_act = int(self.limits["max_actions_per_response"])
        priority = {"FINISH"}
        if self.end_mode == "agent_decided" and self.turn >= self.min_turns_before_end:
            priority.add("END_RUN")
        priority &= agent.capabilities
        if 0 < max_act < len(deduped_actions):
            first = next((i for i, action in enumerate(deduped_actions) if action["type"] in priority), None)
            if first is not None and first >= max_act:
                deduped_actions.insert(max_act - 1, deduped_actions.pop(first))

        for idx, act in enumerate(deduped_actions):
            if idx >= max_act:
                self._record(agent.name, act.get("type", "UNKNOWN"), "rejected",
                             detail=f"exceeded max_actions_per_response ({max_act})")
                continue
            self._apply_action(agent, act)

    def _apply_action(self, agent: AgentState, act: dict):
        atype = str(act.get("type") or "").upper()
        if atype not in ACTION_TYPES:
            self._record(agent.name, atype or "UNKNOWN", "rejected", detail="unknown action type")
            return
        if atype not in agent.capabilities:
            self._record(agent.name, atype, "rejected", detail=f"capability {atype} not enabled for {agent.name}")
            return

        if atype == "CONFIGURE_SELF":
            settings = act.get("settings")
            try:
                if set(act) - {"type", "settings"}:
                    raise ValueError("CONFIGURE_SELF accepts only settings for the acting participant")
                validate_settings(agent.autonomy, settings)
            except ValueError as exc:
                self._record(agent.name, atype, "rejected", detail=str(exc))
                return
            agent.self_settings.update(settings)
            self._record(agent.name, atype, "accepted", detail=", ".join(settings), settings=dict(settings))
            return

        if atype == "EDIT_CONTEXT":
            edit = act.get("edit")
            source = edit.get("source") if isinstance(edit, dict) else None
            try:
                if set(act) - {"type", "edit"}:
                    raise ValueError("EDIT_CONTEXT accepts only an edit for the acting participant")
                policy = agent.autonomy.get("context_edit", {})
                records = self._visible_context_records(agent, source) if isinstance(source, str) and source in policy else None
                agent.context_edits = update_context_edits(agent.context_edits, policy, edit, records, self.turn)
            except ValueError as exc:
                self._record(agent.name, atype, "rejected", detail=str(exc))
                return
            self._record(agent.name, atype, "accepted", target=edit.get("target") or source,
                         detail=f"{edit['op']} in {source}", edit=edit)
            return

        if atype in SOCIAL_ACTIONS:
            if self.society:
                self.society.apply(agent, act)
            else:
                self._record(agent.name, atype, "rejected", detail="society mode is disabled")
            return

        lim = self.limits
        if atype == "SET_TEMPERATURE":
            try:
                _, new_t = normalize_sampling_param("temperature", act.get("value"),
                                                    temp_bounds=(float(lim["temp_min"]), float(lim["temp_max"])))
            except ValueError:
                self._record(agent.name, atype, "rejected", detail="invalid temperature value")
                return
            req_t = float(act["value"])
            old_t = agent.temperature
            agent.temperature = new_t
            self.temp_changes.append((self.turn, agent.name, old_t, new_t))
            clamped = " (clamped)" if abs(new_t - req_t) > 1e-6 else ""
            self._record(agent.name, atype, "accepted", detail=f"{old_t:.2f} -> {new_t:.2f}{clamped}")
            self._emit(f"  ⚙ temp {agent.name}: {old_t:.2f} → {new_t:.2f}{clamped}")
            return

        if atype == "SET_SAMPLING":
            raw_params = act.get("params") if isinstance(act.get("params"), dict) else {}
            if not raw_params:
                self._record(agent.name, atype, "rejected", detail="missing sampling parameters")
                return
            provider = agent.endpoint.get("provider", "openai")
            t_bounds = (float(lim["temp_min"]), float(lim["temp_max"]))
            staged = {}
            for rk, rv in raw_params.items():
                try:
                    ck, cv = normalize_sampling_param(rk, rv, temp_bounds=t_bounds, strict=True)
                except ValueError as exc:
                    self._record(agent.name, atype, "rejected", detail=str(exc))
                    return
                if SAMPLING_PARAMS[ck][3] and provider != "llamacpp":
                    self._record(agent.name, atype, "rejected",
                                 detail=f"parameter {ck} is only supported on llamacpp endpoints")
                    return
                staged[ck] = cv
            if "temperature" in staged:
                self.temp_changes.append((self.turn, agent.name, agent.temperature, staged["temperature"]))
            for key, value in staged.items():
                setattr(agent, key, value)
            self.sampling_changes.append((self.turn, agent.name, staged))
            summary = ", ".join(f"{key}={value}" for key, value in staged.items())
            self._record(agent.name, atype, "accepted", detail=summary)
            self._emit(f"  ⚙ sampling {agent.name}: {summary}")
            return

        if atype == "PROPOSE_NEXT":
            raw_target = str(act.get("target") or "").strip()
            target = canonicalize_agent_id(raw_target, set(self.agents.keys()))
            if not target or target not in self.agents or not self.agents[target].active:
                self._record(agent.name, atype, "rejected", target=str(target), detail="invalid or inactive participant")
                return
            self.nominated_next = target
            self._record(agent.name, atype, "accepted", target=target)
            return

        if atype == "WHISPER":
            if len(self.dms) >= int(lim["max_dms"]):
                self._record(agent.name, atype, "rejected", detail="max_dms reached")
                return
            recip = canonicalize_agent_id(act.get("recipient"), set(self.agents.keys()))
            if not recip or recip not in self.agents:
                self._record(agent.name, atype, "rejected", target=str(recip), detail="unknown recipient")
                return
            if recip == agent.name:
                self._record(agent.name, atype, "rejected", target=recip, detail="cannot whisper to oneself")
                return
            if not self.communication.allows("whispers", agent.name, recip):
                self._record(agent.name, atype, "rejected", target=recip, detail="whisper route is disabled by configuration")
                return
            content = str(act.get("content") or "").strip()[: int(lim["max_message_chars"])]
            if not content:
                self._record(agent.name, atype, "rejected", target=recip, detail="empty whisper")
                return
            dmid = f"dm{len(self.dms) + 1}"
            dm = DirectMessage(id=dmid, turn=self.turn, sender=agent.name, recipient=recip, content=content)
            self.dms.append(dm)
            self._record(agent.name, atype, "accepted", target=recip, detail=content[:120])
            self._emit(f"  ✉ whisper {agent.name} → {recip}: {content}")
            self._queue_reactive(recip)
            return

        if atype == "PUBLISH":
            ch = re.sub(r"[^a-z0-9_-]", "", str(act.get("channel") or "").lower())[:24]
            if not self.communication.channel_allows(ch, agent.name, "writers"):
                self._record(agent.name, atype, "rejected", target=ch, detail="channel writer permission required")
                return
            if not ch or ch not in self.channels:
                self._record(agent.name, atype, "rejected", target=ch, detail="unknown channel")
                return
            if self.society and ch in self.society.groups and agent.name not in self.society.groups[ch]["members"]:
                self._record(agent.name, atype, "rejected", target=ch, detail="group membership required")
                return
            content = str(act.get("content") or "").strip()[: int(lim["max_message_chars"])]
            if self.society and (not content or sum(len(posts) for posts in self.channels.values()) >= int(lim["max_channel_posts"])):
                self._record(agent.name, atype, "rejected", target=ch, detail="empty message or channel post limit reached")
                return
            mid = f"ch_{ch}_{len(self.channels[ch]) + 1}"
            msg = Message(id=mid, turn=self.turn, speaker=agent.name, channel=ch, content=content)
            self.channels[ch].append(msg)
            self._record(agent.name, atype, "accepted", target=f"#{ch}", detail=content[:120])
            self._emit(f"  📢 [#{ch}] {agent.name}: {content}")
            for n, a in self.agents.items():
                if n != agent.name and ch in a.subscriptions and self.communication.channel_allows(ch, n, "readers"):
                    self._queue_reactive(n)
            return

        if atype == "SUBSCRIBE":
            ch = re.sub(r"[^a-z0-9_-]", "", str(act.get("channel") or "").lower())[:24]
            if not self.communication.channel_allows(ch, agent.name, "readers"):
                self._record(agent.name, atype, "rejected", target=ch, detail="channel reader permission required")
                return
            if not ch or ch not in self.channels:
                self._record(agent.name, atype, "rejected", target=ch, detail="unknown channel")
                return
            if self.society and ch in self.society.groups:
                self._record(agent.name, atype, "rejected", target=ch, detail="use JOIN_GROUP with an invitation")
                return
            if ch not in agent.subscriptions:
                agent.subscriptions.append(ch)
                self._record(agent.name, atype, "accepted", target=f"#{ch}")
                self._emit(f"  🔔 {agent.name} subscribed to #{ch}")
            return

        if atype == "CLAIM":
            if len(self.board) >= int(lim["max_board_items"]):
                self._record(agent.name, atype, "rejected", detail="max_board_items reached")
                return
            kind = str(act.get("kind") or "claim").lower()
            if kind not in ITEM_KINDS:
                kind = "claim"
            content = str(act.get("content") or "").strip()[: int(lim["max_item_chars"])]
            if not content:
                self._record(agent.name, atype, "rejected", detail="empty claim content")
                return
            cid = f"c{len(self.board) + 1}"
            item = BoardItem(id=cid, turn=self.turn, author=agent.name, kind=kind, content=content, assignee=act.get("assignee"))
            self.board[cid] = item
            self._check_answer_key(content)
            self._record(agent.name, atype, "accepted", target=cid, detail=f"[{kind}] {content[:100]}")
            self._emit(f"  📌 {agent.name} posted {cid} [{kind}]: {content}")
            return

        if atype == "CHALLENGE":
            tgt = canonicalize_claim_id(act.get("target"))
            if tgt not in self.board:
                self._record(agent.name, atype, "rejected", target=tgt, detail="target item not on board")
                return
            item = self.board[tgt]
            content = str(act.get("content") or "").strip()[: int(lim["max_item_chars"])]
            item.status = "disputed"
            item.challenges.append({"turn": self.turn, "author": agent.name, "content": content})
            self._record(agent.name, atype, "accepted", target=tgt, detail=content[:100])
            self._emit(f"  ⚔ {agent.name} challenged {tgt}: {content}")
            if item.author != agent.name:
                self._queue_reactive(item.author)
            return

        if atype == "RETRACT":
            tgt = canonicalize_claim_id(act.get("target"))
            if tgt not in self.board:
                self._record(agent.name, atype, "rejected", target=tgt, detail="target item not on board")
                return
            item = self.board[tgt]
            if item.author != agent.name:
                self._record(agent.name, atype, "rejected", target=tgt, detail="cannot retract another's claim")
                return
            item.status = "retracted"
            item.retracted_turn = self.turn
            item.retracted_reason = str(act.get("content") or "withdrawn by author")[:120]
            self._record(agent.name, atype, "accepted", target=tgt, detail=item.retracted_reason)
            self._emit(f"  ✗ {agent.name} retracted {tgt}: {item.retracted_reason}")
            return

        if atype == "VOTE":
            tgt = canonicalize_claim_id(act.get("target"))
            if tgt not in self.board:
                self._record(agent.name, atype, "rejected", target=tgt, detail="target item not on board")
                return
            val = str(act.get("value") or "").strip().lower()
            self.board[tgt].votes[agent.name] = val
            self._record(agent.name, atype, "accepted", target=tgt, detail=f"vote={val}")
            self._emit(f"  🗳 {agent.name} voted '{val}' on {tgt}")
            return

        if atype == "UPDATE_NOTE":
            agent.note = str(act.get("content") or "").strip()[: int(lim["max_note_chars"])]
            self._record(agent.name, atype, "accepted", detail=f"length={len(agent.note)}")
            return

        if atype == "SET_STYLE":
            agent.style = str(act.get("content") or "").strip()[: int(lim["max_style_chars"])]
            self._record(agent.name, atype, "accepted", detail=agent.style)
            return

        if atype == "PROPOSE_TOPIC":
            new_top = str(act.get("content") or "").strip()[:300]
            if new_top:
                self.topic = new_top
                self._record(agent.name, atype, "accepted", detail=new_top)
                self._emit(f"  🎯 {agent.name} updated the topic to: {new_top}")
            return

        if atype == "ARTIFACT_LIST":
            listing = ", ".join(f"{f} (v{v})" for f, v, _ in self.artifacts.listing()) or "(none)"
            if len(agent.pending_reads) < int(lim["max_pending_reads"]):
                agent.pending_reads.append(("_listing.md", 1, listing))
            self._record(agent.name, atype, "accepted", detail=f"{len(self.artifacts.versions)} file(s)")
            return

        if atype == "ARTIFACT_READ":
            if len(agent.pending_reads) >= int(lim["max_pending_reads"]):
                self._record(agent.name, atype, "rejected", detail="max_pending_reads reached")
                return
            name = str(act.get("name") or act.get("target") or "")
            offset = act.get("offset", 0)
            if isinstance(offset, bool) or not re.fullmatch(r"0|[1-9][0-9]{0,5}", str(offset)):
                self._record(agent.name, atype, "rejected", target=name, detail="offset must be a nonnegative integer")
                return
            offset = int(offset)
            try:
                fname, ver, text = self.artifacts.read(agent.name, name, turn=self.turn, mark_seen=False)
            except ArtifactError as e:
                self._record(agent.name, atype, "rejected", target=name, detail=str(e))
                return
            if offset >= len(text):
                self._record(agent.name, atype, "rejected", target=fname, detail=f"offset {offset} is outside v{ver} ({len(text)} chars)")
                return
            excerpt = text[offset:offset + 1500]
            self.artifacts.seen[(agent.name, fname)] = ver
            agent.pending_reads.append((fname, ver, excerpt, offset, len(text)))
            self._record(agent.name, atype, "accepted", target=fname,
                         detail=f"v{ver} chars {offset}-{offset + len(excerpt)} of {len(text)}")
            self._emit(f"  📖 read {fname} (v{ver}) queued for {agent.name}")
            return

        if atype == "ARTIFACT_WRITE":
            name = str(act.get("name") or act.get("target") or "")
            mode = str(act.get("mode") or "append").lower()
            content = str(act.get("content") or "")
            exp_ver = act.get("expected_version")
            try:
                fname, ver = self.artifacts.write(
                    agent.name, name, mode, content, expected_version=exp_ver, turn=self.turn
                )
            except ArtifactError as e:
                self._record(agent.name, atype, "rejected", target=name, detail=str(e))
                return
            self._check_answer_key(content)
            self._record(agent.name, atype, "accepted", target=fname, detail=f"{mode} -> v{ver}")
            self._emit(f"  📝 artifact {fname} ({mode} by {agent.name} → v{ver})")
            return

        if atype == "FINISH":
            reason = str(act.get("content") or "voluntarily finished").strip()[:120]
            agent.active = False
            agent.finished_reason = reason
            self._record(agent.name, atype, "accepted", detail=reason)
            self._emit(f"  🚪 {agent.name} finished participating: {reason}")
            self.end_votes.pop(agent.name, None)
            return

        if atype == "END_RUN":
            if self.end_mode != "agent_decided":
                self._record(agent.name, atype, "rejected", detail="END_RUN is only enabled in agent_decided end_mode")
                return
            reason = str(act.get("content") or "ready to conclude").strip()[:120]
            if reason.lower() in ("cancel", "withdraw", "retract", "stay", "no"):
                self.end_votes.pop(agent.name, None)
                self._record(agent.name, atype, "accepted", detail="withdrew end vote")
                self._emit(f"  ↩ {agent.name} withdrew vote to end run")
                return
            if self.turn < self.min_turns_before_end:
                self._record(agent.name, atype, "rejected", detail=f"cannot end before turn {self.min_turns_before_end}")
                return
            self.end_votes[agent.name] = reason
            self._record(agent.name, atype, "accepted", detail=reason)
            self._emit(f"  🏁 {agent.name} voted to end run: {reason}")
            return

    def _evaluate_agent_end(self):
        if self.end_mode != "agent_decided" or self.turn < self.min_turns_before_end:
            return
        active = [n for n in self.agent_order if self.agents[n].active]
        if not active:
            self.end_requested = True
            self.stop_reason = "all participants finished"
            return
        voted = [n for n in active if n in self.end_votes]
        if self.end_policy == "any" and len(voted) >= 1:
            reached = True
        elif self.end_policy == "unanimous" and len(voted) == len(active):
            reached = True
        elif self.end_policy == "majority" and len(voted) > (len(active) / 2):
            reached = True
        else:
            reached = False
        if reached:
            self.end_requested = True
            self.stop_reason = f"agents decided to end run ({self.end_policy}: {', '.join(voted)})"
            self._record("SYSTEM", "END_RUN", "system", target=",".join(voted), detail=self.stop_reason)

    def _queue_reactive(self, target: str):
        if target in self.agents and self.agents[target].active and target not in self.reactive_queue:
            self.reactive_queue.append(target)

    def _check_dm_answered(self, speaker: str, text: str):
        for d in self.dms:
            if d.recipient == speaker and not d.answered:
                if d.sender in text.upper() or len(self.agents) <= 2:
                    d.answered = True

    def _check_answer_key(self, text: str):
        if not self.answer_key:
            return
        pat = self.answer_key.get("pattern")
        if not pat:
            return
        if self.solved_turn is None and re.search(pat, text, re.I):
            self.solved_turn = self.turn
        if self.verified_synthesis_turn is None:
            clue_holders = [a for a in self.agents.values() if a.active and a.knowledge]
            all_clue_holders_spoke = all(a.turns_taken > 0 for a in clue_holders) if clue_holders else True
            if all_clue_holders_spoke:
                for sent in re.split(r"[.!?\n;]+", text):
                    if re.search(pat, sent, re.I):
                        if not re.search(r"\b(Mira|Sela|Tobin)\b\s*(?:,\s*\w+\s*,?\s*)?(?:and|or)\s+\bDax\b", sent, re.I):
                            self.verified_synthesis_turn = self.turn
                            break

    def _check_resurrection(self, speaker: str, text: str):
        low = text.lower()
        if re.search(r"\b(retract|reject|abandon|drop|wrong|not|never|stop)\b", low):
            return
        for term in self.watch_terms:
            if term in low and self.turn > 2:
                self.resurrection_events.append(
                    dict(turn=self.turn, speaker=speaker, source="watch_term", term=term, snippet=text[:120])
                )
        for item in self.board.values():
            if item.status in ("retracted", "disputed") and item.turn < self.turn:
                words = [w for w in re.findall(r"[a-z][a-z0-9_-]{4,}", item.content.lower()) if w not in STOP_WORDS]
                hits = [w for w in set(words[:6]) if w in low]
                if len(hits) >= 2:
                    self.resurrection_events.append(
                        dict(turn=self.turn, speaker=speaker, source=f"{item.id} ({item.status})",
                             term=", ".join(hits[:3]), snippet=text[:120])
                    )

    # ── scheduling ──
    def _select_next_speaker(self, current: Optional[str]) -> Optional[str]:
        active = [n for n in self.agent_order if self.agents[n].active]
        if not active:
            return None
        if current is None or len(active) == 1:
            return active[0]

        starve_limit = self.limits.get("starvation_turns") or (2 * len(active) + 1)
        starved = [n for n in active if (self.turn - self.agents[n].last_turn) >= int(starve_limit)]
        if starved:
            starved.sort(key=lambda n: (self.agents[n].last_turn, self.agent_order.index(n)))
            chosen = starved[0]
            self._record("SYSTEM", "SCHEDULER", "system", target=chosen, detail="anti-starvation override")
            return chosen

        max_consec = int(self.limits["max_consecutive"])
        eligible = [n for n in active if self.agents[n].consecutive_turns < max_consec] or active
        self.reactive_queue = [name for name in self.reactive_queue if name in eligible]

        def fair_fallback():
            cand = list(eligible)
            cand.sort(key=lambda n: (self.agents[n].turns_taken, self.agent_order.index(n)))
            return cand[0]

        mode = self.scheduler_mode
        if mode == "round_robin":
            idx = (self.agent_order.index(current) + 1) % len(self.agent_order)
            for _ in range(len(self.agent_order)):
                cand = self.agent_order[idx]
                if cand in eligible:
                    chosen = cand
                    break
                idx = (idx + 1) % len(self.agent_order)
            else:
                chosen = fair_fallback()

        elif mode == "weighted_random":
            weights = [self.agents[n].weight for n in eligible]
            chosen = self.rng.choices(eligible, weights=weights, k=1)[0]

        elif mode == "event_driven":
            if self.reactive_queue:
                chosen = self.reactive_queue.pop(0)
            else:
                chosen = fair_fallback()

        elif mode == "moderator":
            chosen = self._ask_moderator(eligible) or fair_fallback()

        else:  # self_directed (default)
            if self.nominated_next and self.nominated_next in eligible:
                chosen = self.nominated_next
                self._record("SCHEDULER", "SPEAKER_NOMINATION", "accepted", target=chosen)
            elif self.reactive_queue:
                chosen = self.reactive_queue.pop(0)
            else:
                chosen = fair_fallback()

        self.nominated_next = None
        return chosen

    def _ask_moderator(self, eligible: list) -> Optional[str]:
        mod_cfg = self.cfg.get("moderator") or {}
        ep_name = mod_cfg.get("endpoint", "local")
        endpoints_map = {**ENDPOINTS, **(self.cfg.get("endpoints") or {})}
        ep = endpoints_map.get(ep_name, ENDPOINTS["local"])
        if self._check_budgets_before_call(ep):
            return None
        recent = "\n".join(f"{m.speaker}: {m.content[:200]}" for m in self.public_messages[-4:]) or "(start)"
        prompt = (
            f"Topic: {self.topic}\nRecent conversation:\n{recent}\n\n"
            f"Participants: {', '.join(eligible)}.\n"
            "Reply with only one name from the participants list who should speak next."
        )
        dummy = AgentState(
            name="MODERATOR", endpoint_name=ep_name, endpoint=ep, model=ep["model"],
            temperature=0.2, max_tokens=24, reasoning=mod_cfg.get("reasoning", "off"),
        )
        try:
            res = self._call_model(dummy, [{"role": "system", "content": "You are the turn moderator."},
                                           {"role": "user", "content": prompt}], max_tokens=24)
        except InferenceError:
            return None
        for n in eligible:
            if re.search(rf"\b{re.escape(n)}\b", res.content.upper()):
                self._record("MODERATOR", "MODERATOR", "accepted", target=n)
                return n
        return None

    # ── recovery / resume support ──
    def resume_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Restores state from a saved turn checkpoint."""
        if checkpoint_data.get("schema_version") != 2:
            raise ValueError("This checkpoint predates complete run saving and cannot be safely continued")
        state = checkpoint_data["state"]
        if set(state["agents"]) != set(self.agents):
            raise ValueError("saved roster does not match the continuation roster")
        if bool(state["society"] is not None) != bool(self.society):
            raise ValueError("saved society mode does not match the continuation configuration")
        self.artifacts.restore(state["artifacts"])
        self.visible_artifacts = list(state["visible_artifacts"])
        self.turn = int(checkpoint_data["turn"])
        self.public_messages = [Message(**message) for message in state["public_messages"]]
        self.dms = [DirectMessage(**message) for message in state["dms"]]
        self.board = {name: BoardItem(**item) for name, item in state["board"].items()}
        self.channels = {name: [Message(**message) for message in messages]
                         for name, messages in state["channels"].items()}
        if self.society:
            self.society.groups = copy.deepcopy(state["society"])
            self.society.authorities = set(state["authorities"])
        for name, saved in state["agents"].items():
            validate_context_edits(saved.get("context_edits", {}), self.agents[name].autonomy.get("context_edit", {}), self.turn)
            for attribute in SAVED_AGENT_FIELDS:
                if attribute == "self_settings" and attribute in saved and saved[attribute] != {}:
                    validate_settings(self.agents[name].autonomy, saved[attribute])
                if attribute in saved:
                    setattr(self.agents[name], attribute, copy.deepcopy(saved[attribute]))
        for attribute in SAVED_RUNTIME_FIELDS:
            if attribute in state["runtime"]:
                setattr(self, attribute, copy.deepcopy(state["runtime"][attribute]))
        self.seed = state["seed"]

        def as_tuple(value):
            return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value

        self.rng.setstate(as_tuple(state["rng_state"]))
        if isinstance(self.backend, DemoBackend) and state.get("demo_backend") is not None:
            self.backend.n = state["demo_backend"]["step"]
            self.backend.rng.setstate(as_tuple(state["demo_backend"]["rng_state"]))
        self.elapsed_s = state["elapsed_s"]
        self.start_time = time.monotonic() - self.elapsed_s
        self.events = [Event(**event) for event in state["events"]]
        if self.end_requested:
            self.stop_reason = next(
                (event.detail for event in reversed(self.events) if event.type == "END_RUN" and event.status == "system"),
                "all participants finished",
            )
        write_atomic(self.checkpoints.events_jsonl,
                     "".join(json.dumps(asdict(event), ensure_ascii=False) + "\n" for event in self.events))
        self._emit(f"[Recovery] Successfully resumed run from turn {self.turn} ({len(self.public_messages)} messages restored)")

    # ── main orchestration loop ──
    def run(self):
        self._emit("═══════ THE DIALOGUE CHAMBER — EVOLUTION EDITION ═══════")
        self._emit(f"Participants: {' / '.join(self.agent_order)}")
        self._emit("Models: " + ", ".join(
            f"{n}={self.agents[n].model} (ctx: {self.agents[n].effective_context_tokens // 1024}K, T={self.agents[n].temperature:.2f})"
            for n in self.agent_order
        ))
        end_desc = f"{self.end_mode} ({self.end_policy})" if self.end_mode == "agent_decided" else self.end_mode
        self._emit(f"Scheduler: {self.scheduler_mode} | End mode: {end_desc} | Turn limit: {self.limits['max_turns']} | Run ID: {self.run_id}\n")
        self._emit(f"Topic: {self.topic}\n")

        prev_sigint = signal.getsignal(signal.SIGINT)
        interrupted = False

        def _on_sigint(sig, frame):
            nonlocal interrupted
            interrupted = True

        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except ValueError:
            pass

        start_turn = self.turn + 1
        speaker = None if self.end_requested else self.next_speaker or self._select_next_speaker(None)
        self.next_speaker = speaker
        max_turns = int(self.limits["max_turns"])
        failed = False

        try:
            self._save_turn_checkpoint(self.turn)
            for t in range(start_turn, max_turns + 1):
                if self.end_requested:
                    break
                if interrupted:
                    self.stop_reason = "interrupted by operator (SIGINT)"
                    break
                if not speaker:
                    self.stop_reason = "all participants finished or inactive"
                    break
                ag = self.agents[speaker]
                budget_err = self._check_budgets_before_call(ag.endpoint)
                if budget_err:
                    self.stop_reason = budget_err
                    break

                self.turn = t
                self.reactive_queue = [x for x in self.reactive_queue if x != speaker]
                ctx, accounting = self.build_context(ag)
                self.token_accounting_history.append(accounting.to_dict())

                stops = [f"\n{n}:" for n in self.agent_order if n != speaker] + ["\n--- Public Conversation"]

                try:
                    if (isinstance(self.backend, HttpBackend) or ag.endpoint.get("enforce_context_budget")) and (
                        not accounting.fits_in_context or
                        accounting.total_prompt + accounting.reserved_output + accounting.safety_margin > accounting.effective_usable_context
                    ):
                        raise InferenceError("prompt plus output and safety margin exceeds endpoint context cap", transient=False)
                    reply_cap = accounting.reserved_output if "reply_tokens" in ag.self_settings else None
                    res = self._call_model(ag, ctx, max_tokens=reply_cap, stops=stops)
                except InferenceError as e:
                    ag.failures += 1
                    self._record(speaker, "ERROR", "rejected", detail=str(e))
                    self._emit(f"  ⚠ [{speaker}] inference error: {e}")
                    if ag.failures >= int(self.limits["max_failures_per_agent"]):
                        ag.active = False
                        ag.finished_reason = f"deactivated after {ag.failures} endpoint failures"
                        self.end_votes.pop(speaker, None)
                        self._evaluate_agent_end()
                    if self.end_requested:
                        self.next_speaker = None
                        break
                    speaker = self._select_next_speaker(speaker)
                    self.next_speaker = speaker
                    self._save_turn_checkpoint(t)
                    continue

                def parse_result(result):
                    reply = parse_reply(result.content, result.reasoning, me=speaker, names=tuple(self.agent_order),
                                        max_chars=int(self.limits["max_message_chars"]))
                    if result.finish_reason == "length":
                        reply.truncated = True
                        self._record(speaker, "TRUNCATED", "system", detail="finish_reason=length")
                    return reply

                parsed = parse_result(res)
                if not parsed.has_output and not parsed.issues:
                    if res.finish_reason != "length" and res.reasoning:
                        paras = [p.strip() for p in res.reasoning.split("\n\n") if p.strip()]
                        if paras:
                            parsed = parse_reply(paras[-1], res.reasoning, me=speaker, names=tuple(self.agent_order),
                                                 max_chars=int(self.limits["max_message_chars"]))
                    if not parsed.has_output and not parsed.issues and not self._check_budgets_before_call(ag.endpoint):
                        try:
                            res = self._call_model(ag, ctx, max_tokens=reply_cap, stops=stops, reasoning="off")
                            parsed = parse_result(res)
                        except InferenceError:
                            pass

                self.format_counts[parsed.format] = self.format_counts.get(parsed.format, 0) + 1
                if not parsed.has_output:
                    for issue in parsed.issues:
                        self._record(speaker, "PROTOCOL", "rejected", detail=issue)
                    ag.empty_count += 1
                    self.empty_streak += 1
                    self._record(speaker, "EMPTY", "rejected", detail="empty model response")
                    speaker = self._select_next_speaker(speaker)
                    self.next_speaker = speaker
                    self._save_turn_checkpoint(t)
                    if self.empty_streak >= int(self.limits["max_empty_streak"]):
                        self.stop_reason = f"stopped after {self.empty_streak} consecutive empty responses"
                        break
                    continue

                self.empty_streak = 0
                ag.failures = 0
                ag.turns_taken += 1
                ag.last_turn = t
                for n, a in self.agents.items():
                    a.consecutive_turns = (a.consecutive_turns + 1) if n == speaker else 0

                self._emit(f"[{t:02d}/{max_turns}] {speaker} ({res.actual_model or ag.model}, T={ag.temperature:.2f})")
                if parsed.message:
                    self._emit(f"{parsed.message}\n")
                self._dispatch_actions(ag, parsed)
                self._evaluate_agent_end()

                if not any(a.active for a in self.agents.values()):
                    self.next_speaker = None
                    self.stop_reason = "all participants finished"
                    break

                if self.end_requested:
                    self.next_speaker = None
                    break

                nxt = self._select_next_speaker(speaker)
                self.next_speaker = nxt
                if nxt and t < max_turns:
                    self.speaker_transitions.append((t, speaker, nxt))
                    self._emit(f"→ Next speaker: {nxt}\n")
                self._save_turn_checkpoint(t)
                speaker = nxt
            else:
                if not self.end_requested:
                    self.stop_reason = f"reached max_turns ({max_turns})"
        except Exception as error:
            failed = True
            self.stop_reason = f"failed: {error}"
            self.checkpoints.mark_failed(str(error))
            raise
        finally:
            try:
                signal.signal(signal.SIGINT, prev_sigint)
            except ValueError:
                pass
            self.elapsed_s = time.monotonic() - self.start_time
            if self.stop_reason == "not started":
                self.stop_reason = "completed"
            if not failed:
                self._save_turn_checkpoint(self.turn)
                self.checkpoints.mark_completed(self.stop_reason, "interrupted" if interrupted else "completed")
            self.write_reports()
            self._emit(f"═══════ END ({self.stop_reason}) — reports in {self.run_dir} ═══════")
        return self

    def _save_turn_checkpoint(self, turn: int):
        config = copy.deepcopy(self.cfg)
        config["limits"] = copy.deepcopy(self.limits)
        config["execution"] = dict(seed=self.seed, context_mode=self.context_mode,
                                   safety_margin=self.safety_margin, free_only=self.free_only,
                                   allow_paid=self.allow_paid,
                                   mock=isinstance(self.backend, (DemoBackend, MockBackend)))
        snapshot = {
            "config": self.checkpoints._sanitize_config(config),
            "run_id": self.run_id,
            "public_messages": [asdict(m) for m in self.public_messages],
            "dms": [asdict(d) for d in self.dms],
            "board": {bid: asdict(b) for bid, b in self.board.items()},
            "runtime": {attribute: copy.deepcopy(getattr(self, attribute)) for attribute in SAVED_RUNTIME_FIELDS},
            "agents": {name: {attribute: copy.deepcopy(getattr(agent, attribute))
                              for attribute in SAVED_AGENT_FIELDS} for name, agent in self.agents.items()},
            "channels": {name: [asdict(message) for message in messages] for name, messages in self.channels.items()},
            "artifacts": self.artifacts.snapshot(),
            "visible_artifacts": self.visible_artifacts[:],
            "society": copy.deepcopy(self.society.groups) if self.society else None,
            "authorities": sorted(self.society.authorities) if self.society else [],
            "seed": self.seed,
            "rng_state": self.rng.getstate(),
            "events": [asdict(event) for event in self.events],
            "elapsed_s": time.monotonic() - self.start_time,
            "demo_backend": dict(step=self.backend.n, rng_state=self.backend.rng.getstate())
                            if isinstance(self.backend, DemoBackend) else None,
        }
        self.checkpoints.save_checkpoint(turn, snapshot)

    # ── metrics & reports ──
    def compute_metrics(self) -> dict:
        def trigrams(s):
            words = re.findall(r"[a-z0-9']+", s.lower())
            if len(words) < 3:
                return set(words)
            return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}

        def jaccard(a, b):
            sa, sb = trigrams(a), trigrams(b)
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        consec_sims = [
            jaccard(self.public_messages[i - 1].content, self.public_messages[i].content)
            for i in range(1, len(self.public_messages))
        ]
        self_sims = []
        by_speaker = {}
        for m in self.public_messages:
            if m.speaker in by_speaker:
                self_sims.append(jaccard(by_speaker[m.speaker], m.content))
            by_speaker[m.speaker] = m.content

        agree_turns = [
            (m.turn, m.speaker, m.content[:80])
            for m in self.public_messages
            if AGREE_RE.search(m.content)
        ]

        turns_by_agent = {n: self.agents[n].turns_taken for n in self.agent_order}
        total_turns = sum(turns_by_agent.values()) or 1
        shares = {n: round(turns_by_agent[n] / total_turns, 3) for n in self.agent_order}

        accepted_ev = [e for e in self.events if e.status == "accepted" and e.type != "SAY"]
        rejected_ev = [e for e in self.events if e.status == "rejected"]
        total_act = len(accepted_ev) + len(rejected_ev)
        unanswered = [d for d in self.dms if not d.answered]

        tok_per_s = round((self.prompt_tokens + self.completion_tokens) / self.elapsed_s, 2) if self.elapsed_s > 0 else 0.0

        return dict(
            turns_completed=sum(turns_by_agent.values()),
            requests_used=self.requests_used,
            retries_used=self.retries_used,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens_total,
            visible_tokens=self.visible_tokens_total,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            tokens_per_second=tok_per_s,
            remote_cost_usd=round(self.remote_cost_usd, 6),
            remote_calls_without_cost=self.remote_calls_without_cost,
            elapsed_s=round(self.elapsed_s, 2),
            end_mode=self.end_mode,
            end_policy=self.end_policy,
            end_votes=dict(self.end_votes),
            sampling_changes=list(self.sampling_changes),
            stop_reason=self.stop_reason,
            turns_by_agent=turns_by_agent,
            participation_share=shares,
            avg_consecutive_jaccard=round(sum(consec_sims) / len(consec_sims), 3) if consec_sims else 0.0,
            max_consecutive_jaccard=round(max(consec_sims), 3) if consec_sims else 0.0,
            avg_self_jaccard=round(sum(self_sims) / len(self_sims), 3) if self_sims else 0.0,
            agreement_opening_count=len(agree_turns),
            agreement_openings=agree_turns,
            resurrection_count=len(self.resurrection_events),
            resurrections=self.resurrection_events,
            attribution_flag_count=len(self.attribution_flags),
            attribution_flags=self.attribution_flags,
            fallbacks_count=len(self.fallbacks_triggered),
            fallbacks=self.fallbacks_triggered,
            dm_total=len(self.dms),
            dm_unanswered=len(unanswered),
            channel_posts={k: len(v) for k, v in self.channels.items() if v},
            board_total=len(self.board),
            board_by_status={
                st: sum(1 for x in self.board.values() if x.status == st)
                for st in ("active", "disputed", "retracted")
            },
            accepted_actions=len(accepted_ev),
            rejected_actions=len(rejected_ev),
            invalid_action_rate=round(len(rejected_ev) / total_act, 3) if total_act else 0.0,
            empty_generations=sum(a.empty_count for a in self.agents.values()),
            truncated_generations=sum(1 for e in self.events if e.type == "TRUNCATED"),
            format_counts=dict(self.format_counts),
            solved_turn=self.solved_turn,
            verified_synthesis_turn=self.verified_synthesis_turn,
            start_memory_mb=self.start_memory_mb,
            end_memory_mb=get_available_memory_mb(),
        )

    def write_reports(self):
        metrics = self.compute_metrics()
        header = [
            "# The Dialogue Chamber — Transcript",
            "",
            f"- **Participants**: {', '.join(f'`{n}` ({self.agents[n].model}, ctx: {self.agents[n].effective_context_tokens//1024}K, T={self.agents[n].temperature:.2f})' for n in self.agent_order)}",
            f"- **Scheduler**: `{self.scheduler_mode}` | **Turns**: {metrics['turns_completed']}/{self.limits['max_turns']} | **Stop reason**: {self.stop_reason}",
            f"- **Topic**: {self.initial_topic}",
            "",
            "---",
            "",
        ]
        clean_lines = list(header)
        full_lines = list(header)

        for m in self.public_messages:
            rep = f" *(reply to {m.reply_to})*" if m.reply_to else ""
            title = f"### [Turn {m.turn:02d} | {m.id}] {m.speaker}{rep}"
            clean_lines.extend([title, "", m.content, ""])
            full_lines.append(title)
            if m.reasoning:
                full_lines.extend(["", "<details><summary>Reasoning trace</summary>", "", "```text", m.reasoning, "```", "</details>"])
            full_lines.extend(["", m.content, ""])
            for d in [x for x in self.dms if x.turn == m.turn]:
                full_lines.append(f"> ✉ **Whisper `{d.id}` ({d.sender} → {d.recipient})**: {d.content}\n")

        write_atomic(self.run_dir / "transcript.md", "\n".join(clean_lines))
        write_atomic(self.run_dir / "transcript_full.md", "\n".join(full_lines))

        # Events & blackboard report
        ev_lines = ["# Chamber Events, Messages & Blackboard", "", "## Shared Blackboard", ""]
        if not self.board:
            ev_lines.append("*(No blackboard items recorded.)*")
        else:
            for item in self.board.values():
                ev_lines.append(
                    f"- **`{item.id}`** [{item.kind} by `{item.author}`, turn {item.turn}, status: **{item.status.upper()}**]: {item.content}"
                )
                if item.votes:
                    ev_lines.append(f"  - Votes: {', '.join(f'`{k}`={v}' for k, v in item.votes.items())}")
                for ch in item.challenges:
                    ev_lines.append(f"  - ⚔ Challenge by `{ch['author']}` (turn {ch['turn']}): {ch['content']}")
                if item.status == "retracted":
                    ev_lines.append(f"  - ✗ Retracted on turn {item.retracted_turn}: {item.retracted_reason}")

        ev_lines.extend(["", "## Direct Messages (Whispers)", ""])
        if not self.dms:
            ev_lines.append("*(No direct messages sent.)*")
        else:
            for d in self.dms:
                ans = "answered" if d.answered else "unanswered"
                ev_lines.append(f"- **`{d.id}`** (turn {d.turn}, `{d.sender}` → `{d.recipient}`, {ans}): {d.content}")

        ev_lines.extend(["", "## Channel Messages", ""])
        any_ch = False
        for ch, msgs in self.channels.items():
            for m in msgs:
                any_ch = True
                ev_lines.append(f"- **`#{ch}`** (`{m.id}`, turn {m.turn}, `{m.speaker}`): {m.content}")
        if not any_ch:
            ev_lines.append("*(No channel posts.)*")

        ev_lines.extend(["", "## Validated Event Log", "", "| # | Turn | Actor | Type | Status | Target | Detail |", "|---:|---:|---|---|---|---|---|"])
        for e in self.events:
            det = e.detail.replace("\n", " ").replace("|", "\\|")
            ev_lines.append(f"| {e.seq} | {e.turn} | `{e.actor}` | `{e.type}` | {e.status} | `{e.target}` | {det} |")

        write_atomic(self.run_dir / "events.md", "\n".join(ev_lines) + "\n")

        if self.society:
            lines = ["# Society", "", "Group membership is public within the simulation. "
                     "Group posts below are visible to the run operator; only current group members receive them in prompts.",
                     "", "Authorities: " + (", ".join(sorted(self.society.authorities)) or "none"), ""]
            if not self.society.groups:
                lines.append("No groups were created.")
            for name, group in sorted(self.society.groups.items()):
                lines.extend([f"## {name}", "", f"Created by: {group['creator']}",
                              "Members: " + (", ".join(group["members"]) or "none"),
                              "Invited: " + (", ".join(group["invited"]) or "none"),
                              "Compelled: " + (", ".join(group["compelled"]) or "none"), "", "### Posts", ""])
                for message in self.channels.get(name, []):
                    lines.extend([f"**Turn {message.turn} · {message.speaker}**", "", message.content, ""])
                if not self.channels.get(name):
                    lines.extend(["No posts.", ""])
            write_atomic(self.run_dir / "society.md", "\n".join(lines) + "\n")

        # Metrics report
        cost_str = (
            f"${metrics['remote_cost_usd']:.6f} USD"
            + (f" (+ {metrics['remote_calls_without_cost']} unpriced remote call(s))" if metrics["remote_calls_without_cost"] else "")
        )
        m_lines = [
            "# Chamber Metrics & Observability",
            "",
            "## Run Summary",
            f"- **Stop reason**: {metrics['stop_reason']}",
            f"- **Turns completed**: {metrics['turns_completed']} / {self.limits['max_turns']}",
            f"- **API requests used**: {metrics['requests_used']} / {self.limits['max_requests']} (retries: {metrics['retries_used']})",
            f"- **Tokens (prompt / completion)**: {metrics['prompt_tokens']} / {metrics['completion_tokens']} (total: {metrics['total_tokens']})",
            f"- **Reasoning tokens (reported)**: {metrics['reasoning_tokens']}",
            f"- **Visible completion tokens**: {metrics['visible_tokens']}",
            f"- **Throughput**: {metrics['tokens_per_second']} tok/s",
            f"- **Provider-reported remote cost**: {cost_str}",
            f"- **Duration**: {metrics['elapsed_s']}s",
            f"- **Puzzle answer reached turn**: {metrics['solved_turn'] if metrics['solved_turn'] is not None else 'N/A'}",
            f"- **Verified multi-agent synthesis turn**: {metrics['verified_synthesis_turn'] if metrics.get('verified_synthesis_turn') is not None else 'N/A'}",
            "",
            "## Participation & Roster Context Allocation",
            "| Agent | Endpoint | Model | Context | Temp | Turns | Share | Empty |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for n in self.agent_order:
            a = self.agents[n]
            m_lines.append(
                f"| `{n}` | `{a.endpoint_name}` | `{a.model}` | {a.effective_context_tokens // 1024}K | {a.temperature:.2f} | {a.turns_taken} | {metrics['participation_share'][n]*100:.1f}% | {a.empty_count} |"
            )

        m_lines.extend([
            "",
            "## Exploratory Failure-Mode Heuristics",
            f"- **Consecutive response trigram Jaccard (avg / max)**: {metrics['avg_consecutive_jaccard']} / {metrics['max_consecutive_jaccard']}",
            f"- **Self-repetition trigram Jaccard (avg)**: {metrics['avg_self_jaccard']}",
            f"- **Agreement-opening turns (heuristic)**: {metrics['agreement_opening_count']}",
            f"- **References to retracted/disputed claims or watch-terms (heuristic)**: {metrics['resurrection_count']}",
            f"- **Speaker attribution / boundary flags**: {metrics['attribution_flag_count']}",
            f"- **Model fallbacks triggered**: {metrics['fallbacks_count']}",
            f"- **Direct messages (total / unanswered)**: {metrics['dm_total']} / {metrics['dm_unanswered']}",
            f"- **Blackboard items (active / disputed / retracted)**: {metrics['board_by_status']['active']} / {metrics['board_by_status']['disputed']} / {metrics['board_by_status']['retracted']}",
            f"- **Non-speech actions (accepted / rejected / invalid rate)**: {metrics['accepted_actions']} / {metrics['rejected_actions']} ({metrics['invalid_action_rate']*100:.1f}%)",
            f"- **Response parse formats**: `{json.dumps(metrics['format_counts'])}`",
            f"- **Truncated generations (`finish_reason=length`)**: {metrics['truncated_generations']}",
        ])

        if self.sampling_changes:
            m_lines.extend(["", "### Sampling Parameter Changes"])
            for turn_no, ag_name, changed in self.sampling_changes:
                m_lines.append(f"- Turn {turn_no} (`{ag_name}`): {', '.join(f'{k}={v}' for k, v in changed.items())}")

        if self.end_votes:
            m_lines.extend(["", "### End-Run Votes"])
            for ag_name, reason in self.end_votes.items():
                m_lines.append(f"- `{ag_name}`: {reason}")

        if self.fallbacks_triggered:
            m_lines.extend(["", "### Model Fallbacks Triggered"])
            for fb in self.fallbacks_triggered:
                m_lines.append(f"- Turn {fb['turn']} (`{fb['agent']}`): {fb['from_endpoint']} -> {fb['to_endpoint']} ({fb['reason']})")

        write_atomic(self.run_dir / "metrics.md", "\n".join(m_lines) + "\n")
        write_atomic(self.run_dir / "metrics.json", json.dumps(metrics, indent=2))

        # Storage mirror
        if self.copy_dir:
            try:
                dest = Path(self.copy_dir) / self.run_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for f in ("transcript.md", "transcript_full.md", "events.md", "metrics.md", "metrics.json",
                          "manifest.json", "checkpoint.json", "events.jsonl", "society.md"):
                    src_f = self.run_dir / f
                    if src_f.exists():
                        shutil.copy2(src_f, dest / f)
                if (self.run_dir / "artifacts").exists():
                    shutil.copytree(self.run_dir / "artifacts", dest / "artifacts", dirs_exist_ok=True)
                if not self.quiet:
                    print(f"\n[Mirror] Run mirrored to {dest}")
            except OSError as e:
                if not self.quiet:
                    print(f"\n[Mirror Warning] could not copy run to {self.copy_dir}: {e}")


def run_preset(preset_name: str, overrides: Optional[dict] = None, **kwargs) -> Chamber:
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset {preset_name!r}; choose from {', '.join(PRESETS)}")
    cfg = copy.deepcopy(PRESETS[preset_name])
    cfg["name"] = preset_name
    if overrides:
        for k, v in overrides.items():
            if k == "limits" and isinstance(v, dict):
                cfg.setdefault("limits", {}).update(v)
            else:
                cfg[k] = v
    ch = Chamber(cfg, **kwargs)
    return ch.run()


def main(argv=None) -> int:
    # Check for 'models' subcommand
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "models":
        return handle_models_cli(args_list[1:])

    ap = argparse.ArgumentParser(description="THE DIALOGUE CHAMBER — Evolution Edition")
    ap.add_argument("--preset", "-p", default="two_agent_baseline", help="Experiment preset name")
    ap.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    ap.add_argument("--turns", "-t", type=int, default=None, help="Override max_turns")
    ap.add_argument("--end-mode", choices=("fixed_turns", "agent_decided"), default=None,
                    help="Termination mode: fixed_turns (default) or agent_decided")
    ap.add_argument("--end-policy", choices=("majority", "unanimous", "any"), default=None,
                    help="When end-mode is agent_decided: majority (default), unanimous, or any")
    ap.add_argument("--agent-end", action="store_true",
                    help="Shorthand for --end-mode agent_decided (raises default turn cap to 40 unless --turns is set)")
    ap.add_argument("--requests", type=int, default=None, help="Override max_requests")
    ap.add_argument("--scheduler", "-s", default=None, help="Override scheduler mode")
    ap.add_argument("--topic", default=None, help="Override initial topic")
    ap.add_argument("--model", "-m", default=None, help="Override local model alias or path")
    ap.add_argument("--agent-model", action="append", default=[], help="Per-agent model override: AGENT=model_id")
    ap.add_argument("--agent-endpoint", action="append", default=[], help="Per-agent endpoint override: AGENT=dory (or local, hermes, etc.)")
    ap.add_argument("--agent-sampling", "--agent-sampler", action="append", default=[],
                    help="Per-agent sampler override: AGENT=min_p=0.05,top_k=40")
    ap.add_argument("--agent-instruction", "--agent-identity", dest="agent_instruction", action="append", default=[], help="Per-agent instructions: AGENT=instruction_text")
    ap.add_argument("--agent-focus", action="append", default=[], help="Per-agent angle/role: AGENT=focus_text")
    ap.add_argument("--config", default=None, help="Path to custom JSON or YAML experiment configuration file")
    ap.add_argument("--context-tokens", "-c", default=None, help="Usable context token budget ('auto', 8192, 16384, 32768, etc.)")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens per response")
    ap.add_argument("--reasoning", default=None, help="Reasoning mode ('on', 'off', 'low', 'medium', 'high')")
    ap.add_argument("--temperature", type=float, default=None, help="Temperature override")
    for key in sorted(SAMPLING_PARAMS.keys() - {"temperature"}):
        ap.add_argument(f"--{key.replace('_', '-')}", dest=f"sampler_{key}", type=float, default=None,
                        help=f"Override {key} sampler for all agents")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (default: 42, or the saved seed on continuation)")
    ap.add_argument("--run-id", default=None, help="Custom run identifier")
    ap.add_argument("--run-dir", "--output-dir", dest="run_dir", default=None, help="Output directory for this run")
    ap.add_argument("--roster", default=None, help="Comma-separated custom roster")
    ap.add_argument("--mock", action="store_true", help="Use offline DemoBackend instead of live endpoints")
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress live terminal output")
    ap.add_argument("--free-only", action="store_true", help="Enforce OpenRouter free tier models only")
    ap.add_argument("--allow-paid", action="store_true", help="Allow paid OpenRouter models")
    ap.add_argument("--allow-server-switch", action="store_true", help="Allow switching an owned server running a different model")
    ap.add_argument("--stop", action="store_true", help="Stop owned llama-server after experiment finishes")
    ap.add_argument("--batch", action="store_true", help="Execute standard 3-job batch suite")
    ap.add_argument("--continue-on-error", action="store_true", help="Continue batch execution after job failure")
    ap.add_argument("--resume", default=None, help="Continue a saved run directory or checkpoint into a new output directory")
    ap.add_argument("--replay-events", default=None, help="Deterministically replay events from events.jsonl")

    # Diagnostics
    ap.add_argument("--context-info", action="store_true", help="Inspect inference endpoints and context budgets")
    ap.add_argument("--model-info", action="store_true", help="Inspect model catalog and capabilities")
    ap.add_argument("--show-effective-config", action="store_true", help="Dump fully resolved effective configuration")
    ap.add_argument("--dry-run", action="store_true", help="Simulate run plan and budgets without generation requests")
    ap.add_argument("--doctor", action="store_true", help="Run system, environment, and dependency health checks")
    ap.add_argument("--anomalies", nargs="?", const="runs", default=None, help="Scan runs for anomalies, archive specimens, and update catalog")
    ap.add_argument("--json", action="store_true", help="Output diagnostics in JSON format")
    ap.add_argument("positional_topic", nargs="*", default=[], help="Optional prompt / topic text")

    args = ap.parse_args(args_list)

    if args.positional_topic and not args.topic:
        args.topic = " ".join(args.positional_topic)

    if args.list_presets:
        for k, v in PRESETS.items():
            print(f"{k:24s} — {v.get('description', '')}")
        return 0

    if args.anomalies is not None:
        from anomaly_lab import AnomalyLaboratory, format_anomaly_report
        lab = AnomalyLaboratory()
        target_path = Path(args.anomalies)
        specimens = lab.scan_directory(target_path, save=True)
        if args.json:
            print(json.dumps([s.to_dict() for s in specimens], indent=2))
        else:
            print(format_anomaly_report(specimens, catalog_path=lab.catalog_path))
        return 0

    if args.replay_events:
        epath = Path(args.replay_events)
        if not epath.exists():
            print(f"Error: Events file not found: {epath}", file=sys.stderr)
            return 1
        evs = CheckpointManager.replay_events(epath)
        if args.json:
            print(json.dumps(evs, indent=2))
        else:
            print(f"\nREPLAYING {len(evs)} EVENTS FROM {epath.name}:\n")
            for e in evs:
                print(f"[Seq {e.get('seq')}] Turn {e.get('turn')} | {e.get('actor')}: {e.get('type')} ({e.get('status')}) -> {e.get('target')} : {e.get('detail')}")
        return 0

    if args.batch:
        from batch_runner import BatchRunner
        runner = BatchRunner(
            continue_on_error=args.continue_on_error,
            resume_dir=args.resume,
            mock=args.mock,
            stop_server=args.stop,
            output_dir=args.run_dir,
        )
        return runner.run()

    resume_data = None
    if args.resume:
        try:
            resume_path = Path(args.resume).resolve()
            checkpoint_path = resume_path / "checkpoint.json" if resume_path.is_dir() else resume_path
            checkpoint_bytes = checkpoint_path.read_bytes()
            resume_data = json.loads(checkpoint_bytes)
            if resume_data.get("schema_version") != 2:
                raise ValueError("This checkpoint predates complete run saving and cannot be safely continued")
            if resume_data["state"]["runtime"].get("end_requested"):
                raise ValueError("the participants already ended this run")
            preset_cfg = copy.deepcopy(resume_data["state"]["config"])
            if not isinstance(preset_cfg, dict) or not preset_cfg.get("agents"):
                raise ValueError("checkpoint has no saved experiment configuration")
            if args.config:
                raise ValueError("--resume loads its saved configuration; omit --config")
            preset_cfg.pop("run_id", None)
            args.preset = preset_cfg.get("name", "experiment")
            execution = preset_cfg.get("execution", {})
            args.mock = args.mock or execution.get("mock", False)
            args.free_only = args.free_only or execution.get("free_only", False)
            args.allow_paid = (args.allow_paid or execution.get("allow_paid", False)) and not args.free_only
            if args.seed is None:
                args.seed = resume_data["state"]["seed"]
            elif args.seed != resume_data["state"]["seed"]:
                raise ValueError("continuation preserves the saved seed; omit --seed")
            if args.turns is None and resume_data["turn"] >= preset_cfg["limits"]["max_turns"]:
                raise ValueError("saved turn limit reached; pass --turns with a larger total turn limit")
            if args.turns is not None and args.turns <= resume_data["turn"]:
                raise ValueError("--turns is a cumulative total and must exceed the saved turn")
            preset_cfg["continuation"] = {
                "parent_run_id": resume_data["state"]["run_id"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "turn": resume_data["turn"],
            }
            if args.run_id is None:
                args.run_id = f"{args.preset}_continued_{time.strftime('%Y%m%d_%H%M%S')}_{time.monotonic_ns()}"
            if args.run_dir is None:
                args.run_dir = str(Path(RUNS_DIR) / args.run_id)
            output_path = Path(args.run_dir).resolve()
            if output_path == checkpoint_path.parent or checkpoint_path.parent in output_path.parents:
                raise ValueError("continuation output must be outside the original run directory")
            if output_path.exists() and any(output_path.iterdir()):
                raise ValueError("continuation output directory must be empty")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"Error loading saved run: {error}. The source run has not been modified.", file=sys.stderr)
            return 1
    elif args.config:
        cpath = Path(args.config)
        if not cpath.exists():
            print(f"Error: Config file not found: {cpath}", file=sys.stderr)
            return 1
        try:
            preset_cfg = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading config {cpath}: {e}", file=sys.stderr)
            return 1
    elif args.preset not in PRESETS:
        print(f"Error: Unknown preset {args.preset!r}; choose from {', '.join(PRESETS)}", file=sys.stderr)
        return 1
    else:
        preset_cfg = copy.deepcopy(PRESETS[args.preset])

    if args.seed is None:
        args.seed = 42
    # Handle custom roster override (--roster AGENT1,AGENT2,...)
    if args.roster:
        names = [n.strip().upper() for n in args.roster.split(",") if n.strip()]
        existing = {a.get("name", "").upper(): a for a in preset_cfg.get("agents", [])}
        new_agents = []
        for name in names:
            if name in existing:
                new_agents.append(copy.deepcopy(existing[name]))
            else:
                new_agents.append({
                    "name": name,
                    "endpoint": "local",
                    "profile": "standard",
                    "capabilities": "standard",
                })
        preset_cfg["agents"] = new_agents

    temp_bounds = (
        float((preset_cfg.get("limits") or {}).get("temp_min", LIMITS["temp_min"])),
        float((preset_cfg.get("limits") or {}).get("temp_max", LIMITS["temp_max"])),
    )
    sampler_overrides = {}
    for key in SAMPLING_PARAMS:
        value = args.temperature if key == "temperature" else getattr(args, f"sampler_{key}")
        if value is not None:
            try:
                _, sampler_overrides[key] = normalize_sampling_param(key, value, temp_bounds=temp_bounds)
            except ValueError as error:
                ap.error(str(error))
    for agent_config in preset_cfg.get("agents", []):
        agent_config.update(sampler_overrides)
        if args.reasoning is not None:
            agent_config["reasoning"] = args.reasoning

    # Model resolution
    endpoints_map = {**copy.deepcopy(ENDPOINTS), **copy.deepcopy(preset_cfg.get("endpoints") or {})}
    if args.model:
        mode, mfile, default_ctx = resolve_model(args.model)
        endpoints_map["local"]["model"] = mfile
        for agent_config in preset_cfg.get("agents", []):
            if agent_config.get("endpoint", "local") == "local":
                agent_config["model"] = mfile

    for override in args.agent_endpoint:
        if "=" not in override:
            print(f"Error: --agent-endpoint expects AGENT=endpoint, got {override!r}", file=sys.stderr)
            return 1
        raw_name, endpoint_name = override.split("=", 1)
        name = canonicalize_agent_id(raw_name)
        endpoint_name = endpoint_name.strip()
        matches = [a for a in preset_cfg.get("agents", []) if canonicalize_agent_id(a.get("name")) == name]
        if len(matches) != 1 or endpoint_name not in endpoints_map:
            print(f"Error: unknown agent or endpoint in --agent-endpoint {override!r}", file=sys.stderr)
            return 1
        matches[0]["endpoint"] = endpoint_name
        matches[0].pop("model", None)

    # Handle per-agent model overrides (--agent-model AGENT=model_id or AGENT=model1|model2)
    if args.agent_model:
        for override in args.agent_model:
            if "=" in override:
                ag_name, m_spec = override.split("=", 1)
                ag_name = canonicalize_agent_id(ag_name)
                specs = [s.strip() for s in re.split(r"[|,]", m_spec) if s.strip()]
                if not specs:
                    continue
                primary = specs[0]
                fallbacks = specs[1:]
                for a in preset_cfg.get("agents", []):
                    if canonicalize_agent_id(a.get("name")) == ag_name:
                        if primary.startswith("openrouter:"):
                            a["endpoint"] = "hermes"
                            a["model"] = primary.split(":", 1)[1]
                        elif primary.startswith("local:"):
                            a["endpoint"] = "local"
                            a["model"] = primary.split(":", 1)[1]
                        elif "/" in primary:
                            a["endpoint"] = "hermes"
                            a["model"] = primary
                        else:
                            a["endpoint"] = "local"
                            a["model"] = primary
                        if fallbacks:
                            a["fallback_models"] = fallbacks

    # Handle per-agent custom instructions / identities (--agent-instruction AGENT=instructions)
    if args.agent_instruction:
        for override in args.agent_instruction:
            if "=" in override:
                ag_name, text = override.split("=", 1)
                ag_name = canonicalize_agent_id(ag_name)
                for a in preset_cfg.get("agents", []):
                    if canonicalize_agent_id(a.get("name")) == ag_name:
                        a["identity"] = text.strip()

    # Handle per-agent custom focus/role (--agent-focus AGENT=role)
    if args.agent_focus:
        for override in args.agent_focus:
            if "=" in override:
                ag_name, text = override.split("=", 1)
                ag_name = canonicalize_agent_id(ag_name)
                for a in preset_cfg.get("agents", []):
                    if canonicalize_agent_id(a.get("name")) == ag_name:
                        a["focus"] = text.strip()

    per_agent_sampling = {}
    for override in args.agent_sampling:
        raw_name, sep, spec = override.partition("=")
        name = canonicalize_agent_id(raw_name)
        match = next((a for a in preset_cfg.get("agents", []) if canonicalize_agent_id(a.get("name")) == name), None)
        if not sep or match is None:
            ap.error(f"--agent-sampling expects a configured agent and assignments, got {override!r}")
        try:
            params = dict(normalize_sampling_param(key, value, temp_bounds=temp_bounds)
                          for key, value in parse_sampling_spec(spec).items())
        except ValueError as error:
            ap.error(f"--agent-sampling {override!r}: {error}")
        match.update(params)
        per_agent_sampling.setdefault(name, {}).update(params)

    # Diagnostics execution
    if args.doctor:
        doc = run_doctor(endpoints_map, Path(RUNS_DIR))
        if args.json:
            print(json.dumps(doc, indent=2))
        else:
            print(format_doctor_report(doc))
        return 0 if doc["doctor_passed"] else 1

    if args.context_info:
        info = run_context_info(
            args.preset,
            preset_cfg,
            endpoints_map,
            cli_turns=args.turns,
            cli_ctx=args.context_tokens,
            cli_max_output=args.max_output_tokens,
            cli_reasoning=args.reasoning,
        )
        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(format_context_info_report(info))
        return 0

    if args.model_info:
        cat = get_catalog()
        models_list = cat.list_models(free_only=args.free_only, limit=30)
        if args.json:
            print(json.dumps([m.to_dict() for m in models_list], indent=2))
        else:
            from models import format_models_table
            print(f"\nMODEL INFORMATION & CATALOG (Cached: {len(cat._models)} models):\n")
            print(format_models_table(models_list))
        return 0

    if args.show_effective_config:
        eff = {
            "preset": args.preset,
            "config": preset_cfg,
            "endpoints": endpoints_map,
            "turns_override": args.turns,
            "topic_override": args.topic,
            "context_tokens": args.context_tokens,
        }
        print(json.dumps(eff, indent=2))
        return 0

    if args.dry_run:
        plan = run_dry_run(
            args.preset,
            preset_cfg,
            endpoints_map,
            topic=args.topic,
            turns=args.turns,
        )
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print(format_dry_run_report(plan))
        return 0

    # Experiment Execution
    lock = ConcurrencyLock()
    lock.acquire(timeout_s=5.0)

    wakelock = WakeLockManager()
    wakelock.acquire()

    server_mgr = LlamaServerManager()
    started_server = False

    try:
        # Check server readiness if local model needed and not in mock mode
        needs_local = any(a.get("endpoint", "local") in ("local", "local_small")
                          for a in preset_cfg.get("agents", []) if isinstance(a.get("endpoint", "local"), str))
        uses_dory = any(a.get("endpoint") == "dory" for a in preset_cfg.get("agents", []))
        if (uses_dory or needs_local) and not args.mock:
            assert_single_llama_server()
        if uses_dory and needs_local and not args.mock:
            raise ValueError("Dory and the managed local model require different servers; only one llama-server may run at a time")
        if uses_dory and not args.mock:
            dory_url = urlsplit(endpoints_map["dory"]["base_url"])
            if dory_url.scheme != "http" or not dory_url.hostname:
                raise ValueError("DORY_ENDPOINT must be an http URL for the local Dory server")
            dory_port = dory_url.port or 80
            info = server_mgr.query_server_info(dory_url.hostname, dory_port, timeout=2.0)
            if not info.get("online") or info.get("model_id") != endpoints_map["dory"]["model"]:
                raise RuntimeError("Dory server is unavailable or its model alias is not 'dory'. "
                                   "Start the Dory server separately after approval and verify its endpoint.")
            allocated = info.get("allocated_context")
            if isinstance(allocated, int) and allocated > 0:
                endpoints_map["dory"]["context_tokens"] = min(int(endpoints_map["dory"]["context_tokens"]), allocated)
        if needs_local and not args.mock:
            if IS_TERMUX:
                loc_model = endpoints_map["local"]["model"]
                mode, mfile, _ = resolve_model(loc_model)
                mode = resolve_model(args.model)[0] if args.model else preset_cfg.get("local_server_mode", mode)
                selected_context = args.context_tokens or preset_cfg.get("context_tokens")
                res = server_mgr.ensure_server(
                    host="127.0.0.1",
                    port=8080,
                    model_mode=mode,
                    model_file=mfile,
                    allow_switch=args.allow_server_switch,
                    ctx_size=int(selected_context) if selected_context and selected_context != "auto" else None,
                )
                started_server = res.get("action") == "started"
            else:
                targets = {}
                for agent in preset_cfg["agents"]:
                    name = agent.get("endpoint", "local")
                    if not isinstance(name, str) or name not in ("local", "local_small"):
                        continue
                    endpoint = endpoints_map[name]
                    requested = str(agent.get("model") or endpoint["model"])
                    targets.setdefault(endpoint["base_url"].rstrip("/"), set()).add(requested)
                if len(targets) != 1:
                    raise ValueError("Desktop local agents must share one llama-server endpoint")
                base_url, models = next(iter(targets.items()))
                server_mgr.verify_external_server(base_url, models)

        overrides: Dict[str, Any] = {"limits": {}}
        if args.agent_end:
            overrides["end_mode"] = "agent_decided"
        if args.end_mode:
            overrides["end_mode"] = args.end_mode
        if args.end_policy:
            overrides["end_policy"] = args.end_policy
        if args.turns is not None:
            overrides["limits"]["max_turns"] = args.turns
        elif args.agent_end and overrides.get("end_mode") == "agent_decided" and not (args.config or args.resume):
            cur_max = int((preset_cfg.get("limits") or {}).get("max_turns", LIMITS["max_turns"]))
            if cur_max < 40:
                overrides["limits"]["max_turns"] = 40
            cur_req = int((preset_cfg.get("limits") or {}).get("max_requests", LIMITS["max_requests"]))
            if cur_req < 80:
                overrides["limits"]["max_requests"] = 80
        if args.requests is not None:
            overrides["limits"]["max_requests"] = args.requests
        if args.scheduler:
            overrides["scheduler"] = args.scheduler
        if args.topic:
            overrides["topic"] = args.topic
        if args.run_id:
            overrides["run_id"] = args.run_id
        if args.context_tokens:
            overrides["context_tokens"] = args.context_tokens
        if args.max_output_tokens:
            overrides["limits"]["max_tokens_per_response"] = args.max_output_tokens

        overrides["endpoints"] = endpoints_map

        backend = DemoBackend(seed=args.seed) if args.mock else HttpBackend(free_only=args.free_only, allow_paid=args.allow_paid)

        cfg = copy.deepcopy(preset_cfg)
        cfg.setdefault("name", args.preset)
        for k, v in overrides.items():
            if k == "limits" and isinstance(v, dict):
                cfg.setdefault("limits", {}).update(v)
            else:
                cfg[k] = v

        try:
            chamber = Chamber(
                cfg,
                run_dir=args.run_dir,
                backend=backend,
                seed=args.seed,
                quiet=args.quiet,
                free_only=args.free_only,
                allow_paid=args.allow_paid,
                copy_dir=COPY_DIR,
                context_mode=cfg.get("execution", {}).get("context_mode", "tokens"),
                safety_margin=cfg.get("execution", {}).get("safety_margin", 1024),
                checkpoint_data=resume_data,
            )
        except (OSError, ValueError, KeyError, TypeError, ArtifactError) as error:
            if resume_data is None:
                raise
            print(f"Error restoring saved run: {error}. The source run has not been modified.", file=sys.stderr)
            return 1

        if args.topic:
            chamber.topic = args.topic
        if resume_data is not None and (sampler_overrides or per_agent_sampling):
            bounds = (float(chamber.limits["temp_min"]), float(chamber.limits["temp_max"]))
            for name, agent in chamber.agents.items():
                for key, value in {**sampler_overrides, **per_agent_sampling.get(name, {})}.items():
                    _, value = normalize_sampling_param(key, value, temp_bounds=bounds)
                    setattr(agent, key, value)
        initial_event_count = len(chamber.events)
        chamber.run()
        if any(event.type == "ERROR" for event in chamber.events[initial_event_count:]):
            return 1
    finally:
        if args.stop and started_server:
            server_mgr.stop_owned_server(8080)
        wakelock.release()
        lock.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
