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

from artifacts import ArtifactError, ArtifactManager, validate_artifact
from checkpoint import CheckpointManager, RunManifest, write_atomic
from documents import DocumentError, DocumentLibrary
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
    ACTION_TYPES,
    ITEM_KINDS,
    canonicalize_agent_id,
    canonicalize_claim_id,
    parse_read_offset,
    parse_reply,
)
from server_manager import LlamaServerManager, assert_single_llama_server, resolve_model
from society import Society, SOCIAL_ACTIONS
from communication import Communication

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
    max_tokens: int = 16384
    context_chars: int = 300000
    effective_context_tokens: int = 32768
    context_source: str = "config"
    reasoning: Optional[str] = None
    weight: float = 1.0
    priority: int = 0
    profile: str = "standard"
    capabilities: set = field(default_factory=set)
    subscriptions: list = field(default_factory=list)
    structured: bool = False
    note: str = ""
    style: str = ""
    pending_reads: list = field(default_factory=list)   # list of (fname, version, content)
    pending_doc_reads: list = field(default_factory=list)  # list of formatted doc excerpts / search results
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
    "temp_changes", "sampling_changes", "end_votes", "end_votes_attempted", "end_requested",
    "deliverable_verified", "deliverable_issues", "pending_validation_feedback",
    "speaker_transitions", "resurrection_events", "fallbacks_triggered",
    "solved_turn", "verified_synthesis_turn", "token_accounting_history", "inference_history",
    "doc_access_log",
)
SAVED_AGENT_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "typical_p", "xtc_probability", "xtc_threshold",
    "repeat_penalty", "presence_penalty", "frequency_penalty", "dry_multiplier", "dynatemp_range",
    "note", "style", "pending_reads", "pending_doc_reads", "active", "finished_reason",
    "turns_taken", "consecutive_turns", "last_turn", "failures", "empty_count",
    "total_prompt_tokens", "total_completion_tokens", "subscriptions",
)


def preview_artifact(name: str, content: str, limit: int = 8000) -> str:
    if len(content) <= limit:
        return content
    marker = f"\n[Middle omitted; use @READ {name.removesuffix('.md')} OFFSET to inspect it.]\n"
    prefix = min(2000, (limit - len(marker)) // 2)
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
        reopen: bool = False,
        override_topic: Optional[str] = None,
        override_mode_text: Optional[str] = None,
        override_min_turns_before_end: Optional[int] = None,
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
        self.prompt_layout = str(self.cfg.get("prompt_layout") or "auto").strip().lower()
        if self.prompt_layout not in ("auto", "single", "kv_messages"):
            raise ValueError(
                f"unknown prompt_layout {self.prompt_layout!r}; expected 'auto', 'single', or 'kv_messages'"
            )
        self.watch_terms = [t.lower() for t in (self.cfg.get("watch_terms") or [])]
        self.answer_key = self.cfg.get("answer_key")

        # Workspace & Artifact manager
        ts = time.strftime("%Y%m%d_%H%M%S")
        preset_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(self.cfg.get("name") or "experiment"))
        self.run_id = str(self.cfg.get("run_id") or f"{preset_name}_{ts}")
        self.run_dir = Path(run_dir or os.path.join(RUNS_DIR, self.run_id)).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        art_cfg_init = self.cfg.get("artifacts") or {}
        art_max_bytes = int(art_cfg_init.get("max_bytes", self.limits["max_artifact_bytes"]))
        art_max_total = int(art_cfg_init.get("max_workspace_bytes", self.limits["max_workspace_bytes"]))
        art_max_count = int(art_cfg_init.get("max_count", self.limits["max_artifacts"]))
        per_art_limits = art_cfg_init.get("per_artifact_max_bytes")
        self.artifacts = ArtifactManager(
            self.run_dir / "artifacts",
            max_count=art_max_count,
            max_bytes=art_max_bytes,
            max_total_bytes=art_max_total,
            per_artifact_max_bytes=per_art_limits,
            max_history_per_artifact=int(art_cfg_init.get("max_history_per_artifact", 24)),
        )
        self.visible_artifacts: List[str] = []
        self.deliverable_validation = art_cfg_init.get("validation") or self.cfg.get("deliverable_validation")
        self.documents = DocumentLibrary(
            manifest_path=self.cfg.get("doc_manifest"),
            inline_docs=self.cfg.get("documents"),
            base_dir=Path(__file__).resolve().parent,
        )
        self.doc_access_log: List[Dict[str, Any]] = []

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
        needs_local_info = isinstance(self.backend, HttpBackend) and any(
            a.get("endpoint", "local") in ("local", "local_small")
            for a in raw_agents
            if isinstance(a.get("endpoint", "local"), str)
        )
        local_url = urlsplit(str((endpoints_map.get("local") or {}).get("base_url") or "http://127.0.0.1:8080/v1"))
        local_host = local_url.hostname or "127.0.0.1"
        local_port = local_url.port or (443 if local_url.scheme == "https" else 8080)
        local_server_info = server_mgr.query_server_info(local_host, local_port, timeout=0.8) if needs_local_info else {}
        if local_server_info.get("online"):
            server_reported_local_ctx = local_server_info.get("context_window")

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
            elif (
                ep.get("provider") == "llamacpp"
                and not ep.get("externally_managed")
                and isinstance(self.backend, HttpBackend)
                and local_server_info.get("online")
                and local_server_info.get("model_name")
            ):
                ep["model"] = str(local_server_info["model_name"])

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
                m_meta = catalog.get_model(ep["model"])
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
                caps = {str(c).upper() for c in caps_spec}

            subs = []
            for ch in (merged.get("subscriptions") or []):
                cname = re.sub(r"[^a-z0-9_-]", "", str(ch).lower())[:24]
                if cname and cname not in subs and len(subs) < int(self.limits["max_subscriptions"]):
                    subs.append(cname)
                    self.channels.setdefault(cname, [])

            if ep_name == "dory" and "temperature" not in raw:
                merged["temperature"] = 1.0
            temp = max(float(self.limits["temp_min"]), min(float(self.limits["temp_max"]), float(merged["temperature"])))
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
        self.end_votes_attempted: bool = False
        self.end_requested: bool = False
        self.deliverable_verified: Optional[bool] = None
        self.deliverable_issues: List[str] = []
        self.pending_validation_feedback: List[str] = []
        self._turn_replaced_artifacts: Set[str] = set()
        self.speaker_transitions: List[Tuple[int, str, str]] = []
        self.resurrection_events: List[dict] = []
        self.fallbacks_triggered: List[dict] = []
        self.solved_turn: Optional[int] = None
        self.verified_synthesis_turn: Optional[int] = None
        self.start_memory_mb = get_available_memory_mb()
        self.token_accounting_history: List[Dict[str, Any]] = []
        self.inference_history: List[Dict[str, Any]] = []
        self.initial_sampling_config: Dict[str, Dict[str, Any]] = {
            n: {k: getattr(a, k) for k in SAMPLING_PARAMS if getattr(a, k, None) is not None}
            for n, a in self.agents.items()
        }

        if checkpoint_data is not None:
            self.resume_from_checkpoint(checkpoint_data)
            if reopen:
                self.end_requested = False
                self.end_votes = {}
                self.end_votes_attempted = False
                self.deliverable_verified = None
                self.deliverable_issues = []
                self.pending_validation_feedback = []
                for ag_state in self.agents.values():
                    if not ag_state.active and "deactivated" not in ag_state.finished_reason.lower():
                        ag_state.active = True
                        ag_state.finished_reason = ""
                self._record(
                    "SYSTEM", "CONTINUATION_REOPEN", "system",
                    detail=f"Reopened run at turn {self.turn} for continued deliberation",
                )
            if override_topic:
                self.topic = override_topic.strip()
                self.cfg["topic"] = self.topic
                self._record("SYSTEM", "TOPIC_UPDATED", "system", detail=self.topic[:200])
            if override_mode_text is not None:
                self.mode_text = override_mode_text.strip()
                self.cfg["mode_text"] = self.mode_text
            if override_min_turns_before_end is not None:
                self.min_turns_before_end = max(0, int(override_min_turns_before_end))
                self.cfg["min_turns_before_end"] = self.min_turns_before_end
            if (reopen or self.cfg.get("doc_manifest")) and self.documents.has_docs():
                self._record(
                    "SYSTEM", "DOC_MANIFEST_MOUNTED", "system",
                    detail=f"Mounted {len(self.documents.entries)} reference document(s): {', '.join(self.documents.ids())}",
                )

        if self.documents.has_docs():
            self.cfg["mounted_documents"] = self.documents.manifest_metadata()

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
            detail=str(detail)[:1000],
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
    def build_context(self, agent: AgentState) -> Tuple[List[Dict[str, str]], TokenAccounting]:
        others = [n for n in self.agent_order if n != agent.name]
        all_names = ", ".join(self.agent_order)
        other_names = ", ".join(others) if others else agent.name

        # Keep system sections agent-invariant when agents share profile/capabilities so backend KV/prompt cache hits.
        sys_sections = {
            "preamble": f"Open-ended multi-agent experiment; participants: {all_names}.",
            "mode": self.mode_text if self.mode_text else "",
            "boundary": "You operate inside a bounded chamber: no external tools, shell, files, or web. "
                        "Respond naturally to the conversation — express curiosity, disagreement, humor, or new ideas. "
                        "Write only your own response; never write lines for another participant.",
            "protocol": self._protocol_instructions(agent),
        }

        fixed_sections: Dict[str, str] = {
            "topic": f"Topic: {self.topic}",
        }
        if self.documents.has_docs():
            fixed_sections["doc_manifest"] = self.documents.render_manifest()

        id_lines = [f"--- Your Identity & Role ({agent.name}) ---", f"You are {agent.name}. Other participants: {other_names}."]
        if agent.identity.strip():
            id_lines.append(agent.identity.strip())
        if agent.focus.strip():
            id_lines.append(f"Your current angle or role: {agent.focus.strip()}.")
        if agent.style.strip():
            id_lines.append(f"Your chosen speaking style: {agent.style.strip()}.")

        tail_sections: Dict[str, str] = {
            "agent_identity": "\n".join(id_lines),
            "knowledge": f"--- Your Private Initial Knowledge (visible only to {agent.name}) ---\n{agent.knowledge}" if agent.knowledge else "",
            "scratchpad": f"--- Your Private Scratchpad (visible only to {agent.name}) ---\n{agent.note}" if agent.note else "",
        }
        if "communication" in self.cfg:
            tail_sections["communication"] = self.communication.describe(agent.name)

        min_fb_turn = max(0, int(agent.last_turn)) if agent.last_turn else 0
        recent_feedback = [
            e for e in self.events
            if e.actor == agent.name and e.turn >= min_fb_turn
            and (e.status == "rejected" or e.type in ("DELIVERABLE_WARNING", "DELIVERABLE_HINT"))
        ][-6:]
        feedback_lines = [
            (f"{e.type}" + (f" [{e.target}]" if e.target else "") + f": REJECTED ({e.detail})")
            if e.status == "rejected"
            else e.detail
            for e in recent_feedback
        ]
        if self.pending_validation_feedback:
            feedback_lines.append("END_RUN BLOCKED — Deliverable validation failed:")
            feedback_lines.extend(f"  - {issue}" for issue in self.pending_validation_feedback)

        if self.society:
            tail_sections["society"] = "--- Society ---\n" + self.society.describe(agent.name)
            my_events = [e for e in self.events if e.actor == agent.name and
                         e.type in SOCIAL_ACTIONS | {"PUBLISH", "ARTIFACT_WRITE"}][-8:]
            if my_events:
                tail_sections["society_results"] = "--- Results of your recent actions ---\n" + "\n".join(
                    f"{e.type} {e.target}: {e.status}" + (f" ({e.detail})" if e.detail else "") for e in my_events)
            if self.pending_validation_feedback:
                tail_sections["action_feedback"] = "--- Action & Deliverable Feedback ---\n" + "\n".join(feedback_lines)
        elif feedback_lines:
            tail_sections["action_feedback"] = "--- Action & Deliverable Feedback ---\n" + "\n".join(feedback_lines)

        # Artifacts
        art_blocks = []
        shown_files = set()
        read_limit = int(self.limits.get("artifact_read_chars", 8000))
        preview_limit = int(self.limits.get("artifact_preview_chars", 8000))
        for saved_read in agent.pending_reads:
            fname, ver, text = saved_read[:3]
            shown_files.add(fname)
            if len(saved_read) == 5:
                _, _, _, offset, total = saved_read
                label = f"[Artifact {fname} (v{ver}, requested read, chars {offset}-{offset + len(text)} of {total})]"
            else:
                label = f"[Artifact {fname} (v{ver}, requested read)]"
            art_blocks.append(f"{label}\n{text[:read_limit]}")
        for fname in self.visible_artifacts:
            if fname in self.artifacts.versions and fname not in shown_files:
                try:
                    _, ver, text = self.artifacts.read(agent.name, fname, turn=self.turn)
                    sz_b = len(text.encode("utf-8"))
                    trail = self.artifacts.recent_history_summary(fname, limit=4)
                    trail_line = f"Version history: {trail}\n" if trail else ""
                    art_blocks.append(f"[Shared Artifact {fname} (v{ver}, {sz_b}B)]\n{trail_line}{preview_artifact(fname, text, limit=preview_limit)}")
                except ArtifactError:
                    pass
        if "ARTIFACT_WRITE" in agent.capabilities or "ARTIFACT_READ" in agent.capabilities:
            listing = ", ".join(f"{f} (v{v}, {sz}B)" for f, v, sz in self.artifacts.listing()) or "(none yet)"
            art_blocks.append(f"Available workspace artifacts: {listing}")
        if art_blocks:
            tail_sections["artifacts"] = "--- Workspace Artifacts ---\n" + "\n\n".join(art_blocks)

        if self.documents.has_docs() and agent.pending_doc_reads:
            tail_sections["doc_excerpts"] = (
                "--- Requested Reference Document Excerpts ---\n"
                + "\n\n".join(agent.pending_doc_reads)
            )

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
            tail_sections["blackboard"] = "--- Shared Blackboard ---\n" + "\n".join(b_lines)

        # Subscribed channels
        if agent.subscriptions:
            ch_lines = []
            for cname in agent.subscriptions:
                if not self.communication.channel_allows(cname, agent.name, "readers"):
                    continue
                for m in self.channels.get(cname, [])[-6:]:
                    ch_lines.append(f"[#{cname} | {m.id} | turn {m.turn}] {m.speaker}: {m.content}")
            if ch_lines:
                tail_sections["channels"] = "--- Subscribed Channels ---\n" + "\n".join(ch_lines)

        # Whispers (plus whisper notices if enabled, kept in tail so Public Conversation prefix stays cache-stable)
        my_dms = [d for d in self.dms if d.sender == agent.name or d.recipient == agent.name][-8:]
        whisper_lines = [
            f"[PRIVATE WHISPER {d.id} | turn {d.turn} | {d.sender} → {d.recipient}]: {d.content}"
            for d in my_dms
        ]
        if self.dm_visibility == "notice":
            for d in self.dms[-6:]:
                if d.sender != agent.name and d.recipient != agent.name:
                    whisper_lines.append(f"[Notice: {d.sender} sent a private whisper to {d.recipient} on turn {d.turn}]")
        if whisper_lines:
            tail_sections["whispers"] = "--- Private Messages (visible only to sender & recipient) ---\n" + "\n".join(whisper_lines)

        if self.deliverable_validation or self.end_mode == "agent_decided" or self.documents.has_docs() or self.turn >= 2:
            status_parts = [f"Turn {self.turn} / {self.limits['max_turns']}"]
            if self.end_mode == "agent_decided":
                active_cnt = sum(1 for a in self.agents.values() if a.active)
                voted_cnt = sum(1 for n, a in self.agents.items() if a.active and n in self.end_votes)
                status_parts.append(f"End votes ({self.end_policy}): {voted_cnt}/{active_cnt} active")
            status_lines = [" | ".join(status_parts)]
            if self.deliverable_validation:
                deliv_ok, deliv_issues, deliv_det = validate_artifact(self.artifacts, self.deliverable_validation)
                dname = deliv_det.get("artifact", str(self.deliverable_validation.get("artifact", "")))
                dver = deliv_det.get("version")
                dsz = deliv_det.get("size_bytes", 0)
                meta_str = f"v{dver}, {dsz}B" if dver else "missing"
                trail = self.artifacts.recent_history_summary(dname, limit=5) if dver else ""
                if deliv_ok:
                    status_lines.append(f"Deliverable `{dname}`: READY ({meta_str}, all required sections present)")
                    if trail:
                        status_lines.append(f"  Versions: {trail}")
                else:
                    status_lines.append(f"Deliverable `{dname}`: INCOMPLETE ({meta_str})")
                    if trail:
                        status_lines.append(f"  Versions: {trail}")
                    for iss in deliv_issues:
                        status_lines.append(f"  - {iss}")
            if self.documents.has_docs():
                all_doc_ids = self.documents.ids()
                doc_audit = self._compute_doc_citation_audit()
                inspected_ids = set(doc_audit["docs_inspected_ids"])
                unread = [d for d in all_doc_ids if d not in inspected_ids]
                cov = (
                    f"Ref docs inspected ({len(inspected_ids)}/{len(all_doc_ids)}): unread = {', '.join(unread)}"
                    if unread
                    else f"Ref docs inspected ({len(all_doc_ids)}/{len(all_doc_ids)}): all mounted docs inspected"
                )
                recent_reads: List[str] = []
                for rec in reversed(self.doc_access_log):
                    if rec.get("ok") and rec.get("op") == "read":
                        tag = f"{rec.get('target')}@{rec.get('offset', 0)}"
                        if tag not in recent_reads:
                            recent_reads.append(tag)
                        if len(recent_reads) >= 6:
                            break
                if recent_reads:
                    recent_reads.reverse()
                    cov += f" | Recent reads: {', '.join(recent_reads)}"
                status_lines.append(cov)
                unverified = doc_audit["unverified_doc_citations"]
                if unverified:
                    status_lines.append(
                        "Unverified doc citations (cited without @DOC_READ/@DOC_TOC/@DOC_SEARCH): "
                        + ", ".join(unverified.keys())
                    )
            if self.board_visible:
                if self.board:
                    b_counts = {"active": 0, "disputed": 0, "retracted": 0}
                    total_ch = 0
                    for b_item in self.board.values():
                        st_k = b_item.status if b_item.status in b_counts else "active"
                        b_counts[st_k] += 1
                        total_ch += len(b_item.challenges)
                    status_lines.append(
                        f"Blackboard: {b_counts['active']} active, {b_counts['disputed']} disputed, "
                        f"{b_counts['retracted']} retracted ({total_ch} challenge(s))"
                    )
                else:
                    status_lines.append(
                        "Blackboard: 0 items (use @CLAIM / @CHALLENGE / @VOTE to record and stress-test key claims)"
                    )
            tail_sections["session_status"] = "--- Session Status ---\n" + "\n".join(status_lines)

        # Public messages (append-only shared prefix across all agents)
        pub_msgs = [{"id": m.id, "turn": m.turn, "speaker": m.speaker, "content": m.content, "reply_to": m.reply_to}
                    for m in self.public_messages if m.speaker == agent.name or
                    self.communication.allows("public", m.speaker, agent.name)]

        messages, accounting = self.ctx_intelligence.assemble_and_budget(
            agent_name=agent.name,
            endpoint=agent.endpoint if isinstance(self.backend, HttpBackend) else {**agent.endpoint, "provider": "mock"},
            system_sections=sys_sections,
            fixed_sections=fixed_sections,
            public_messages=pub_msgs,
            effective_context=agent.effective_context_tokens,
            context_source=agent.context_source,
            reserved_output=agent.max_tokens,
            safety_margin=int(agent.endpoint.get("safety_margin", self.safety_margin)),
            context_mode=self.context_mode,
            max_context_chars=int(self.limits["max_context_chars"]),
            agent_context_chars=int(agent.context_chars),
            tail_sections=tail_sections,
            prompt_layout=self.prompt_layout,
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
            if self.documents.has_docs() and ("DOC_READ" in caps or "DOC_SEARCH" in caps):
                lines.append("Reference library: @DOC_TOC <doc_id> | @DOC_SEARCH <doc_id|all> <query> | @DOC_READ <doc_id> 0 | @DOC_LIST")
            if self.end_mode == "agent_decided" and ("END_RUN" in caps or "FINISH" in caps):
                lines.append(
                    f"Run ending is agent-decided ({self.end_policy}, safety cap {self.limits['max_turns']} turns): "
                    f"@END_RUN: <reason> to vote to conclude (or @FINISH: <reason> to exit)."
                )
            return "\n".join(lines)

        supports_extra_samplers = agent.endpoint.get("provider") == "llamacpp" and not agent.endpoint.get("externally_managed")
        if agent.profile == "rich":
            allowed = sorted(caps - {"SAY"})
            extra = []
            if supports_extra_samplers and "SET_SAMPLING" in caps:
                extra.append('  "sampling_request": {"min_p": 0.05, "top_k": 40, "xtc_probability": 0.3, "xtc_threshold": 0.1},')
            if self.end_mode == "agent_decided" and ("END_RUN" in caps or "FINISH" in caps):
                extra.append(f'Run ending is agent-decided ({self.end_policy}, safety cap {self.limits["max_turns"]} turns): use END_RUN (or FINISH) action when ready to wrap up.')
            extra_json = ('\n' + extra[0]) if (supports_extra_samplers and "SET_SAMPLING" in caps) else ''
            extra_note = ('\n' + extra[-1]) if (self.end_mode == "agent_decided" and ("END_RUN" in caps or "FINISH" in caps)) else ''
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
                '    {"type": "ARTIFACT_WRITE", "name": "world", "mode": "create|append|replace|replace_section|append_section|revert", "section": "optional heading", "target_version": 2, "content": "markdown"},\n'
                '    {"type": "ARTIFACT_READ", "name": "world"}'
                + (',\n    {"type": "DOC_TOC", "doc_id": "..."},\n    {"type": "DOC_SEARCH", "target": "all", "query": "..."},\n    {"type": "DOC_READ", "doc_id": "...", "offset": 0}' if self.documents.has_docs() else '')
                + "\n  ]\n}\n"
                f"Permitted action types for you: {', '.join(allowed)}.{extra_note}"
            )

        cmds = [
            f"NEXT: <name> ({names}" + ("; omit or write NONE when ending)" if self.end_mode == "agent_decided" else ")"),
            f"TEMP: <{lo}..{hi}>",
        ]
        if supports_extra_samplers and "SET_SAMPLING" in caps:
            cmds.append("@SET_SAMPLING <key=val ...>  (optional — omit unless changing your sampling; keys: temp, top_p, top_k, min_p, xtc_prob, xtc_thresh, repeat_penalty, dry_multiplier, dynatemp)")
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
            cmds.append(
                "@APPEND <artifact>: <markdown to append>  |  @SECTION <artifact> | <heading>: <replace one section>  |  "
                "@APPEND_SECTION <artifact> | <heading>: <add to section>  |  @REPLACE <artifact>: <full markdown>  |  "
                "@REVERT <artifact> v<N>  (or @READ <artifact> 0)"
            )
        if self.documents.has_docs() and ("DOC_READ" in caps or "DOC_SEARCH" in caps or "DOC_TOC" in caps):
            cmds.append("@DOC_TOC <doc_id>  |  @DOC_SEARCH <doc_id|all> <query>  |  @DOC_READ <doc_id> 0 (use numeric integer offset)  |  @DOC_LIST")
        if self.end_mode == "agent_decided" and ("END_RUN" in caps or "FINISH" in caps):
            cmds.append(f"@END_RUN: <reason> (vote to conclude session; policy: {self.end_policy}, cap: {self.limits['max_turns']} turns) | @FINISH: <reason>")
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
        if max_spend is not None and endpoint.get("remote"):
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
        accounting: Optional[TokenAccounting] = None,
    ):
        budget_err = self._check_budgets_before_call(agent.endpoint)
        if budget_err:
            raise InferenceError(budget_err, transient=False)

        mtok = min(int(self.limits["max_tokens_per_response"]), int(max_tokens or agent.max_tokens))
        rem_tok = int(self.limits["max_total_tokens"]) - self.tracked_tokens
        mtok = max(16, min(mtok, rem_tok))

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

                    if cand_ep.get("remote"):
                        if res.cost is not None:
                            self.remote_cost_usd += float(res.cost)
                        else:
                            self.remote_calls_without_cost += 1

                    # Record per-inference telemetry (distinguishing configured, transmitted, and backend-confirmed values)
                    transmitted_sampling = {
                        k: payload[k] for k in SAMPLING_PARAMS if k in payload
                    }
                    comp_tok_s = (
                        round(ctok / res.latency, 2)
                        if res.latency and res.latency > 0 and ctok is not None
                        else None
                    )
                    ctx_budget_info = None
                    if accounting is not None:
                        ctx_budget_info = {
                            "total_prompt": accounting.total_prompt,
                            "available_prompt_budget": accounting.available_prompt_budget,
                            "reserved_output": accounting.reserved_output,
                            "safety_margin": accounting.safety_margin,
                            "remaining_context": accounting.remaining_context,
                            "omitted_messages_count": accounting.omitted_messages_count,
                            "fits_in_context": accounting.fits_in_context,
                            "counting_method": accounting.counting_method,
                        }
                    self.inference_history.append({
                        "turn": self.turn,
                        "agent": agent.name,
                        "endpoint": cand_name,
                        "requested_model": cand_ep.get("model", agent.model),
                        "actual_model": res.actual_model or cand_ep.get("model", agent.model),
                        "effective_context_window": agent.effective_context_tokens,
                        "context_source": agent.context_source,
                        "configured_sampling": copy.deepcopy(self.initial_sampling_config.get(agent.name, {})),
                        "transmitted_sampling": transmitted_sampling,
                        "max_tokens_requested": payload.get("max_tokens"),
                        "prompt_tokens": ptok,
                        "completion_tokens": ctok,
                        "reasoning_tokens": res.reasoning_tokens,
                        "visible_tokens": res.visible_tokens,
                        "token_accounting": res.token_accounting,
                        "cached_tokens": res.cached_tokens,
                        "cache_source": res.cache_source,
                        "latency_s": round(res.latency, 3) if res.latency else 0.0,
                        "completion_tok_per_s": comp_tok_s,
                        "finish_reason": res.finish_reason,
                        "truncated": res.finish_reason == "length",
                        "context_budget": ctx_budget_info,
                        "sampling_requests": [],
                        "rejected_actions": [],
                    })
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
        agent.pending_doc_reads.clear()

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

        # Structured / command actions: deduplicate identical/no-op actions first,
        # and ensure an explicit END_RUN/FINISH inside an over-budget response
        # is not starved by a flood of earlier votes/notes (still counts against max_act).
        deduped_actions = []
        seen_keys = set()
        for act in parsed.actions:
            atype = str(act.get("type") or "").upper()
            if atype == "VOTE":
                tgt = canonicalize_claim_id(act.get("target"))
                val = str(act.get("value") or "").strip().lower()
                if tgt in self.board and self.board[tgt].votes.get(agent.name) == val:
                    continue
            try:
                if atype == "DOC_READ":
                    key = json.dumps({"type": atype, "doc_id": str(act.get("doc_id") or "").lower(), "offset": int(act.get("offset", 0))})
                elif atype == "DOC_SEARCH":
                    key = json.dumps({"type": atype, "target": str(act.get("target") or "all").lower(), "query": " ".join(str(act.get("query") or "").lower().split())})
                elif atype == "DOC_TOC":
                    key = json.dumps({"type": atype, "doc_id": str(act.get("doc_id") or "").lower()})
                elif atype == "ARTIFACT_READ":
                    key = json.dumps({"type": atype, "name": str(act.get("name") or "").lower(), "offset": int(act.get("offset", 0))})
                else:
                    key = json.dumps(act, sort_keys=True)
            except TypeError:
                key = str(act)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_actions.append(act)

        max_act = int(self.limits["max_actions_per_response"])
        if max_act >= 1 and len(deduped_actions) > max_act:
            term_indices = [i for i, a in enumerate(deduped_actions)
                            if str(a.get("type") or "").upper() in ("END_RUN", "FINISH")]
            if term_indices and term_indices[0] >= max_act:
                term_act = deduped_actions.pop(term_indices[0])
                deduped_actions.insert(max_act - 1, term_act)

        self._turn_replaced_artifacts.clear()
        try:
            for idx, act in enumerate(deduped_actions):
                if idx >= max_act:
                    self._record(agent.name, act.get("type", "UNKNOWN"), "rejected",
                                 detail=f"exceeded max_actions_per_response ({max_act}); keep <= {max_act} actions per turn")
                    continue
                self._apply_action(agent, act)
        finally:
            self._turn_replaced_artifacts.clear()

    def _apply_action(self, agent: AgentState, act: dict):
        atype = str(act.get("type") or "").upper()
        if atype not in ACTION_TYPES:
            self._record(agent.name, atype or "UNKNOWN", "rejected", detail="unknown action type")
            return
        if atype not in agent.capabilities:
            self._record(agent.name, atype, "rejected", detail=f"capability {atype} not enabled for {agent.name}")
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
                req_t = float(act.get("value"))
            except (TypeError, ValueError):
                self._record(agent.name, atype, "rejected", detail="invalid temperature value")
                return
            old_t = agent.temperature
            new_t = max(float(lim["temp_min"]), min(float(lim["temp_max"]), req_t))
            if abs(new_t - old_t) < 1e-6:
                self._record(agent.name, atype, "accepted", detail=f"{old_t:.2f} -> {new_t:.2f} (unchanged)")
                return
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
            applied = {}
            changed_any = False
            for ck, cv in staged.items():
                old_v = getattr(agent, ck, None)
                if old_v != cv:
                    changed_any = True
                setattr(agent, ck, cv)
                applied[ck] = {"old": old_v, "new": cv}
                if ck == "temperature" and old_v != cv:
                    self.temp_changes.append((self.turn, agent.name, float(old_v), float(cv)))
            summary = ", ".join(f"{k}={v['new']}" for k, v in applied.items())
            if changed_any:
                self.sampling_changes.append((self.turn, agent.name, {k: v["new"] for k, v in applied.items()}))
                self._record(agent.name, atype, "accepted", detail=summary)
                self._emit(f"  ⚙ sampling {agent.name}: {summary}")
            else:
                self._record(agent.name, atype, "accepted", detail=f"{summary} (unchanged)")
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
                valid_chs = ", ".join(f"#{c}" for c in sorted(self.channels)) or "none"
                hint = ""
                if ch:
                    try:
                        art_cand = self.artifacts.normalize(ch)
                        if art_cand in self.artifacts.versions:
                            hint = f"; `{art_cand}` is an artifact — use @APPEND {ch}: ... (valid channels: {valid_chs})"
                    except ArtifactError:
                        pass
                if not hint:
                    hint = f" (valid channels: {valid_chs})"
                self._record(agent.name, atype, "rejected", target=ch, detail=f"unknown channel{hint}")
                return
            if self.society and ch in self.society.groups and agent.name not in self.society.groups[ch]["members"]:
                self._record(agent.name, atype, "rejected", target=ch, detail="group membership required")
                return
            content = str(act.get("content") or "").strip()[: int(lim["max_message_chars"])]
            if not content or sum(len(posts) for posts in self.channels.values()) >= int(lim["max_channel_posts"]):
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
                valid_chs = ", ".join(f"#{c}" for c in sorted(self.channels)) or "none"
                self._record(agent.name, atype, "rejected", target=ch, detail=f"unknown channel (valid channels: {valid_chs})")
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
            if re.fullmatch(r"c\d+", content, re.I) and content.lower() in self.board:
                self._record(
                    agent.name, atype, "rejected", target=content.lower(),
                    detail=f"CLAIM content cannot be existing claim ID {content.lower()!r}; use @VOTE or @CHALLENGE instead",
                )
                return
            m_prefix = re.match(r"^(c\d+)\s*:\s*(.+)$", content, re.I)
            if m_prefix:
                content = m_prefix.group(2).strip()
            norm_claim = " ".join(content.lower().split())
            for existing in self.board.values():
                if existing.status != "retracted" and " ".join(existing.content.lower().split()) == norm_claim:
                    self._record(
                        agent.name, atype, "rejected", target=existing.id,
                        detail=f"duplicate of active board item {existing.id}; use @VOTE {existing.id} yes|no instead",
                    )
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
                active_ids = ", ".join(list(self.board.keys())[-10:]) or "none"
                self._record(agent.name, atype, "rejected", target=tgt,
                             detail=f"target item {tgt!r} not on board (existing IDs: {active_ids})")
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
                active_ids = ", ".join(list(self.board.keys())[-10:]) or "none"
                self._record(agent.name, atype, "rejected", target=tgt,
                             detail=f"target item {tgt!r} not on board (existing IDs: {active_ids})")
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
                active_ids = ", ".join(list(self.board.keys())[-10:]) or "none"
                self._record(agent.name, atype, "rejected", target=tgt,
                             detail=f"target item {tgt!r} not on board (existing IDs: {active_ids})")
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
            new_top = str(act.get("content") or "").strip()[:2000]
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
            offset, off_err = parse_read_offset(act.get("offset", 0))
            if off_err:
                self._record(agent.name, atype, "rejected", target=name, detail=off_err)
                return
            try:
                fname, ver, text = self.artifacts.read(agent.name, name, turn=self.turn, mark_seen=False)
            except ArtifactError as e:
                detail_msg = str(e)
                if self.documents.has_docs():
                    try:
                        doc_cand = self.documents.normalize_id(name)
                        if doc_cand in self.documents.entries:
                            detail_msg = f"{detail_msg} — `{doc_cand}` is a reference document; use @DOC_READ {doc_cand} 0 instead"
                    except Exception:
                        pass
                self._record(agent.name, atype, "rejected", target=name, detail=detail_msg)
                return
            if offset >= len(text):
                self._record(agent.name, atype, "rejected", target=fname, detail=f"offset {offset} is outside v{ver} ({len(text)} chars)")
                return
            read_limit = int(lim.get("artifact_read_chars", 8000))
            excerpt = text[offset:offset + read_limit]
            self.artifacts.seen[(agent.name, fname)] = ver
            agent.pending_reads.append((fname, ver, excerpt, offset, len(text)))
            self._record(agent.name, atype, "accepted", target=fname,
                         detail=f"v{ver} chars {offset}-{offset + len(excerpt)} of {len(text)}")
            self._emit(f"  📖 read {fname} (v{ver}) queued for {agent.name}")
            return

        if atype == "DOC_LIST":
            if not self.documents.has_docs():
                self._record(agent.name, atype, "rejected", detail="no reference document manifest is mounted")
                return
            max_pend = int(lim.get("max_pending_doc_reads", 8))
            if len(agent.pending_doc_reads) < max_pend:
                agent.pending_doc_reads.append(self.documents.render_manifest(max_summary_chars=400))
            self.doc_access_log.append({"turn": self.turn, "agent": agent.name, "op": "list", "target": "all", "ok": True})
            self._record(agent.name, atype, "accepted", detail=f"{len(self.documents.entries)} doc(s)")
            return

        if atype == "DOC_TOC":
            if not self.documents.has_docs():
                self._record(agent.name, atype, "rejected", detail="no reference document manifest is mounted")
                return
            max_pend = int(lim.get("max_pending_doc_reads", 8))
            if len(agent.pending_doc_reads) >= max_pend:
                self._record(agent.name, atype, "rejected", detail=f"max_pending_doc_reads ({max_pend}) reached")
                return
            raw_doc = act.get("doc_id") or act.get("name") or act.get("target") or ""
            try:
                doc_id, _title, toc_entries, formatted = self.documents.table_of_contents(raw_doc)
            except DocumentError as exc:
                self._record(agent.name, atype, "rejected", target=str(raw_doc), detail=str(exc))
                return
            agent.pending_doc_reads.append(formatted)
            self.doc_access_log.append({
                "turn": self.turn, "agent": agent.name, "op": "toc",
                "target": doc_id, "entries": len(toc_entries), "ok": True,
            })
            self._record(agent.name, atype, "accepted", target=doc_id, detail=f"{len(toc_entries)} heading(s)")
            self._emit(f"  📑 doc toc `{doc_id}` ({len(toc_entries)} headings) queued for {agent.name}")
            return

        if atype == "DOC_READ":
            if not self.documents.has_docs():
                self._record(agent.name, atype, "rejected", detail="no reference document manifest is mounted")
                return
            max_pend = int(lim.get("max_pending_doc_reads", 8))
            if len(agent.pending_doc_reads) >= max_pend:
                self._record(agent.name, atype, "rejected", detail=f"max_pending_doc_reads ({max_pend}) reached")
                return
            raw_doc = act.get("doc_id") or act.get("name") or act.get("target") or ""
            offset, off_err = parse_read_offset(act.get("offset", 0))
            if off_err:
                self._record(agent.name, atype, "rejected", target=str(raw_doc), detail=off_err)
                return
            read_limit = int(lim.get("doc_read_chars", 12000))
            try:
                doc_id, title, excerpt, used_off, total = self.documents.read(raw_doc, offset=offset, max_chars=read_limit)
            except DocumentError as exc:
                self._record(agent.name, atype, "rejected", target=str(raw_doc), detail=str(exc))
                return
            block = f"[Ref Doc `{doc_id}` — {title} | chars {used_off}-{used_off + len(excerpt)} of {total}]\n{excerpt}"
            agent.pending_doc_reads.append(block)
            self.doc_access_log.append({
                "turn": self.turn, "agent": agent.name, "op": "read",
                "target": doc_id, "offset": used_off, "chars": len(excerpt), "ok": True,
            })
            self._record(agent.name, atype, "accepted", target=doc_id,
                         detail=f"chars {used_off}-{used_off + len(excerpt)} of {total}")
            self._emit(f"  📚 doc read `{doc_id}` ({used_off}..{used_off + len(excerpt)}/{total}) queued for {agent.name}")
            return

        if atype == "DOC_SEARCH":
            if not self.documents.has_docs():
                self._record(agent.name, atype, "rejected", detail="no reference document manifest is mounted")
                return
            max_pend = int(lim.get("max_pending_doc_reads", 8))
            if len(agent.pending_doc_reads) >= max_pend:
                self._record(agent.name, atype, "rejected", detail=f"max_pending_doc_reads ({max_pend}) reached")
                return
            tgt = act.get("target") or act.get("doc_id") or "all"
            query = str(act.get("query") or act.get("content") or "").strip()
            try:
                scope, hits, formatted = self.documents.search(tgt, query)
            except DocumentError as exc:
                self._record(agent.name, atype, "rejected", target=str(tgt), detail=str(exc))
                return
            agent.pending_doc_reads.append(formatted)
            self.doc_access_log.append({
                "turn": self.turn, "agent": agent.name, "op": "search",
                "target": scope, "query": query, "hits": len(hits), "ok": True,
            })
            self._record(agent.name, atype, "accepted", target=scope,
                         detail=f"query={query[:80]!r} ({len(hits)} hit(s))")
            self._emit(f"  🔎 doc search ({scope}) {query[:40]!r} -> {len(hits)} hit(s) for {agent.name}")
            return

        if atype == "ARTIFACT_WRITE":
            name = str(act.get("name") or act.get("target") or "")
            mode = str(act.get("mode") or "append").lower()
            content = str(act.get("content") or "")
            exp_ver = act.get("expected_version")
            section = act.get("section")
            force = bool(act.get("force", False))
            try:
                norm_name = self.artifacts.normalize(name)
            except ArtifactError:
                norm_name = ""
            existed_before = bool(norm_name and norm_name in self.artifacts.versions)

            if mode == "revert":
                try:
                    fname, ver, restored_from, size_bytes, delta_bytes = self.artifacts.revert(
                        agent.name, name, act.get("target_version"), turn=self.turn,
                    )
                except ArtifactError as e:
                    self._record(agent.name, atype, "rejected", target=name, detail=str(e))
                    return
                mode_label = f"revert(v{restored_from})"
                self._record(
                    agent.name, atype, "accepted", target=fname,
                    detail=f"{mode_label} -> v{ver} ({size_bytes}B, {delta_bytes:+d}B)",
                )
                self._emit(f"  ⏪ artifact {fname} (reverted to v{restored_from} by {agent.name} → v{ver}, {size_bytes}B, {delta_bytes:+d}B)")
            else:
                if mode == "replace" and not force and norm_name and norm_name in self._turn_replaced_artifacts:
                    self._record(
                        agent.name, atype, "rejected", target=norm_name,
                        detail=f"duplicate @REPLACE on {norm_name} in the same turn ignored to protect your earlier write; combine edits or use @SECTION",
                    )
                    return
                try:
                    fname, ver = self.artifacts.write(
                        agent.name, name, mode, content, expected_version=exp_ver, turn=self.turn,
                        section=section, force=force, min_replace_ratio=float(lim.get("min_replace_ratio", 0.25)),
                    )
                except ArtifactError as e:
                    self._record(agent.name, atype, "rejected", target=name, detail=str(e))
                    return
                if mode == "replace":
                    self._turn_replaced_artifacts.add(fname)
                try:
                    size_bytes = os.lstat(self.artifacts._path(fname)).st_size
                except OSError:
                    size_bytes = len(content.encode("utf-8"))
                hist = self.artifacts.history.get(fname) or []
                delta_bytes = hist[-1]["delta_bytes"] if hist else size_bytes
                self._check_answer_key(content)
                mode_label = f"{mode}[{str(section or '')[:40]}]" if mode in ("replace_section", "append_section") else mode
                sz_label = f"{size_bytes}B, {delta_bytes:+d}B" if ver > 1 else f"{size_bytes}B"
                self._record(agent.name, atype, "accepted", target=fname, detail=f"{mode_label} -> v{ver} ({sz_label})")
                self._emit(f"  📝 artifact {fname} ({mode_label} by {agent.name} → v{ver}, {sz_label})")

            if self.deliverable_validation:
                raw_deliv = str(self.deliverable_validation.get("artifact") or "")
                try:
                    deliv_fname = self.artifacts.normalize(raw_deliv) if raw_deliv else ""
                except ArtifactError:
                    deliv_fname = ""
                if self.pending_validation_feedback or (deliv_fname and fname == deliv_fname):
                    ok, issues, _ = validate_artifact(self.artifacts, self.deliverable_validation)
                    self.deliverable_verified = ok
                    self.deliverable_issues = issues
                    if self.pending_validation_feedback:
                        self.pending_validation_feedback = [] if ok else issues
                if mode != "revert" and not existed_before and fname not in self.visible_artifacts and deliv_fname and fname != deliv_fname:
                    short_new = fname.removesuffix(".md")
                    short_deliv = deliv_fname.removesuffix(".md")
                    if short_new in short_deliv or short_deliv in short_new:
                        self._record(
                            agent.name, "DELIVERABLE_HINT", "system", target=fname,
                            detail=f"Created separate artifact {fname} ({size_bytes}B), not the required deliverable {deliv_fname}.",
                        )
            return

        if atype == "FINISH":
            reason = str(act.get("content") or "voluntarily finished").strip()[:120]
            agent.active = False
            agent.finished_reason = reason
            self._record(agent.name, atype, "accepted", detail=reason)
            self._emit(f"  🚪 {agent.name} finished participating: {reason}")
            if self.end_mode == "agent_decided":
                self._evaluate_agent_end()
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
            already_voted = agent.name in self.end_votes
            self.end_votes[agent.name] = reason
            if already_voted:
                self._record(agent.name, atype, "accepted", detail=f"{reason} (vote reaffirmed)")
                self._emit(f"  🏁 {agent.name} reaffirmed vote to end run: {reason}")
            else:
                self._record(agent.name, atype, "accepted", detail=reason)
                self._emit(f"  🏁 {agent.name} voted to end run: {reason}")
            self._evaluate_agent_end()
            if not self.end_requested and self.deliverable_validation and not self.pending_validation_feedback:
                d_ok, d_issues, d_det = validate_artifact(self.artifacts, self.deliverable_validation)
                if not d_ok:
                    d_name = d_det.get("artifact", str(self.deliverable_validation.get("artifact", "")))
                    self._record(
                        agent.name, "DELIVERABLE_WARNING", "system", target=d_name,
                        detail=f"Your @END_RUN vote is recorded, but deliverable {d_name} is still INCOMPLETE: {'; '.join(d_issues)}",
                    )
            return

    def _evaluate_agent_end(self):
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
        if not reached:
            return

        self.end_votes_attempted = True
        if self.deliverable_validation:
            ok, issues, details = validate_artifact(self.artifacts, self.deliverable_validation)
            self.deliverable_verified = ok
            self.deliverable_issues = issues
            if not ok:
                self.pending_validation_feedback = list(issues)
                art_name = details.get("artifact", str(self.deliverable_validation.get("artifact", "")))
                msg = f"Deliverable validation failed for {art_name}: {'; '.join(issues)}"
                self._record("SYSTEM", "END_RUN", "rejected", target=art_name, detail=msg)
                self._emit(f"  ⚠ END_RUN blocked — {msg}")
                return
            self.pending_validation_feedback = []
            art_name = details.get("artifact", "")
            self._record(
                "SYSTEM", "DELIVERABLE_VERIFIED", "system", target=art_name,
                detail=f"{art_name} v{details.get('version')} ({details.get('size_bytes')}B) verified",
            )

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
            self.nominated_next = None
            self._record("SYSTEM", "SCHEDULER", "system", target=chosen, detail="anti-starvation override")
            return chosen

        max_consec = int(self.limits["max_consecutive"])
        eligible = [n for n in active if self.agents[n].consecutive_turns < max_consec] or active

        def fair_fallback(pool=None):
            cand = list(eligible if pool is None else pool)
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
            max_dyad = self.limits.get("max_dyad_turns")
            dyad_set: Optional[Set[str]] = None
            non_dyad_eligible: List[str] = []
            if max_dyad is not None and int(max_dyad) >= 2 and len(active) >= 3:
                dyad_win = int(max_dyad)
                if len(self.public_messages) >= dyad_win:
                    recent_set = {m.speaker for m in self.public_messages[-dyad_win:]}
                    if len(recent_set) <= 2:
                        cands = [n for n in eligible if n not in recent_set]
                        if cands:
                            dyad_set = recent_set
                            non_dyad_eligible = cands

            if dyad_set is not None:
                if self.nominated_next and self.nominated_next in non_dyad_eligible:
                    chosen = self.nominated_next
                    self._record("SCHEDULER", "SPEAKER_NOMINATION", "accepted", target=chosen)
                else:
                    rq_non_dyad = [x for x in self.reactive_queue if x in non_dyad_eligible]
                    if rq_non_dyad:
                        chosen = rq_non_dyad[0]
                        self.reactive_queue.remove(chosen)
                    else:
                        chosen = fair_fallback(non_dyad_eligible)
                    self._record(
                        "SYSTEM", "SCHEDULER", "system", target=chosen,
                        detail=f"dyad lock-in breaker ({'/'.join(sorted(dyad_set))})",
                    )
            else:
                hub_cfg = self.limits.get("max_speaker_window_turns")
                hub_nominee = None
                hub_count = 0
                hub_win = 0
                non_hub_eligible: List[str] = []
                if hub_cfg and len(active) >= 4 and self.nominated_next and self.nominated_next in eligible:
                    if isinstance(hub_cfg, (list, tuple)) and len(hub_cfg) >= 2:
                        hub_win, hub_max = int(hub_cfg[0]), int(hub_cfg[1])
                    elif isinstance(hub_cfg, dict):
                        hub_win = int(hub_cfg.get("window", max(6, len(active))))
                        hub_max = int(hub_cfg.get("max_turns", 3))
                    else:
                        hub_win = max(6, len(active))
                        hub_max = int(hub_cfg)
                    if hub_win >= 2 and hub_max >= 1 and len(self.public_messages) >= hub_win:
                        recent_slice = self.public_messages[-hub_win:]
                        hub_count = sum(1 for m in recent_slice if m.speaker == self.nominated_next)
                        if hub_count >= hub_max:
                            cands = [
                                n for n in eligible
                                if n != self.nominated_next
                                and sum(1 for m in recent_slice if m.speaker == n) < hub_max
                            ] or [n for n in eligible if n != self.nominated_next]
                            if cands:
                                hub_nominee = self.nominated_next
                                non_hub_eligible = cands
                if hub_nominee is not None:
                    rq_non_hub = [x for x in self.reactive_queue if x in non_hub_eligible]
                    if rq_non_hub:
                        chosen = rq_non_hub[0]
                        self.reactive_queue.remove(chosen)
                    else:
                        chosen = fair_fallback(non_hub_eligible)
                    self._record(
                        "SYSTEM", "SCHEDULER", "system", target=chosen,
                        detail=f"hub lock-in breaker ({hub_nominee} had {hub_count}/{hub_win} recent turns)",
                    )
                elif self.nominated_next and self.nominated_next in eligible:
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
            for attribute in SAVED_AGENT_FIELDS:
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
        speaker = self.next_speaker or self._select_next_speaker(None)
        self.next_speaker = speaker
        max_turns = int(self.limits["max_turns"])
        failed = False

        try:
            self._save_turn_checkpoint(self.turn)
            for t in range(start_turn, max_turns + 1):
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
                    if ag.endpoint.get("enforce_context_budget") and (
                        not accounting.fits_in_context or
                        accounting.total_prompt + accounting.reserved_output + accounting.safety_margin > accounting.effective_usable_context
                    ):
                        raise InferenceError("prompt plus output and safety margin exceeds endpoint context cap", transient=False)
                    res = self._call_model(ag, ctx, stops=stops, accounting=accounting)
                except InferenceError as e:
                    ag.failures += 1
                    self._record(speaker, "ERROR", "rejected", detail=str(e))
                    self._emit(f"  ⚠ [{speaker}] inference error: {e}")
                    if ag.failures >= int(self.limits["max_failures_per_agent"]):
                        ag.active = False
                        ag.finished_reason = f"deactivated after {ag.failures} endpoint failures"
                    speaker = self._select_next_speaker(speaker)
                    self.next_speaker = speaker
                    self._save_turn_checkpoint(t)
                    continue

                parsed = parse_reply(
                    res.content, res.reasoning, me=speaker, names=tuple(self.agent_order),
                    max_chars=int(self.limits["max_message_chars"]),
                )

                if not parsed.message and not parsed.actions:
                    # Only promote reasoning text if generation was NOT cut off mid-thought;
                    # if it hit length inside reasoning, retry with reasoning="off" first.
                    if res.finish_reason != "length" and res.reasoning:
                        paras = [p.strip() for p in res.reasoning.split("\n\n") if p.strip()]
                        if paras:
                            parsed = parse_reply(paras[-1], res.reasoning, me=speaker, names=tuple(self.agent_order),
                                                 max_chars=int(self.limits["max_message_chars"]))
                    if not parsed.message and not parsed.actions and self._check_budgets_before_call(ag.endpoint) is None:
                        try:
                            res2 = self._call_model(ag, ctx, stops=stops, reasoning="off", accounting=accounting)
                            parsed = parse_reply(res2.content, res2.reasoning or res.reasoning, me=speaker,
                                                 names=tuple(self.agent_order), max_chars=int(self.limits["max_message_chars"]))
                            if parsed.message or parsed.actions:
                                res = res2
                        except InferenceError:
                            pass

                if res.finish_reason == "length":
                    parsed.truncated = True
                    self._record(speaker, "TRUNCATED", "system", detail="finish_reason=length")

                self.format_counts[parsed.format] = self.format_counts.get(parsed.format, 0) + 1
                if not parsed.message and not parsed.actions:
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
                if self.inference_history and self.inference_history[-1]["turn"] == t and self.inference_history[-1]["agent"] == speaker:
                    turn_evs = [e for e in self.events if e.turn == t and e.actor == speaker]
                    self.inference_history[-1]["sampling_requests"] = [
                        {"type": e.type, "status": e.status, "detail": e.detail}
                        for e in turn_evs
                        if e.type in ("SET_SAMPLING", "SET_TEMPERATURE") or (e.type == "PROTOCOL" and "SAMPLING" in e.detail.upper())
                    ]
                    self.inference_history[-1]["rejected_actions"] = [
                        {"type": e.type, "target": e.target, "detail": e.detail}
                        for e in turn_evs
                        if e.status == "rejected"
                    ]

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
            if self.deliverable_validation and self.deliverable_verified is None:
                ok, issues, _ = validate_artifact(self.artifacts, self.deliverable_validation)
                self.deliverable_verified = ok
                self.deliverable_issues = issues
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

    def _compute_doc_citation_audit(self) -> Dict[str, Any]:
        """Compare bracketed document citations in speech/artifacts against actual DOC_* tool calls."""
        if not self.documents.has_docs():
            return {"docs_inspected_count": 0, "docs_inspected_ids": [], "unverified_doc_citations": {}}
        all_ids = self.documents.ids()
        accessed_ids = {
            rec.get("target")
            for rec in self.doc_access_log
            if rec.get("ok") and rec.get("op") in ("read", "toc", "search") and rec.get("target") in all_ids
        }
        inspected_ids = [
            d for d in all_ids
            if any(
                rec.get("ok") and rec.get("op") in ("read", "toc") and rec.get("target") == d
                for rec in self.doc_access_log
            )
        ]
        corpus_chunks = [m.content for m in self.public_messages]
        for fname in self.artifacts.names():
            try:
                corpus_chunks.append(self.artifacts._read_file(self.artifacts._path(fname)))
            except Exception:
                pass
        joined = "\n".join(corpus_chunks)
        bracket_blocks = re.findall(r"\[[^\]\n]{1,160}\]", joined)
        unverified: Dict[str, int] = {}
        if bracket_blocks:
            for doc_id in all_ids:
                if doc_id in accessed_ids:
                    continue
                pat = re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(doc_id)}(?![A-Za-z0-9_-])")
                cnt = sum(1 for blk in bracket_blocks if pat.search(blk))
                if cnt > 0:
                    unverified[doc_id] = cnt
        return {
            "docs_inspected_count": len(inspected_ids),
            "docs_inspected_ids": inspected_ids,
            "unverified_doc_citations": unverified,
        }

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
        comp_tok_per_s = round(self.completion_tokens / self.elapsed_s, 2) if self.elapsed_s > 0 else 0.0
        tok_rates = [r["completion_tok_per_s"] for r in self.inference_history if r.get("completion_tok_per_s") is not None]
        mean_comp_tps = round(sum(tok_rates) / len(tok_rates), 2) if tok_rates else None
        cache_recs = [r for r in self.inference_history if r.get("cached_tokens") is not None and r.get("prompt_tokens")]
        total_cached = sum(int(r["cached_tokens"]) for r in cache_recs)
        total_cache_prompt = sum(int(r["prompt_tokens"]) for r in cache_recs)
        cache_hit_rate = round(total_cached / total_cache_prompt, 3) if total_cache_prompt > 0 else None
        doc_searches = [d for d in self.doc_access_log if d.get("op") == "search"]
        doc_hits = sum(1 for d in doc_searches if int(d.get("hits") or 0) > 0)
        doc_search_hit_rate = round(doc_hits / len(doc_searches), 3) if doc_searches else None
        doc_audit = self._compute_doc_citation_audit()

        return dict(
            turns_completed=len(self.public_messages),
            requests_used=self.requests_used,
            retries_used=self.retries_used,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens_total,
            visible_tokens=self.visible_tokens_total,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            tokens_per_second=tok_per_s,
            completion_tokens_per_second=comp_tok_per_s,
            mean_completion_tok_per_s=mean_comp_tps,
            prompt_cache_hit_rate=cache_hit_rate,
            prompt_layout=self.prompt_layout,
            remote_cost_usd=round(self.remote_cost_usd, 6),
            remote_calls_without_cost=self.remote_calls_without_cost,
            elapsed_s=round(self.elapsed_s, 2),
            end_mode=self.end_mode,
            end_policy=self.end_policy,
            end_votes=dict(self.end_votes),
            agents_voted_to_stop=bool(self.end_votes),
            end_votes_attempted=self.end_votes_attempted,
            end_requested_by_agents=self.end_requested,
            run_ended=self.stop_reason != "not started",
            deliverable_required=bool(self.deliverable_validation),
            deliverable_verified=self.deliverable_verified,
            deliverable_issues=list(self.deliverable_issues),
            sampling_changes=list(self.sampling_changes),
            inference_history=list(self.inference_history),
            documents_mounted=len(self.documents.entries),
            doc_toc_count=sum(1 for d in self.doc_access_log if d.get("op") == "toc"),
            doc_reads_count=sum(1 for d in self.doc_access_log if d.get("op") == "read"),
            doc_searches_count=len(doc_searches),
            doc_search_hits=doc_hits,
            doc_search_hit_rate=doc_search_hit_rate,
            docs_inspected_count=doc_audit["docs_inspected_count"],
            docs_inspected_ids=doc_audit["docs_inspected_ids"],
            unverified_doc_citations=doc_audit["unverified_doc_citations"],
            doc_access_log=list(self.doc_access_log),
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
        cache_hit_line = (
            f"{metrics['prompt_cache_hit_rate']*100:.1f}%"
            if metrics.get("prompt_cache_hit_rate") is not None
            else "N/A"
        )
        mean_gen_tps = metrics.get("mean_completion_tok_per_s")
        gen_tps_str = f"{mean_gen_tps} tok/s" if mean_gen_tps is not None else "N/A"
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
            f"- **Throughput (total / mean gen)**: {metrics['tokens_per_second']} tok/s / {gen_tps_str}",
            f"- **Prompt cache hit rate**: {cache_hit_line}",
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

        if self.deliverable_validation:
            ver_label = "VERIFIED" if metrics["deliverable_verified"] else "FAILED"
            m_lines.extend([
                "",
                "### Deliverable Verification",
                f"- **Agents voted to stop**: {metrics['agents_voted_to_stop']} (`end_requested_by_agents={metrics['end_requested_by_agents']}`)",
                f"- **Deliverable verified**: **{ver_label}**",
            ])
            if metrics["deliverable_issues"]:
                for iss in metrics["deliverable_issues"]:
                    m_lines.append(f"  - Issue: {iss}")

        if self.sampling_changes:
            m_lines.extend(["", "### Sampling Parameter Changes"])
            for turn_no, ag_name, changed in self.sampling_changes:
                m_lines.append(f"- Turn {turn_no} (`{ag_name}`): {', '.join(f'{k}={v}' for k, v in changed.items())}")

        if self.inference_history:
            m_lines.extend([
                "",
                "### Per-Inference Sampling & Context Telemetry",
                "| Turn | Agent | Model | Ctx | Prompt/Comp | Lat(s) | Tok/s | Stop | Transmitted Sampling |",
                "|---:|---|---|---:|---|---:|---:|---|---|",
            ])
            for rec in self.inference_history:
                samp_str = ", ".join(f"{k}={v}" for k, v in (rec.get("transmitted_sampling") or {}).items())
                cache_note = f" (cached={rec['cached_tokens']})" if rec.get("cached_tokens") is not None else ""
                m_lines.append(
                    f"| {rec['turn']} | `{rec['agent']}` | `{rec['actual_model']}` | {rec['effective_context_window']//1024}K | "
                    f"{rec['prompt_tokens']}/{rec['completion_tokens']}{cache_note} | {rec['latency_s']} | "
                    f"{rec['completion_tok_per_s'] if rec['completion_tok_per_s'] is not None else 'N/A'} | "
                    f"{rec['finish_reason']} | `{samp_str}` |"
                )

        if self.documents.has_docs():
            all_doc_ids = self.documents.ids()
            inspected_docs = list(metrics.get("docs_inspected_ids") or [])
            unverified_cites = metrics.get("unverified_doc_citations") or {}
            hit_str = (
                f" ({metrics['doc_search_hits']}/{metrics['doc_searches_count']} hit, "
                f"{metrics['doc_search_hit_rate'] * 100:.0f}%)"
                if metrics.get("doc_search_hit_rate") is not None
                else ""
            )
            m_lines.extend([
                "",
                "### Reference Document Access",
                f"- **Documents mounted**: {metrics['documents_mounted']} ({', '.join(all_doc_ids)})",
                f"- **Documents inspected (TOC/read)**: {len(inspected_docs)}/{len(all_doc_ids)}"
                + (f" ({', '.join(inspected_docs)})" if inspected_docs else ""),
                f"- **Doc TOCs / reads / searches**: {metrics['doc_toc_count']} / {metrics['doc_reads_count']} / {metrics['doc_searches_count']}{hit_str}",
                f"- **Unverified bracket citations (docs cited without DOC_* access)**: "
                + (", ".join(f"`{k}` ({v}x)" for k, v in unverified_cites.items()) if unverified_cites else "none"),
            ])
            for rec in self.doc_access_log:
                m_lines.append(f"  - Turn {rec.get('turn')} (`{rec.get('agent')}`): `{rec.get('op')}` `{rec.get('target')}`")

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
    ap.add_argument("--topic-file", default=None, help="Read topic override from a text file")
    ap.add_argument("--mode-text", default=None, help="Override mode_text instructions")
    ap.add_argument("--mode-text-file", default=None, help="Read mode_text override from a text file")
    ap.add_argument("--doc-manifest", default=None, help="Path to JSON reference document manifest")
    ap.add_argument("--validation-file", default=None, help="Path to JSON deliverable validation config")
    ap.add_argument("--min-turns-before-end", type=int, default=None, help="Override min_turns_before_end")
    ap.add_argument("--max-duration", type=float, default=None, help="Override max_duration_s")
    ap.add_argument("--max-total-tokens", type=int, default=None, help="Override max_total_tokens")
    ap.add_argument("--reopen", action="store_true", help="When used with --resume, reopen an ended run for another phase")
    ap.add_argument("--model", "-m", default=None, help="Override local model alias or path")
    ap.add_argument("--agent-model", action="append", default=[], help="Per-agent model override: AGENT=model_id")
    ap.add_argument("--agent-endpoint", action="append", default=[], help="Per-agent endpoint override: AGENT=dory (or local, hermes, etc.)")
    ap.add_argument("--agent-sampling", action="append", default=[], help="Per-agent sampling overrides: AGENT=min_p=0.05,top_k=40,xtc_probability=0.3")
    ap.add_argument("--agent-instruction", "--agent-identity", dest="agent_instruction", action="append", default=[], help="Per-agent instructions: AGENT=instruction_text")
    ap.add_argument("--agent-focus", action="append", default=[], help="Per-agent angle/role: AGENT=focus_text")
    ap.add_argument("--config", default=None, help="Path to custom JSON or YAML experiment configuration file")
    ap.add_argument("--context-tokens", "-c", default=None, help="Usable context token budget ('auto', 8192, 16384, 32768, etc.)")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="Max output tokens per response")
    ap.add_argument("--reasoning", default=None, help="Reasoning mode ('on', 'off', 'low', 'medium', 'high')")
    ap.add_argument("--temperature", type=float, default=None, help="Temperature override")
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
    if args.topic_file:
        tpath = Path(args.topic_file)
        if not tpath.exists():
            print(f"Error: Topic file not found: {tpath}", file=sys.stderr)
            return 1
        args.topic = tpath.read_text(encoding="utf-8").strip()
    if args.mode_text_file:
        mpath = Path(args.mode_text_file)
        if not mpath.exists():
            print(f"Error: Mode-text file not found: {mpath}", file=sys.stderr)
            return 1
        args.mode_text = mpath.read_text(encoding="utf-8").strip()
    validation_override = None
    if args.validation_file:
        vpath = Path(args.validation_file)
        if not vpath.exists():
            print(f"Error: Validation file not found: {vpath}", file=sys.stderr)
            return 1
        try:
            validation_override = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Error reading validation file {vpath}: {exc}", file=sys.stderr)
            return 1

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

    # Model resolution
    endpoints_map = {**copy.deepcopy(ENDPOINTS), **copy.deepcopy(preset_cfg.get("endpoints") or {})}
    if args.model:
        mode, mfile, default_ctx = resolve_model(args.model)
        endpoints_map["local"]["model"] = mfile

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

    # Handle per-agent sampling overrides (--agent-sampling AGENT=min_p=0.05,top_k=40)
    for override in args.agent_sampling:
        if "=" not in override:
            print(f"Error: --agent-sampling expects AGENT=key=val[,...], got {override!r}", file=sys.stderr)
            return 1
        raw_name, kv_spec = override.split("=", 1)
        ag_name = canonicalize_agent_id(raw_name)
        matches = [a for a in preset_cfg.get("agents", []) if canonicalize_agent_id(a.get("name")) == ag_name]
        if len(matches) != 1:
            print(f"Error: unknown agent in --agent-sampling {override!r}", file=sys.stderr)
            return 1
        for pair in re.split(r"[,;\s]+", kv_spec.strip()):
            if not pair or "=" not in pair:
                continue
            rk, rv = pair.split("=", 1)
            try:
                ck, cv = normalize_sampling_param(rk, rv)
            except ValueError as exc:
                print(f"Error in --agent-sampling {override!r}: {exc}", file=sys.stderr)
                return 1
            matches[0][ck] = cv

    # Diagnostics execution
    display_preset = str(preset_cfg.get("name") or args.preset)
    if args.doctor:
        doc = run_doctor(endpoints_map, Path(RUNS_DIR))
        if args.json:
            print(json.dumps(doc, indent=2))
        else:
            print(format_doctor_report(doc))
        return 0 if doc["doctor_passed"] else 1

    if args.context_info:
        info = run_context_info(
            display_preset,
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
            "preset": display_preset,
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
            display_preset,
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
        loc_url = urlsplit(str(endpoints_map.get("local", {}).get("base_url") or "http://127.0.0.1:8080/v1"))
        loc_host = loc_url.hostname or "127.0.0.1"
        loc_port = loc_url.port or (443 if loc_url.scheme == "https" else 8080)
        if needs_local and not args.mock:
            loc_model = endpoints_map["local"]["model"]
            # If an external llama-server is already running and no model was explicitly requested, adopt it.
            if not args.model and not os.environ.get("LOCAL_MODEL") and not server_mgr.is_server_owned(loc_port):
                if server_mgr.check_health(loc_host, loc_port, timeout=1.5):
                    running_info = server_mgr.query_server_info(loc_host, loc_port, timeout=1.5)
                    if running_info.get("model_name"):
                        loc_model = str(running_info["model_name"])
                        endpoints_map["local"]["model"] = loc_model
            mode, mfile, def_ctx = resolve_model(loc_model)
            res = server_mgr.ensure_server(
                host=loc_host,
                port=loc_port,
                model_mode=mode,
                model_file=mfile,
                allow_switch=args.allow_server_switch,
                ctx_size=int(args.context_tokens) if (args.context_tokens and args.context_tokens != "auto") else None,
            )
            if res.get("action") == "started":
                started_server = True

        overrides: Dict[str, Any] = {"limits": {}}
        if args.agent_end:
            overrides["end_mode"] = "agent_decided"
        if args.end_mode:
            overrides["end_mode"] = args.end_mode
        if args.end_policy:
            overrides["end_policy"] = args.end_policy
        if args.turns is not None:
            overrides["limits"]["max_turns"] = args.turns
        elif overrides.get("end_mode") == "agent_decided":
            cur_max = int((preset_cfg.get("limits") or {}).get("max_turns", LIMITS["max_turns"]))
            if cur_max < 40:
                overrides["limits"]["max_turns"] = 40
            cur_req = int((preset_cfg.get("limits") or {}).get("max_requests", LIMITS["max_requests"]))
            if cur_req < 80:
                overrides["limits"]["max_requests"] = 80
        if args.requests is not None:
            overrides["limits"]["max_requests"] = args.requests
        if args.max_duration is not None:
            overrides["limits"]["max_duration_s"] = args.max_duration
        if args.max_total_tokens is not None:
            overrides["limits"]["max_total_tokens"] = args.max_total_tokens
        if args.scheduler:
            overrides["scheduler"] = args.scheduler
        if args.topic:
            overrides["topic"] = args.topic
        if args.mode_text is not None:
            overrides["mode_text"] = args.mode_text
        if args.min_turns_before_end is not None:
            overrides["min_turns_before_end"] = args.min_turns_before_end
        if args.doc_manifest:
            overrides["doc_manifest"] = str(Path(args.doc_manifest).resolve())
        if validation_override is not None:
            overrides["deliverable_validation"] = validation_override
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
                reopen=bool(args.reopen),
                override_topic=args.topic if resume_data is not None else None,
                override_mode_text=args.mode_text if resume_data is not None else None,
                override_min_turns_before_end=args.min_turns_before_end if resume_data is not None else None,
            )
        except (OSError, ValueError, KeyError, TypeError, ArtifactError) as error:
            if resume_data is None:
                raise
            print(f"Error restoring saved run: {error}. The source run has not been modified.", file=sys.stderr)
            return 1

        chamber.run()
    finally:
        if args.stop and started_server:
            server_mgr.stop_owned_server(loc_port)
        wakelock.release()
        lock.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
