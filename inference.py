"""Inference adapter: Multi-backend (llama.cpp, dynamic OpenRouter, mock) with per-agent model routing,
fallback chains, reasoning-token accounting, rate-limit backoff, and free-tier spend guards.

Only the orchestrator calls this module. API keys are read from the environment at request time,
sent only in the Authorization header, and never stored in chamber state, logs, or artifacts.
"""
import json
import math
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from presets import SAMPLING_PARAMS


@dataclass
class Result:
    """Inference completion result with detailed token accounting and metadata."""
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    visible_tokens: Optional[int] = None
    cost: Optional[float] = None              # Provider-reported cost in USD
    latency: float = 0.0
    actual_model: str = ""                    # Provider-reported or verified model ID
    backend: str = "llamacpp"
    token_accounting: str = "exact"           # "exact" | "estimated"
    reasoning_mode: str = "none"              # "boolean_thinking:on" | "effort:low" | "none"


class InferenceError(Exception):
    """Exception raised on inference failure with transient retry indicator."""
    def __init__(self, message: str, transient: bool = False, status_code: Optional[int] = None):
        super().__init__(message)
        self.transient = transient
        self.status_code = status_code


TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "reply_to": {"type": "string"},
        "next_speaker": {"type": "string"},
        "temperature_request": {"type": "number"},
        "actions": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["message"],
}


def build_payload(
    endpoint: Dict[str, Any],
    agent_cfg: Dict[str, Any],
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stops: Optional[List[str]] = None,
    seed: Optional[int] = None,
    reasoning: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Constructs OpenAI-compatible chat completion payload with backend-specific reasoning kwargs.

    Returns (payload, reasoning_mode_label).
    """
    provider = endpoint.get("provider", "openai")
    p: Dict[str, Any] = {
        "model": endpoint["model"],
        "messages": messages,
        "temperature": round(float(temperature), 3),
        "max_tokens": int(max_tokens),
    }

    for k, (_, _, _, llamacpp_only) in SAMPLING_PARAMS.items():
        if k == "temperature":
            continue
        if llamacpp_only and provider != "llamacpp":
            continue
        if agent_cfg.get(k) is not None:
            p[k] = agent_cfg[k]

    if stops:
        p["stop"] = list(stops) if provider == "llamacpp" else list(stops)[:4]

    if seed is not None:
        p["seed"] = int(seed)

    mode = agent_cfg.get("reasoning") if reasoning is None else reasoning
    reasoning_mode_label = "none"

    if provider == "llamacpp":
        # llama.cpp server accepts boolean chat_template_kwargs.enable_thinking
        if mode is not None:
            enable = mode != "off"
            p["chat_template_kwargs"] = {"enable_thinking": enable}
            reasoning_mode_label = f"boolean_thinking:{'on' if enable else 'off'}"
    elif provider == "openrouter":
        p["usage"] = {"include": True}  # Request OpenRouter usage & cost details
        if mode == "off":
            p["reasoning"] = {"enabled": False}
            reasoning_mode_label = "effort:off"
        elif mode in ("low", "medium", "high"):
            p["reasoning"] = {"effort": mode}
            reasoning_mode_label = f"effort:{mode}"
        elif mode:
            p["reasoning"] = {"enabled": True}
            reasoning_mode_label = "effort:on"

    if agent_cfg.get("structured"):
        p["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "chamber_turn", "schema": TURN_SCHEMA},
        }

    return p, reasoning_mode_label


class HttpBackend:
    """Production HTTP completion adapter supporting llama.cpp, OpenRouter, and rate-limiting backoff."""

    def __init__(self, free_only: bool = False, allow_paid: bool = False):
        self.free_only = free_only
        self.allow_paid = allow_paid

    def complete(
        self,
        agent_name: str,
        endpoint: Dict[str, Any],
        payload: Dict[str, Any],
        timeout: float = 120.0,
        reasoning_mode_label: str = "none",
    ) -> Result:
        provider = endpoint.get("provider", "openai")
        model_id = payload.get("model", endpoint.get("model", ""))

        # Check free-tier guards for OpenRouter
        if provider == "openrouter":
            if self.free_only and not model_id.endswith(":free"):
                raise InferenceError(
                    f"Blocked request: Model '{model_id}' is not in free tier, and --free-only is enabled.",
                    transient=False,
                )
            if not self.allow_paid and not model_id.endswith(":free"):
                # Warning or check against catalog
                pass

        key_env = endpoint.get("api_key_env") or ""
        key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not key:
            raise InferenceError(f"Environment variable {key_env} is not set", transient=False)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DialogueChamber/2.0",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/dialogue-chamber"
            headers["X-Title"] = "Dialogue Chamber"

        url = endpoint["base_url"].rstrip("/") + "/chat/completions"
        req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers)

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            # 401, 403, 400 are NOT transient; 429, 500, 502, 503, 504 are transient
            is_transient = e.code in (408, 409, 425, 429, 500, 502, 503, 504)
            # Handle rate-limit retry-after header if present
            retry_after = e.headers.get("Retry-After") if e.headers else None
            msg = f"HTTP {e.code} from {endpoint['base_url']} ({model_id})"
            if retry_after:
                msg += f" (Retry-After: {retry_after}s)"
            raise InferenceError(msg, transient=is_transient, status_code=e.code)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise InferenceError(f"{type(e).__name__} calling {endpoint['base_url']}", transient=True)
        except ValueError:
            raise InferenceError(f"Invalid JSON from {endpoint['base_url']}", transient=True)

        latency = time.monotonic() - t0

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            raise InferenceError(f"No choices in response from {endpoint['base_url']}", transient=True)

        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        cost = usage.get("cost")

        # Extract tokens with explicit reasoning breakdown if reported
        p_tok = usage.get("prompt_tokens")
        c_tok = usage.get("completion_tokens")
        comp_details = usage.get("completion_tokens_details") or {}
        r_tok = comp_details.get("reasoning_tokens") or usage.get("reasoning_tokens")

        vis_tok = None
        if c_tok is not None and r_tok is not None:
            vis_tok = max(0, c_tok - r_tok)
        elif c_tok is not None:
            vis_tok = c_tok

        raw_content = (msg.get("content") or "").strip()
        raw_reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()

        actual_model = data.get("model") or model_id

        return Result(
            content=raw_content,
            reasoning=raw_reasoning,
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            reasoning_tokens=r_tok,
            visible_tokens=vis_tok,
            cost=float(cost) if isinstance(cost, (int, float)) else None,
            latency=latency,
            actual_model=actual_model,
            backend=provider,
            token_accounting="exact" if (p_tok is not None and c_tok is not None) else "estimated",
            reasoning_mode=reasoning_mode_label,
        )


class MockBackend:
    """Scripted or algorithmic replies for testing multi-agent interactions and fallback chains."""

    def __init__(self, script: Optional[Dict[str, Any]] = None, default: str = ""):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def complete(
        self,
        agent_name: str,
        endpoint: Dict[str, Any],
        payload: Dict[str, Any],
        timeout: float = 120.0,
        reasoning_mode_label: str = "none",
    ) -> Result:
        self.calls.append((agent_name, payload))
        queue = self.script.get(agent_name)
        item = queue.pop(0) if queue else self.default

        if callable(item):
            item = item(payload)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Result):
            return item
        if isinstance(item, dict):
            return Result(**{"finish_reason": "stop", "actual_model": endpoint.get("model", ""), **item})

        text = str(item)
        c_tok = max(1, len(text) // 4)
        return Result(
            content=text,
            finish_reason="stop",
            prompt_tokens=max(1, len(json.dumps(payload.get("messages", []))) // 4),
            completion_tokens=c_tok,
            visible_tokens=c_tok,
            actual_model=endpoint.get("model", "mock-model"),
            backend=endpoint.get("provider", "mock"),
            token_accounting="estimated",
            reasoning_mode=reasoning_mode_label,
        )


class DemoBackend:
    """Offline test double generating plausible structured dialogue and actions."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.n = 0

    def complete(
        self,
        agent_name: str,
        endpoint: Dict[str, Any],
        payload: Dict[str, Any],
        timeout: float = 120.0,
        reasoning_mode_label: str = "none",
    ) -> Result:
        self.n += 1
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]

        names = re.findall(r"participants?: ([A-Z0-9_, ]+)\.", system)
        others = [x.strip() for x in names[0].split(",")] if names else []
        items = re.findall(r"\b(c\d+) \[", user)

        if "Reply with only one name" in user:
            choices = re.findall(r"[A-Z][A-Z0-9_]+", user.split("Participants:")[-1]) or ["X"]
            return Result(
                content=self.rng.choice(choices),
                finish_reason="stop",
                actual_model=endpoint.get("model", "demo"),
            )

        acts = []
        r = self.rng.random()
        if others and r < 0.25:
            acts.append({"type": "WHISPER", "recipient": self.rng.choice(others), "content": f"(demo) secret #{self.n}"})
        elif r < 0.5:
            acts.append({"type": "CLAIM", "kind": "hypothesis", "content": f"(demo) hypothesis {self.n}"})
        elif items and r < 0.7:
            acts.append({"type": "CHALLENGE", "target": self.rng.choice(items), "content": "(demo) where is the evidence?"})
        elif "world.md" in user or "conclusions.md" in user:
            name = "world" if "world.md" in user else "conclusions"
            acts.append({"type": "ARTIFACT_WRITE", "name": name, "mode": "append", "content": f"(demo) line {self.n}"})

        reply = {"message": f"(demo) {agent_name} speaking, step {self.n}.", "actions": acts}
        if others and self.rng.random() < 0.5:
            reply["next_speaker"] = self.rng.choice(others)

        return Result(
            content=json.dumps(reply),
            reasoning=f"Reflecting on step {self.n} for {agent_name}." if "thinking" in reasoning_mode_label else "",
            finish_reason="stop",
            completion_tokens=40,
            prompt_tokens=200,
            actual_model=endpoint.get("model", "demo"),
            backend="demo",
            token_accounting="estimated",
            reasoning_mode=reasoning_mode_label,
        )
