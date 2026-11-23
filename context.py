"""Tokenizer-aware context intelligence, token budgeting, and negotiation.

Calculates exact or conservative estimated token usage, negotiates server context limits,
allocates budgets across system instructions, topic, scratchpads, blackboards, whispers,
and artifacts, and handles context pressure gracefully without silent instruction loss.
Preserves character-budget constraints for backwards compatibility.
"""
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TokenAccounting:
    """Detailed token budget and context breakdown for an inference request."""
    system_instructions: int = 0
    agent_personality_rules: int = 0
    current_topic: int = 0
    private_notes: int = 0
    public_transcript: int = 0
    blackboard: int = 0
    whispers: int = 0
    shared_artifacts: int = 0
    tool_protocol_instructions: int = 0
    reference_documents: int = 0
    other_fixed: int = 0
    reserved_output: int = 4096
    safety_margin: int = 1024
    total_prompt: int = 0
    effective_usable_context: int = 8192
    available_prompt_budget: int = 3072
    remaining_context: int = 0
    counting_method: str = "estimated"       # "exact" | "estimated"
    context_source: str = "config"           # "server-reported" | "metadata" | "config" | "manual" | "unknown"
    confidence: str = "conservative"         # "exact" | "high" | "conservative" | "unverified"
    model_id: str = ""
    endpoint_url: str = ""
    omitted_messages_count: int = 0
    fits_in_context: bool = True
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenizerEngine:
    """Exact endpoint tokenization with safe conservative estimation fallback."""

    def __init__(self, timeout_s: float = 3.0):
        self.timeout_s = timeout_s
        self._cache: Dict[Tuple[str, str], int] = {}

    def count_tokens(self, text: str, endpoint: Optional[Dict[str, Any]] = None) -> Tuple[int, str]:
        """Counts tokens using backend endpoint if available, else conservative estimation.

        Returns (token_count, "exact" | "estimated").
        Never labels approximate token counts as exact.
        """
        if not text:
            return 0, "exact"

        ep = endpoint or {}
        provider = ep.get("provider", "")
        base_url = (ep.get("base_url") or "").rstrip("/")

        # 1. Try llama.cpp /tokenize endpoint if local
        if provider == "llamacpp" and base_url:
            tok_url = f"{base_url}/tokenize"
            if base_url.endswith("/v1"):
                root_tok_url = base_url[:-3] + "/tokenize"
            else:
                root_tok_url = tok_url

            for url in (tok_url, root_tok_url):
                cache_key = (url, text)
                if cache_key in self._cache:
                    return self._cache[cache_key], "exact"

                try:
                    payload = json.dumps({"content": text}).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": "DialogueChamber/2.0"},
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            tokens = data.get("tokens")
                            if isinstance(tokens, list):
                                count = len(tokens)
                                self._cache[cache_key] = count
                                return count, "exact"
                except Exception:
                    continue

        # 2. Conservative estimation fallback with explicit label
        return self.estimate_tokens_conservative(text), "estimated"

    @staticmethod
    def estimate_tokens_conservative(text: str) -> int:
        """Conservative token count estimation.

        Uses ~3.4 chars/token heuristic plus boundary overhead to ensure prompts never
        overflow physical model limits due to undercounting.
        """
        if not text:
            return 0
        char_len = len(text)
        special_chars = len(re.findall(r'[{}\[\]":,`\n\t_\\-]', text))
        base_tokens = math.ceil(char_len / 3.4)
        overhead = math.ceil(special_chars * 0.15)
        return max(1, base_tokens + overhead)


class ContextIntelligence:
    """Manages context budget allocation, negotiation, and pressure trimming."""

    def __init__(
        self,
        default_safety_margin: int = 512,
        min_transcript_tokens: int = 128,
        tokenizer: Optional[TokenizerEngine] = None,
    ):
        self.default_safety_margin = default_safety_margin
        self.min_transcript_tokens = min_transcript_tokens
        self.tokenizer = tokenizer or TokenizerEngine()

    def negotiate_context_limit(
        self,
        server_reported_ctx: Optional[int],
        user_configured_ctx: Optional[Any],
        model_default_ctx: int = 32768,
        hard_limit: int = 1048576,
    ) -> Tuple[int, str, str]:
        """Negotiates usable context limit.

        Returns (effective_limit, context_source, confidence).
        Source: 'server-reported', 'config', 'manual', 'unknown'.
        """
        if user_configured_ctx is not None and str(user_configured_ctx).lower() != "auto":
            try:
                user_val = int(user_configured_ctx)
                if server_reported_ctx and user_val > server_reported_ctx:
                    return server_reported_ctx, "server-reported", "high"
                return min(user_val, hard_limit), "config", "conservative"
            except (ValueError, TypeError):
                pass

        if server_reported_ctx and server_reported_ctx > 0:
            return min(server_reported_ctx, hard_limit), "server-reported", "exact"

        if model_default_ctx and model_default_ctx > 0:
            return min(model_default_ctx, hard_limit), "manual", "conservative"

        return 8192, "unknown", "unverified"

    def _classify_section(self, accounting: TokenAccounting, sec_name: str, cnt: int) -> None:
        low = sec_name.lower()
        if "protocol" in low:
            accounting.tool_protocol_instructions += cnt
        elif "preamble" in low or "mode" in low or "boundary" in low or "system" in low:
            accounting.system_instructions += cnt
        elif "identity" in low or "role" in low or "style" in low:
            accounting.agent_personality_rules += cnt
        elif "topic" in low:
            accounting.current_topic += cnt
        elif "scratchpad" in low or "note" in low:
            accounting.private_notes += cnt
        elif "blackboard" in low or "claim" in low:
            accounting.blackboard += cnt
        elif "whisper" in low or "private" in low or "knowledge" in low:
            accounting.whispers += cnt
        elif "artifact" in low or "workspace" in low:
            accounting.shared_artifacts += cnt
        elif "doc" in low or "manifest" in low:
            accounting.reference_documents += cnt
        else:
            accounting.other_fixed += cnt

    def assemble_and_budget(
        self,
        agent_name: str,
        endpoint: Dict[str, Any],
        system_sections: Dict[str, str],
        fixed_sections: Dict[str, str],
        public_messages: List[Dict[str, Any]],
        effective_context: int,
        context_source: str,
        reserved_output: int = 4096,
        safety_margin: Optional[int] = None,
        context_mode: str = "tokens",
        max_context_chars: int = 500000,
        agent_context_chars: int = 300000,
        tail_sections: Optional[Dict[str, str]] = None,
        prompt_layout: str = "single",
    ) -> Tuple[List[Dict[str, str]], TokenAccounting]:
        """Assembles prompt messages while enforcing tokenizer-aware context budget.

        Protects system instructions, fixed prefix sections (topic, doc manifest), and
        per-agent tail sections (identity, scratchpad, blackboard, whispers, artifacts),
        preferentially omitting older public dialogue under context pressure while
        preserving a stable shared prefix for backend KV/prompt caching.

        When ``prompt_layout`` is ``"kv_messages"`` (or ``"auto"`` on a ``llamacpp``
        endpoint), public turns are emitted as separate user messages before a final
        user message containing the mutable per-agent tail, allowing SWA checkpoint
        backends to cache the growing shared conversation prefix across turns.
        """
        margin = safety_margin if safety_margin is not None else self.default_safety_margin
        layout = (prompt_layout or "single").strip().lower()
        use_kv_messages = (layout == "kv_messages") or (
            layout == "auto" and endpoint.get("provider") == "llamacpp"
        )

        # Adjust reserved output if context window is tight
        actual_reserved = min(reserved_output, max(128, effective_context // 2))

        accounting = TokenAccounting(
            reserved_output=actual_reserved,
            safety_margin=margin,
            effective_usable_context=effective_context,
            context_source=context_source,
            model_id=endpoint.get("model", ""),
            endpoint_url=endpoint.get("base_url", ""),
        )

        # Tokenize system sections
        all_exact = True
        sys_tokens = 0
        for sec_name, text in system_sections.items():
            if not text.strip():
                continue
            cnt, method = self.tokenizer.count_tokens(text, endpoint)
            if method != "exact":
                all_exact = False
            sys_tokens += cnt
            self._classify_section(accounting, sec_name, cnt)

        system_content = "\n\n".join(t.strip() for t in system_sections.values() if t.strip())

        # Tokenize fixed prefix sections
        fixed_tokens = 0
        fixed_blocks = []
        for sec_name, text in fixed_sections.items():
            if not text.strip():
                continue
            cnt, method = self.tokenizer.count_tokens(text, endpoint)
            if method != "exact":
                all_exact = False
            fixed_tokens += cnt
            fixed_blocks.append(text)
            self._classify_section(accounting, sec_name, cnt)

        fixed_prompt_text = "\n\n".join(fixed_blocks)

        # Tokenize optional per-agent / mutable tail sections (kept after public transcript for KV-cache sharing)
        tail_tokens = 0
        tail_blocks = []
        for sec_name, text in (tail_sections or {}).items():
            if not text or not text.strip():
                continue
            cnt, method = self.tokenizer.count_tokens(text, endpoint)
            if method != "exact":
                all_exact = False
            tail_tokens += cnt
            tail_blocks.append(text)
            self._classify_section(accounting, sec_name, cnt)

        tail_prompt_text = "\n\n".join(tail_blocks)

        # Calculate prompt headroom available for public transcript
        available_prompt_budget = max(0, effective_context - (actual_reserved + margin))
        accounting.available_prompt_budget = available_prompt_budget

        essential_tokens = sys_tokens + fixed_tokens + tail_tokens + 40
        transcript_token_budget = max(0, available_prompt_budget - essential_tokens)

        # Backwards-compatible character budget
        char_budget = min(int(agent_context_chars), int(max_context_chars))
        pub_char_budget = max(200, char_budget - len(fixed_prompt_text) - len(tail_prompt_text) - 160)

        # Format public messages
        formatted_msgs = []
        for m in public_messages:
            rep = f" (reply to {m.get('reply_to')})" if m.get("reply_to") else ""
            line = f"[{m.get('id', 'm')} | turn {m.get('turn', 0)}]{rep} {m.get('speaker', '')}: {m.get('content', '')}"
            formatted_msgs.append(line)

        kept_lines: List[str] = []
        omitted = 0
        used_chars = 0
        used_tokens = 0

        # Traverse from newest to oldest
        msg_overhead = 4 if use_kv_messages else 0
        for line in reversed(formatted_msgs):
            line_tok, line_method = self.tokenizer.count_tokens(line, endpoint)
            if line_method != "exact":
                all_exact = False
            effective_line_tok = line_tok + msg_overhead

            # Check both char limit and token limit
            char_overflow = (used_chars + len(line) + 2 > pub_char_budget)
            token_overflow = (used_tokens + effective_line_tok > transcript_token_budget)

            if (char_overflow or token_overflow) and kept_lines:
                omitted += 1
            else:
                kept_lines.append(line)
                used_chars += len(line) + 2
                used_tokens += effective_line_tok

        accounting.omitted_messages_count = omitted
        kept_lines.reverse()

        notice = [f"[{omitted} earlier public message(s) omitted due to context limit]"] if omitted > 0 else []
        if kept_lines:
            transcript_block = "\n\n".join(notice + kept_lines)
        else:
            transcript_block = "(No public messages yet — you are opening the conversation.)"

        tr_tokens, tr_method = self.tokenizer.count_tokens(transcript_block, endpoint)
        if tr_method != "exact":
            all_exact = False
        accounting.public_transcript = tr_tokens

        framing_overhead = 0
        if use_kv_messages:
            messages = [{"role": "system", "content": system_content}]
            if fixed_prompt_text:
                messages.append({"role": "user", "content": fixed_prompt_text})
            if kept_lines:
                first_body = "--- Public Conversation ---\n" + (f"{notice[0]}\n\n" if notice else "") + kept_lines[0]
                messages.append({"role": "user", "content": first_body})
                for ln in kept_lines[1:]:
                    messages.append({"role": "user", "content": ln})
                final_parts = ([tail_prompt_text] if tail_prompt_text else []) + [f"Your turn ({agent_name}):"]
                messages.append({"role": "user", "content": "\n\n".join(final_parts)})
            else:
                # On turn 1 (no public messages yet), keep empty notice inside the mutable final user
                # message so Turn 2's prefix before [m1 | turn 1] stays identical.
                final_parts = ["--- Public Conversation ---\n(No public messages yet — you are opening the conversation.)"]
                if tail_prompt_text:
                    final_parts.append(tail_prompt_text)
                final_parts.append(f"Your turn ({agent_name}):")
                messages.append({"role": "user", "content": "\n\n".join(final_parts)})
            framing_overhead = 4 * max(0, len(messages) - 2)
        else:
            user_parts = []
            if fixed_prompt_text:
                user_parts.append(fixed_prompt_text)
            user_parts.append(f"--- Public Conversation ---\n{transcript_block}")
            if tail_prompt_text:
                user_parts.append(tail_prompt_text)
            user_parts.append(f"Your turn ({agent_name}):")
            user_content = "\n\n".join(user_parts)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

        total_prompt_tokens = sys_tokens + fixed_tokens + tail_tokens + tr_tokens + framing_overhead
        accounting.total_prompt = total_prompt_tokens
        accounting.remaining_context = max(0, effective_context - (total_prompt_tokens + actual_reserved + margin))
        accounting.counting_method = "exact" if all_exact else "estimated"

        if (total_prompt_tokens + actual_reserved) > effective_context:
            accounting.fits_in_context = False
            accounting.diagnostics.append(
                f"Oversized prompt: Total prompt ({total_prompt_tokens} tokens) + "
                f"output ({actual_reserved} tokens) exceeds context window ({effective_context} tokens)."
            )

        return messages, accounting
