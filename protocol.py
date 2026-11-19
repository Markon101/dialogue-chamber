"""Parse a model reply into public speech plus *proposed* actions, degrading gracefully.

Accepted formats:
  1. JSON: {"message", "reply_to", "next_speaker", "temperature_request", "actions": [...]}
  2. Unambiguously repaired JSON (trailing commas, unclosed brackets from truncation)
  3. Broken or truncated JSON: whatever fields can be salvaged safely
  4. Plain text with command lines: NEXT: / TEMP: / @WHISPER NAME: ... / @CLAIM kind: ...

Nothing here executes anything. The orchestrator validates every proposed action.
"""
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from presets import SAMPLING_ALIASES, normalize_sampling_param

ACTION_TYPES = (
    "SAY", "WHISPER", "PUBLISH", "SUBSCRIBE", "CLAIM", "CHALLENGE", "RETRACT", "VOTE",
    "PROPOSE_NEXT", "SET_TEMPERATURE", "SET_SAMPLING", "SET_STYLE", "PROPOSE_TOPIC", "ARTIFACT_WRITE",
    "ARTIFACT_READ", "ARTIFACT_LIST", "UPDATE_NOTE", "FINISH", "END_RUN",
    "CREATE_GROUP", "INVITE_GROUP", "JOIN_GROUP", "LEAVE_GROUP", "COMPEL_GROUP", "RELEASE_GROUP"
)

ALIASES = {
    "DM": "WHISPER", "PM": "WHISPER", "POST": "PUBLISH", "WRITE": "ARTIFACT_WRITE",
    "APPEND": "ARTIFACT_WRITE", "REPLACE": "ARTIFACT_WRITE", "CREATE": "ARTIFACT_WRITE", "WORLD": "ARTIFACT_WRITE",
    "READ": "ARTIFACT_READ", "LIST": "ARTIFACT_LIST", "NOTE": "UPDATE_NOTE", "NEXT": "PROPOSE_NEXT",
    "WITHDRAW": "RETRACT", "OBJECT": "CHALLENGE", "TOPIC": "PROPOSE_TOPIC", "STYLE": "SET_STYLE",
    "TEMP": "SET_TEMPERATURE", "TEMPERATURE": "SET_TEMPERATURE",
    "SAMPLING": "SET_SAMPLING", "SET_PARAMS": "SET_SAMPLING",
    "SAMPLER": "SET_SAMPLING", "SET_SAMPLER": "SET_SAMPLING", "SET_SAMPLERS": "SET_SAMPLING",
    "PROPOSE_END": "END_RUN", "END": "END_RUN", "DONE": "END_RUN",
}

SAMPLING_SHORTHANDS = {
    "XTC": ("xtc_probability", "xtc_threshold"),
    "MIN_P": ("min_p",),
    "TOP_P": ("top_p",),
    "TOP_K": ("top_k",),
    "REPEAT_PENALTY": ("repeat_penalty",),
    "TYPICAL_P": ("typical_p",),
}
for shorthand in SAMPLING_SHORTHANDS:
    ALIASES[shorthand] = ALIASES[f"SET_{shorthand}"] = "SET_SAMPLING"

ITEM_KINDS = (
    "hypothesis", "claim", "question", "objection", "decision", "proposal", "task", "fact", "topic"
)
JSON_KEYS = {"message", "actions", "next_speaker", "temperature_request", "sampling_request"}

_SAMPLING_CTRL_KEYS = "|".join(sorted((k.upper() for k in SAMPLING_ALIASES if k not in ("temp", "temperature")), key=len, reverse=True))
CMD_RE = re.compile(r"^\s*[@/]([A-Za-z_]+)\b[ \t]*(.*)$")
CTRL_RE = re.compile(rf"^\W*(NEXT(?:_SPEAKER)?|TEMP(?:ERATURE)?|{_SAMPLING_CTRL_KEYS})\W*:\s*(.*)$", re.I)
TAIL_NEXT_RE = re.compile(r"\bNEXT\s*:\s*([A-Za-z][\w]{0,15})\W*$", re.I)
LABEL_RE = re.compile(r"^\s*[*_]*\[?([A-Za-z][A-Za-z0-9_]{0,15})\]?[*_]*\s*:[*_]*\s*")
TRANSCRIPT_TAG_RE = re.compile(r"^\s*[*_]*(?:[@\[]\s*(?:m\d+|[A-Za-z0-9_]+)\s*\|\s*turn\s*\d+\s*\]\s*)+", re.I)
PROMPT_SECTION_LEAK_RE = re.compile(
    r"^\s*---\s*(?:Workspace Artifacts|Shared Blackboard|Your Private Scratchpad|Private Whispers|Public Conversation|Channel\b).*---\s*$",
    re.I,
)


@dataclass
class Parsed:
    message: str = ""
    reply_to: Optional[str] = None
    next_speaker: Optional[str] = None
    temperature: Optional[float] = None
    actions: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    format: str = "plain"                        # json | partial_json | commands | plain | empty
    issues: List[str] = field(default_factory=list)   # protocol problems (become rejected-action events)
    flags: List[str] = field(default_factory=list)    # attribution heuristics
    truncated: bool = False

    @property
    def has_output(self) -> bool:
        return bool(self.message or self.actions or self.next_speaker or self.temperature is not None)


_NO_NOMINATION_TOKENS = {"NONE", "N_A", "NA", "NOBODY", "NO_ONE", "NULL"}


def canonicalize_claim_id(raw_target: Any) -> str:
    """Normalize valid claim IDs like 'C1', '#c1', '[C2]' to 'c1'/'c2'; leave invalid strings intact."""
    s = str(raw_target or "").strip()
    cleaned = re.sub(r"^[#<\[\(\"\'`*]+|[>\]\)\"\'`*:.,!]+$", "", s).strip()
    if re.fullmatch(r"c\d+", cleaned, re.I):
        return cleaned.lower()
    return s


_SAMPLING_PAIR = re.compile(r"([A-Za-z_][\w-]*)\s*[:=]\s*([^\s:;,|=]+)")
_SAMPLING_KEY = re.compile(r"[A-Za-z_][\w-]*\s*[:=]")
_SAMPLING_SEPARATOR = re.compile(r"[\s,;|]+")


def parse_sampling_spec(spec: str) -> Dict[str, str]:
    """Read every assignment or reject the specification as a whole."""
    spec = spec.strip()
    params = {}
    pos = 0
    while pos < len(spec):
        pair = _SAMPLING_PAIR.match(spec, pos)
        if not pair:
            raise ValueError(f"malformed sampling specification near {spec[pos:]!r}")
        key, value = pair.groups()
        canonical = key.lower().replace("-", "_")
        canonical = SAMPLING_ALIASES.get(canonical, canonical)
        if canonical in params:
            raise ValueError(f"duplicate sampling parameter {canonical!r}")
        params[canonical] = value.strip("*`<>[]\"'")
        pos = pair.end()
        if pos < len(spec):
            separator = _SAMPLING_SEPARATOR.match(spec, pos)
            if not separator:
                raise ValueError(f"malformed sampling specification near {spec[pos:]!r}")
            pos = separator.end()
    if not params:
        raise ValueError("missing sampling parameters")
    return params


def _sampling_action(spec: str) -> Dict[str, Any]:
    try:
        return {"type": "SET_SAMPLING", "params": parse_sampling_spec(spec)}
    except ValueError as exc:
        return {"type": "SET_SAMPLING", "_parse_error": f"SET_SAMPLING {exc}"}


def canonicalize_agent_id(raw_id: Any, known_names: Optional[Set[str]] = None) -> Optional[str]:
    """Cleans and canonicalizes an agent identifier (removes wrappers, punctuation, case-normalizes)."""
    if raw_id is None:
        return None
    s = str(raw_id).strip()
    # Strip trailing parenthetical annotations like "BETA (ALPHA, BETA, HERMES)"
    if "(" in s and not s.startswith("("):
        s = s.split("(", 1)[0].strip()
    # Strip markdown wrappers, quotes, brackets, @, colons
    s = re.sub(r"^[@<\[\(\"\'`*]+|[@>\]\)\"\'`*:.,!]+$", "", s).strip()
    # Replace hyphens or spaces with underscores
    s = re.sub(r"[\s-]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s).upper()[:16]

    if not s:
        return None

    if known_names:
        upper_names = {k.upper(): k for k in known_names}
        if s in upper_names:
            return upper_names[s]
        # Check if known name is substring or close match
        for uname, orig in upper_names.items():
            if s == uname or s == uname.replace("_", ""):
                return orig
    return s


def split_thinking(raw: str, api_reasoning: str = "") -> Tuple[str, str]:
    """Move <think>...</think> text out of the visible reply. Returns (visible, reasoning)."""
    raw = raw or ""
    thoughts = [api_reasoning.strip()] if api_reasoning and api_reasoning.strip() else []
    low = raw.lower()
    if "</think>" in low and "<think>" not in low.split("</think>")[0]:
        pre, raw = re.split(r"</think>", raw, maxsplit=1, flags=re.I)
        thoughts.append(pre.strip())
    thoughts += [t.strip() for t in re.findall(r"<think>(.*?)</think>", raw, flags=re.S | re.I)]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)
    if "<think>" in raw.lower():
        raw, tail = re.split(r"<think>", raw, maxsplit=1, flags=re.I)
        thoughts.append(tail.strip() + " [truncated]")
    return raw.strip(), "\n\n".join(t for t in thoughts if t)


def repair_unambiguous_json(text: str) -> str:
    """Deterministically repairs common unambiguous syntax errors in model-generated JSON.

    Fixes trailing commas, cleans markdown wrappers, and closes unclosed braces if truncated.
    Does NOT guess or fabricate missing semantic data.
    """
    cleaned = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if m:
        cleaned = m.group(1).strip()
    else:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # Remove trailing commas before closing braces or brackets: , } -> } and , ] -> ]
    cleaned = re.sub(r'"(?:[^"\\]|\\.)*"|,\s*(?=[\]}])',
                     lambda match: match.group(0) if match.group(0).startswith('"') else "", cleaned)

    # Use stack to close open delimiters in LIFO order
    stack = []
    in_string = False
    escape = False
    for ch in cleaned:
        if ch == '"' and not escape:
            in_string = not in_string
        elif in_string:
            if ch == '\\' and not escape:
                escape = True
            else:
                escape = False
        elif ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    if in_string:
        cleaned += '"'
    while stack:
        opener = stack.pop()
        cleaned += '}' if opener == '{' else ']'

    return cleaned


def _find_json(text: str) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """Return (obj, start, end) for the first JSON object that looks like a turn, else None."""
    # First check if text contains a fenced markdown JSON block
    m_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m_fence:
        candidate = repair_unambiguous_json(m_fence.group(1))
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and JSON_KEYS & obj.keys():
                return obj, m_fence.start(), m_fence.end()
        except Exception:
            pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict) and JSON_KEYS & obj.keys():
                return obj, i, end
        except ValueError:
            pass

        # Try unambiguous repair on substring if raw parse fails
        try:
            sub = text[i:]
            repaired = repair_unambiguous_json(sub)
            repaired_obj = json.loads(repaired)
            if isinstance(repaired_obj, dict) and JSON_KEYS & repaired_obj.keys():
                return repaired_obj, i, len(text)
        except Exception:
            continue

    return None


def _salvage_json(text: str) -> Dict[str, Any]:
    """Pull fields out of broken or truncated JSON (e.g. cut off by token limit)."""
    out: Dict[str, Any] = {}
    m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.S)
    if m:
        s = m.group(1)
        try:
            out["message"] = json.loads('"' + s + '"')
        except ValueError:
            out["message"] = s.replace('\\"', '"').replace("\\n", "\n")

    m = re.search(r'"next_speaker"\s*:\s*"([^"]{1,20})"', text)
    if m:
        out["next_speaker"] = m.group(1)

    m = re.search(r'"temperature_request"\s*:\s*([0-9.]+)', text)
    if m:
        out["temperature_request"] = m.group(1)

    return out


def validate_action_schema(a: Dict[str, Any], issues: List[str], known_names: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
    """Validates required action schema fields, rejecting semantically incomplete actions."""
    if not isinstance(a, dict):
        issues.append(f"Ignored non-object action: {str(a)[:60]}")
        return None

    t = a.get("type")
    if not isinstance(t, str):
        issues.append("Ignored action without a valid string type")
        return None

    t = t.strip().upper().replace("-", "_").replace(" ", "_")
    act_type = ALIASES.get(t, t)
    if act_type not in ACTION_TYPES:
        issues.append(f"Unknown action type: {act_type}")
        return None

    res = dict(a)
    res["type"] = act_type

    # Specific schema validation
    if act_type == "WHISPER":
        recip = canonicalize_agent_id(res.get("recipient"), known_names)
        if not recip:
            issues.append("WHISPER action missing valid recipient")
            return None
        res["recipient"] = recip
        if not str(res.get("content", "")).strip():
            issues.append("WHISPER action missing message content")
            return None

    elif act_type == "CLAIM":
        kind = str(res.get("kind", "claim")).strip().lower()
        if kind not in ITEM_KINDS:
            kind = "claim"
        res["kind"] = kind
        if not str(res.get("content", "")).strip():
            issues.append(f"CLAIM ({kind}) missing content text")
            return None

    elif act_type in ("CHALLENGE", "RETRACT"):
        tgt = canonicalize_claim_id(res.get("target"))
        if not tgt:
            issues.append(f"{act_type} missing target claim ID")
            return None
        res["target"] = tgt
        if not str(res.get("content", "")).strip():
            issues.append(f"{act_type} missing reason / content text")
            return None

    elif act_type == "VOTE":
        tgt = canonicalize_claim_id(res.get("target"))
        val = str(res.get("value", "")).strip().lower()
        if not tgt or not val:
            issues.append("VOTE missing target or value")
            return None
        res["target"] = tgt
        res["value"] = val

    elif act_type == "ARTIFACT_WRITE":
        name = str(res.get("name", "")).strip()
        if not name:
            issues.append("ARTIFACT_WRITE missing artifact name")
            return None
        res["name"] = name
        mode = str(res.get("mode", "append")).strip().lower()
        if mode not in ("create", "append", "replace"):
            mode = "append"
        res["mode"] = mode
        if "content" not in res:
            res["content"] = ""

    elif act_type == "ARTIFACT_READ":
        name = str(res.get("name", "")).strip()
        if not name:
            issues.append("ARTIFACT_READ missing artifact name")
            return None
        res["name"] = name
        offset = res.get("offset", 0)
        if isinstance(offset, bool) or not re.fullmatch(r"0|[1-9][0-9]{0,5}", str(offset)):
            issues.append("ARTIFACT_READ offset must be a nonnegative integer")
            return None
        res["offset"] = int(offset)

    elif act_type == "PROPOSE_NEXT":
        tgt = canonicalize_agent_id(res.get("target"), known_names)
        if tgt in _NO_NOMINATION_TOKENS and tgt not in (known_names or ()):
            return None
        if not tgt:
            issues.append("PROPOSE_NEXT missing target participant")
            return None
        res["target"] = tgt

    elif act_type == "SET_TEMPERATURE":
        val = res.get("value")
        try:
            if isinstance(val, bool):
                raise ValueError("expected a number")
            fval = float(val)
        except (TypeError, ValueError, OverflowError):
            issues.append(f"SET_TEMPERATURE invalid numeric value: {val!r}")
            return None
        if not math.isfinite(fval):
            issues.append(f"SET_TEMPERATURE non-finite value: {val!r}")
            return None
        res["value"] = fval

    elif act_type == "SET_SAMPLING":
        if res.get("_parse_error"):
            issues.append(str(res["_parse_error"]))
            return None
        raw_params = res.get("params")
        if "params" not in res:
            raw_params = {k: v for k, v in res.items() if k != "type"}
        if not isinstance(raw_params, dict) or not raw_params:
            issues.append("SET_SAMPLING missing valid sampling parameters")
            return None
        cleaned_params = {}
        for rk, rv in raw_params.items():
            try:
                ck, cv = normalize_sampling_param(rk, rv, strict=True)
            except ValueError as exc:
                issues.append(f"SET_SAMPLING {exc}")
                return None
            if ck in cleaned_params:
                issues.append(f"SET_SAMPLING duplicate sampling parameter {ck!r}")
                return None
            cleaned_params[ck] = cv
        res = {"type": "SET_SAMPLING", "params": cleaned_params}

    return res


def normalize_action(a: Any, issues: List[str], known_names: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
    return validate_action_schema(a, issues, known_names)


def _command(word: str, rest: str, lines_after: List[str], known_names: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
    """Map one '@WORD rest' command line to an action dict."""
    w = ALIASES.get(word.upper(), word.upper())
    head, sep, body = rest.partition(":")
    head, body = head.strip(), body.strip()
    if not sep:
        head, body = rest.strip(), ""

    def split_head():
        h, _, b = head.partition(" ")
        return h.strip(), (body or b.strip())

    if w == "WHISPER":
        to, text = split_head()
        to_clean = canonicalize_agent_id(to, known_names) or to
        return {"type": "WHISPER", "recipient": to_clean, "content": text}

    if w == "PUBLISH":
        ch, text = split_head()
        return {"type": "PUBLISH", "channel": ch.lstrip("#"), "content": text}

    if w == "SUBSCRIBE":
        return {"type": "SUBSCRIBE", "channel": head.lstrip("#")}

    if word.upper() == "WORLD":
        return {"type": "ARTIFACT_WRITE", "name": "world", "mode": "append", "content": body or head}

    if w in ("CREATE_GROUP", "JOIN_GROUP", "LEAVE_GROUP"):
        return {"type": w, "channel": head.lstrip("#")}

    if w in ("INVITE_GROUP", "COMPEL_GROUP", "RELEASE_GROUP"):
        parts = head.split()
        return {"type": w, "channel": parts[0].lstrip("#") if parts else "",
                "recipient": parts[1] if len(parts) > 1 else ""}

    if w == "CLAIM" or word.lower() in ITEM_KINDS:
        kind = word.lower() if word.lower() in ITEM_KINDS else (head.lower() or "claim")
        text = body if (sep or word.lower() in ITEM_KINDS) else ""
        if word.lower() in ITEM_KINDS and not sep:
            text = rest.strip()
        a = {"type": "CLAIM", "kind": kind.split()[0] if kind else "claim", "content": text}
        if kind.startswith("task") and len(kind.split()) > 1:
            a["assignee"] = kind.split()[1]
        return a

    if w == "CHALLENGE":
        tgt, text = split_head()
        return {"type": "CHALLENGE", "target": tgt, "content": text}

    if w == "RETRACT":
        tgt, text = split_head()
        return {"type": "RETRACT", "target": tgt, "content": text}

    if w == "VOTE":
        parts = (head + " " + body).split()
        return {"type": "VOTE", "target": parts[0] if parts else "", "value": parts[1] if len(parts) > 1 else ""}

    if w == "UPDATE_NOTE":
        return {"type": "UPDATE_NOTE", "content": body or head}

    if w == "SET_STYLE":
        return {"type": "SET_STYLE", "content": body or head}

    if w == "PROPOSE_TOPIC":
        return {"type": "PROPOSE_TOPIC", "content": body or head}

    if w == "ARTIFACT_READ":
        parts = head.split()
        return {"type": "ARTIFACT_READ", "name": parts[0] if parts else "",
                "offset": parts[1] if len(parts) == 2 else (0 if len(parts) == 1 else "invalid")}

    if w == "ARTIFACT_LIST":
        return {"type": "ARTIFACT_LIST"}

    if w == "ARTIFACT_WRITE":
        bits = head.split()
        mode = word.lower() if word.lower() in ("append", "replace", "create") else (bits[1].lower() if len(bits) > 1 else "append")
        text = "\n".join([body] + lines_after).strip() if lines_after else body
        return {"type": "ARTIFACT_WRITE", "name": bits[0] if bits else "", "mode": mode, "content": text}

    if w == "FINISH":
        return {"type": "FINISH", "content": body or head}

    if w == "END_RUN":
        return {"type": "END_RUN", "content": body or head}

    if w == "PROPOSE_NEXT":
        tgt = canonicalize_agent_id(head, known_names) or head
        return {"type": "PROPOSE_NEXT", "target": tgt}

    if w == "SET_TEMPERATURE":
        return {"type": "SET_TEMPERATURE", "value": head}

    if w == "SET_SAMPLING":
        spec = rest.strip().lstrip(":").strip()
        shorthand = SAMPLING_SHORTHANDS.get(word.upper().removeprefix("SET_"))
        if shorthand and not _SAMPLING_KEY.search(spec):
            values = _SAMPLING_SEPARATOR.split(spec)
            if len(values) > len(shorthand):
                return {"type": "SET_SAMPLING", "_parse_error": "SET_SAMPLING too many shorthand values"}
            return {"type": "SET_SAMPLING", "params": dict(zip(shorthand, values))}
        return _sampling_action(spec)

    return None


def _extract_commands(text: str, p: Parsed, known_names: Optional[Set[str]] = None) -> str:
    """Strip NEXT:/TEMP:/@command lines out of text. Returns the remaining speech."""
    lines = text.splitlines()
    speech, i = [], 0
    while i < len(lines):
        line = lines[i]
        c = CTRL_RE.match(line.strip())
        if c:
            key, val = c.group(1).upper(), c.group(2).strip()
            if key.startswith("NEXT"):
                p.next_speaker = p.next_speaker or canonicalize_agent_id(val, known_names) or val
            elif key.startswith("TEMP") and not _SAMPLING_KEY.search(val):
                p.temperature = p.temperature if p.temperature is not None else val.strip("*`<>[]\"'")
            else:
                validated = validate_action_schema(_sampling_action(f"{key}: {val}"), p.issues, known_names)
                if validated:
                    p.actions.append(validated)
            i += 1
            continue

        m = CMD_RE.match(line)
        if m and (m.group(1).upper() in ALIASES or m.group(1).upper() in ACTION_TYPES or m.group(1).lower() in ITEM_KINDS):
            word, rest = m.group(1), m.group(2)
            after = []
            if ALIASES.get(word.upper(), word.upper()) == "ARTIFACT_WRITE" and word.upper() != "WORLD":
                j = i + 1
                while j < len(lines) and not CMD_RE.match(lines[j]) and not CTRL_RE.match(lines[j].strip()):
                    after.append(lines[j])
                    j += 1
                i = j
            else:
                i += 1
            a = _command(word, rest, after, known_names)
            if a:
                validated = validate_action_schema(a, p.issues, known_names)
                if validated:
                    p.actions.append(validated)
            continue

        t = TAIL_NEXT_RE.search(line)
        if t:
            p.next_speaker = p.next_speaker or canonicalize_agent_id(t.group(1), known_names) or t.group(1)
            line = line[:t.start()].rstrip()

        speech.append(line)
        i += 1

    return "\n".join(speech).strip()


def sanitize_speaker(text: str, me: str, names: Set[str], flags: List[str]) -> str:
    """Never let a reply speak for someone else: cut at the first line labelled as another participant."""
    out = []
    for line in text.splitlines():
        if PROMPT_SECTION_LEAK_RE.match(line):
            flags.append("leaked prompt section header; reply cut there")
            break
        has_turn_tag = False
        while True:
            t_match = TRANSCRIPT_TAG_RE.match(line)
            if t_match:
                has_turn_tag = True
                line = line[t_match.end():]
            m = LABEL_RE.match(line)
            if not m:
                break
            cand = canonicalize_agent_id(m.group(1), names)
            if cand and cand in names:
                if cand != me:
                    flags.append(f"wrote a line as {cand}; reply cut there")
                    line = None
                    break
                if out and any(x.strip() for x in out):
                    if has_turn_tag:
                        flags.append("hallucinated subsequent transcript turn mid-reply; reply cut there")
                        line = None
                        break
                    flags.append("repeated its own speaker label mid-reply")
                line = line[m.end():]
            else:
                break
        if line is None:
            break
        out.append(line)
    text = "\n".join(out).strip()
    if re.match(rf"^\W*{re.escape(me)}\b\s*[,!]", text, re.I):
        flags.append("addressed itself by name")
    return text


def parse_reply(
    raw: str,
    api_reasoning: str = "",
    me: str = "",
    names: Tuple[str, ...] = (),
    max_chars: int = 3500,
) -> Parsed:
    """Parses a model output string into visible speech and validated actions."""
    p = Parsed()
    known_names = {n.upper() for n in names}
    visible, p.reasoning = split_thinking(raw, api_reasoning)
    if not visible:
        p.format = "empty"
        return p

    found = _find_json(visible)
    if found:
        obj, start, end = found
        p.format = "json"
        outside = re.sub(r"```(?:json)?", "", visible[:start] + " " + visible[end:]).strip()
        msg = obj.get("message", "")
        if not isinstance(msg, str):
            p.issues.append("message was not text")
            msg = json.dumps(msg) if msg is not None else ""
        if not msg and outside:
            msg = outside

        p.reply_to = obj.get("reply_to") if isinstance(obj.get("reply_to"), str) else None
        p.next_speaker = canonicalize_agent_id(obj.get("next_speaker"), known_names)
        p.temperature = obj.get("temperature_request")
        if "sampling_request" in obj:
            valid_sr = validate_action_schema({"type": "SET_SAMPLING", "params": obj["sampling_request"]}, p.issues, known_names)
            if valid_sr:
                p.actions.append(valid_sr)

        acts = obj.get("actions", [])
        if not isinstance(acts, list):
            p.issues.append("actions was not a list")
            acts = []
        for a in acts:
            valid_a = validate_action_schema(a, p.issues, known_names)
            if valid_a:
                p.actions.append(valid_a)

        unknown = set(obj) - JSON_KEYS - {"reply_to"}
        if unknown:
            p.issues.append(f"ignored unknown fields: {', '.join(sorted(unknown))[:80]}")
        if p.issues:
            p.format = "partial_json"

    elif visible.lstrip().startswith("{") or '"message"' in visible:
        salvaged = _salvage_json(visible)
        p.format = "partial_json"
        p.issues.append("invalid or truncated JSON; salvaged what was readable")
        msg = salvaged.get("message", "") if salvaged else re.sub(r"[{}\"]", "", visible)
        p.next_speaker = canonicalize_agent_id(salvaged.get("next_speaker"), known_names)
        p.temperature = salvaged.get("temperature_request")
    else:
        msg = visible

    before_acts = len(p.actions)
    msg = _extract_commands(msg, p, known_names)
    if p.format == "plain" and (len(p.actions) > before_acts or p.next_speaker or p.temperature is not None):
        p.format = "commands"

    me_canonical = canonicalize_agent_id(me, known_names) or me.upper()
    p.message = sanitize_speaker(msg, me_canonical, known_names, p.flags)[:max_chars]

    if p.next_speaker is not None:
        ns = canonicalize_agent_id(p.next_speaker, known_names)
        if ns in _NO_NOMINATION_TOKENS and ns not in known_names:
            p.next_speaker = None
        elif ns not in known_names:
            p.issues.append(f"unknown next speaker {str(p.next_speaker)[:20]!r}")
            p.next_speaker = None
        else:
            p.next_speaker = ns

    if p.temperature is not None:
        try:
            if isinstance(p.temperature, bool):
                raise ValueError("expected a number")
            p.temperature = float(p.temperature)
            if not math.isfinite(p.temperature):
                raise ValueError("expected a finite temperature")
        except (TypeError, ValueError, OverflowError):
            p.issues.append(f"temperature request was not a number: {str(p.temperature)[:20]!r}")
            p.temperature = None

    return p
