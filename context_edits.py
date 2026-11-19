"""Private, reversible views of conversation history."""

import re
from copy import deepcopy
from dataclasses import replace

OPERATIONS = {"hide", "replace", "summarize"}
MAX_SCOPES = 8
MAX_MESSAGE_EDITS = 16
MAX_EDIT_CHARS = 2400
MAX_REPLACEMENT = 400
MAX_SUMMARY = 1200


def validate_context_policy(policy):
    if not isinstance(policy, dict) or not 1 <= len(policy) <= MAX_SCOPES:
        raise ValueError("autonomy.context_edit requires an object with one to eight sources")
    for source, ops in policy.items():
        if not isinstance(source, str) or not re.fullmatch(r"public|whispers|#[a-z0-9_-]{1,24}", source):
            raise ValueError("context edit sources must be public, whispers, or #channel")
        if (not isinstance(ops, list) or not ops
                or any(not isinstance(op, str) or op not in OPERATIONS for op in ops)
                or len(ops) != len(set(ops))):
            raise ValueError(f"invalid context edit permissions for {source}")


def _valid_text(text, limit):
    return isinstance(text, str) and bool(text.strip()) and len(text) <= limit


def validate_context_edits(state, policy, turn):
    if not isinstance(state, dict) or any(source not in policy for source in state):
        raise ValueError("context edits include an ungranted source")
    edit_count = char_count = 0
    for source, view in state.items():
        if (not isinstance(view, dict) or set(view) != {"messages", "summary"}
                or not isinstance(view["messages"], dict)):
            raise ValueError("invalid saved context edit state")
        for target, text in view["messages"].items():
            if not isinstance(target, str) or not 1 <= len(target) <= 48:
                raise ValueError("invalid context message ID")
            op = "hide" if text is None else "replace"
            if op not in policy[source] or (text is not None and not _valid_text(text, MAX_REPLACEMENT)):
                raise ValueError(f"invalid {op} for {source} (replacements allow 1 to {MAX_REPLACEMENT} characters)")
            edit_count += 1
            char_count += len(text or "")
        summary = view["summary"]
        if summary is not None:
            if ("summarize" not in policy[source] or not isinstance(summary, dict)
                    or set(summary) != {"text", "through_turn"}
                    or type(summary["through_turn"]) is not int or not 0 <= summary["through_turn"] <= turn
                    or not _valid_text(summary["text"], MAX_SUMMARY)):
                raise ValueError(f"invalid summary for {source} (summaries allow 1 to {MAX_SUMMARY} characters)")
            char_count += len(summary["text"])
    if edit_count > MAX_MESSAGE_EDITS or char_count > MAX_EDIT_CHARS:
        raise ValueError(f"context edits exceed {MAX_MESSAGE_EDITS} messages or {MAX_EDIT_CHARS} text characters")


def update_context_edits(state, policy, edit, visible_messages, turn):
    """Validate one edit and return a new state. All source IDs refer to prior turns."""
    if not isinstance(edit, dict) or set(edit) - {"source", "op", "target", "text"}:
        raise ValueError("context edit requires source and op with optional target and text")
    source, op = edit.get("source"), edit.get("op")
    if not isinstance(source, str) or source not in policy or visible_messages is None:
        raise ValueError("context edit source is unavailable")
    if not isinstance(op, str) or (op != "restore" and op not in policy[source]):
        raise ValueError(f"operation is not granted for {source}")
    target = edit.get("target")
    if op in ("hide", "replace") or "target" in edit:
        if op == "summarize" or not isinstance(target, str) or not any(
            message.id == target and message.turn < turn for message in visible_messages
        ):
            raise ValueError("target must be a visible message from a prior turn in the selected source")
    if op not in ("replace", "summarize") and "text" in edit:
        raise ValueError(f"{op} does not accept text")

    new_state = deepcopy(state)
    view = new_state.setdefault(source, {"messages": {}, "summary": None})
    if op == "restore":
        if target is None:
            new_state.pop(source)
        else:
            view["messages"].pop(target, None)
    elif op == "summarize":
        if not any(message.turn < turn for message in visible_messages):
            raise ValueError("no visible prior messages to summarize")
        view["summary"] = {"text": edit.get("text"), "through_turn": turn - 1}
    elif op == "hide":
        view["messages"][target] = None
    else:
        if not _valid_text(edit.get("text"), MAX_REPLACEMENT):
            raise ValueError(f"replacement needs 1 to {MAX_REPLACEMENT} characters")
        view["messages"][target] = edit["text"]
    if source in new_state and not view["messages"] and view["summary"] is None:
        new_state.pop(source)
    validate_context_edits(new_state, policy, turn)
    return new_state


def render_context_view(messages, view):
    """Apply an agent's edits after the harness has checked read access."""
    if not view:
        return messages, None
    edits, summary = view["messages"], view["summary"]
    kept = []
    for message in messages:
        if summary and message.turn <= summary["through_turn"]:
            continue
        if message.id in edits:
            text = edits[message.id]
            if text is None:
                continue
            message = replace(message, content=f"[your replacement for {message.id}] {text}")
        kept.append(message)
    return kept, summary
