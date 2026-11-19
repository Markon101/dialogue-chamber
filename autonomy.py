"""Operator-defined choices for a participant's own turns."""


def validate_policy(policy, max_tokens):
    if not isinstance(policy, dict):
        raise ValueError("autonomy must be an object")
    for key, rule in policy.items():
        if key == "focus_chars":
            if type(rule) is not int or not 1 <= rule <= 400:
                raise ValueError("autonomy.focus_chars must be an integer from 1 to 400")
        elif key in ("history_messages", "reply_tokens"):
            minimum = 0 if key == "history_messages" else 1
            if (not isinstance(rule, list) or not 1 <= len(rule) <= 16
                    or any(type(n) is not int or n < minimum for n in rule)
                    or len(set(rule)) != len(rule)):
                raise ValueError(f"autonomy.{key} requires 1 to 16 distinct integers >= {minimum}")
            if key == "reply_tokens" and max(rule) > max_tokens:
                raise ValueError(f"autonomy.reply_tokens exceeds the agent output cap of {max_tokens}")
        else:
            raise ValueError(f"unknown autonomy permission {key!r}")


def validate_settings(policy, settings):
    if not isinstance(settings, dict) or not settings:
        raise ValueError("settings must be a nonempty object")
    for key, value in settings.items():
        if key == "focus" and "focus_chars" in policy:
            if not isinstance(value, str) or len(value) > policy["focus_chars"]:
                raise ValueError(f"focus must be text up to {policy['focus_chars']} characters")
        elif key in ("reply_tokens", "history_messages") and key in policy:
            if type(value) is not int or value not in policy[key]:
                raise ValueError(f"{key} must be one of {policy[key]}")
        else:
            raise ValueError(f"no permission to change {key!r}")
