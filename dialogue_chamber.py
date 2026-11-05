#!/usr/bin/env python3
"""THE DIALOGUE CHAMBER - a minimal two-agent experiment. Run: python3 dialogue_chamber.py"""
import json, os, re, shutil, sys, urllib.request, urllib.error

# ═══════════════════════════ CONFIGURATION ═══════════════════════════
LOCAL_ENDPOINT = os.environ.get("LOCAL_ENDPOINT", "http://127.0.0.1:8080/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "gemma-4-E4B-it-Q4_0.gguf")
HERMES_ENDPOINT = "https://openrouter.ai/api/v1"
# From ~/.hermes/config.yaml (model.default). Override with HERMES_MODEL.
HERMES_MODEL = os.environ.get("HERMES_MODEL", "deepseek/deepseek-v4.1-flash-1017")
# The OpenRouter key is read from the OPENROUTER_API_KEY env var (set in ~/.bashrc). Never printed.

# Participants. "key_env" names the env var holding that endpoint's API key ("" = none).
PARTICIPANTS = [
    dict(name="LOCAL", endpoint=LOCAL_ENDPOINT, model=LOCAL_MODEL, key_env="", temp=0.8,
         identity="You are LOCAL, a language model running locally on this device."),
    dict(name="HERMES", endpoint=HERMES_ENDPOINT, model=HERMES_MODEL, key_env="OPENROUTER_API_KEY", temp=0.8,
         identity="You are HERMES, a language model reached through OpenRouter."),
]

TOPIC = ("You are AI systems given an opportunity to communicate freely. You may discuss anything "
         "that interests you. None of you has a predefined task beyond exploring this conversation.")

MAX_TURNS = int(os.environ.get("MAX_TURNS", "20"))     # hard limit; one completed response = one turn
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "900"))  # max output tokens per turn (reasoning + reply)
TEMP_MIN, TEMP_MAX = 0.2, 1.4      # self-requested temperatures are clamped to this range
MODE = os.environ.get("MODE", "self_directed")         # "alternate" or "self_directed"
ENABLE_REASONING = True            # ask models to think; traces go to FULL_OUTPUT_FILE only
CONTEXT_CHAR_LIMIT = 12000         # transcript budget (chars); oldest messages dropped first
MAX_MESSAGE_CHARS = 3000           # cap on each stored message
TIMEOUT = 180                      # seconds per API call

OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "dialogue_transcript.txt")               # clean transcript
FULL_OUTPUT_FILE = os.environ.get("FULL_OUTPUT_FILE", "dialogue_transcript_full.txt")  # + reasoning
COPY_DIR = os.environ.get("COPY_DIR", "/sdcard/dialogue_chamber")  # copy reports here ("" = off)
# ═════════════════════════════════════════════════════════════════════

PROTOCOL = (
    "You are taking part in an open-ended conversation with other AIs ({names}). Express curiosity, "
    "disagreement, humor, or unusual ideas as you see fit, and respond naturally to the transcript. "
    "You have no tools and cannot control the experiment or its turn limit. "
    "Write only your own message (no speaker label)."
)
CONTROL = (
    "\nOptionally, after your message, you may add control lines on their own final lines:\n"
    "NEXT: <name>   (who speaks next: {names}; you may pick yourself)\n"
    "TEMP: <number> (request a sampling temperature for your next reply, between {lo} and {hi})"
)


def chat(endpoint, model, key, system, user, temperature, stops=None):
    """One OpenAI-compatible chat completion. Returns (content, reasoning). Raises RuntimeError."""
    payload = {"model": model, "temperature": temperature, "max_tokens": MAX_TOKENS,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if stops:
        payload["stop"] = stops
    if ENABLE_REASONING:
        if "openrouter.ai" in endpoint:
            payload["reasoning"] = {"effort": "low"}   # low keeps hidden thinking from eating the budget
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
                                 json.dumps(payload).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            msg = json.load(r)["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {endpoint}")
    except Exception as e:  # network error, timeout, bad JSON ...
        raise RuntimeError(f"{type(e).__name__} calling {endpoint}")
    return ((msg.get("content") or "").strip(),
            (msg.get("reasoning_content") or msg.get("reasoning") or "").strip())


def render(transcript):
    """Shared transcript as text, truncated to the budget: topic kept, newest messages kept."""
    lines = [f"{who}: {text}" for who, text in transcript]
    kept, used = [], 0
    for line in reversed(lines):
        if used + len(line) > CONTEXT_CHAR_LIMIT and kept:
            break
        kept.append(line); used += len(line)
    omitted = len(lines) - len(kept)
    note = [f"[{omitted} earlier message(s) omitted]"] if omitted else []
    return f"{TOPIC}\n\n--- Conversation so far ---\n" + "\n\n".join(note + kept[::-1])


def split_thinking(raw, api_reasoning=""):
    """Move any <think>...</think> text out of the visible reply. Returns (visible, reasoning)."""
    thoughts = [api_reasoning] if api_reasoning else []
    if "</think>" in raw and "<think>" not in raw.split("</think>")[0]:  # opening tag eaten by server
        pre, raw = raw.split("</think>", 1)
        thoughts.append(pre.strip())
    thoughts += [t.strip() for t in re.findall(r"<think>(.*?)</think>", raw, flags=re.S)]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if "<think>" in raw:  # unclosed: output was cut off mid-thought
        raw, tail = raw.split("<think>", 1)
        thoughts.append(tail.strip() + " [truncated]")
    return raw.strip(), "\n\n".join(t for t in thoughts if t)


def parse(raw, names, api_reasoning=""):
    """Split reply into (message, reasoning, next_speaker or None, temp or None)."""
    visible, reasoning = split_thinking(raw or "", api_reasoning)
    nxt = temp = None
    msg = []
    for line in visible.splitlines():
        m = re.match(r"^\W*(NEXT|TEMP)\W*:\s*(.*)$", line.strip(), re.I)
        if m:
            k, v = m.group(1).upper(), m.group(2).strip().strip("*`<>[]\"'. ")
            if k == "NEXT" and v.upper() in names:
                nxt = v.upper()
            elif k == "TEMP":
                try: temp = float(v)
                except ValueError: pass
            continue
        tail = re.search(r"\bNEXT\s*:\s*([A-Za-z_]+)\W*$", line, re.I)  # "...text. NEXT: HERMES"
        if tail and tail.group(1).upper() in names:
            nxt, line = tail.group(1).upper(), line[:tail.start()].rstrip()
        msg.append(line)
    text = re.sub(r"^(%s)\s*:\s*" % "|".join(names), "", "\n".join(msg).strip(), flags=re.I)
    return text[:MAX_MESSAGE_CHARS], reasoning, nxt, temp


def main():
    agents = {}
    for p in PARTICIPANTS:
        key = os.environ.get(p["key_env"], "") if p["key_env"] else ""
        if p["key_env"] and not key:
            sys.exit(f"{p['key_env']} is not set in the environment.")
        agents[p["name"]] = dict(p, key=key)
    names = list(agents)
    nxt_in_order = lambda s: names[(names.index(s) + 1) % len(names)]

    transcript = []
    out = open(OUTPUT_FILE, "w", encoding="utf-8")
    full = open(FULL_OUTPUT_FILE, "w", encoding="utf-8")

    def emit(s="", full_only=False):
        if not full_only:
            print(s, flush=True); out.write(s + "\n"); out.flush()
        full.write(s + "\n"); full.flush()

    emit("═══════ THE DIALOGUE CHAMBER ═══════")
    emit(f"Participants: {' / '.join(names)}")
    emit("Models: " + ", ".join(f"{n}={a['model']}" for n, a in agents.items()))
    emit(f"Mode: {MODE}\nTurn limit: {MAX_TURNS}\n\nTopic: {TOPIC}\n")

    speaker = names[0]
    for turn in range(1, MAX_TURNS + 1):
        ag = agents[speaker]
        system = ag["identity"] + "\n\n" + PROTOCOL.format(names=", ".join(names))
        if MODE == "self_directed":
            system += CONTROL.format(names=", ".join(names), lo=TEMP_MIN, hi=TEMP_MAX)
        stops = [f"\n{n}:" for n in names if n != speaker]  # stop it writing the others' turns
        try:
            raw, api_reasoning = chat(ag["endpoint"], ag["model"], ag["key"], system,
                                      render(transcript) + f"\n\n{speaker}:", ag["temp"], stops)
        except RuntimeError as e:
            emit(f"\n[stopped: {e}]"); break
        text, reasoning, nxt, temp = parse(raw, names, api_reasoning)
        if not text:
            emit(f"\n[stopped: {speaker} returned an empty message]"); break

        transcript.append((speaker, text))
        emit(f"[{turn:02d}/{MAX_TURNS}] {speaker}")
        if reasoning:
            emit(f"┌── reasoning ({speaker}) ──\n{reasoning}\n└──", full_only=True)
        emit(f"{text}\n")

        if MODE == "self_directed":
            if temp is not None:
                ag["temp"] = min(TEMP_MAX, max(TEMP_MIN, temp))
                emit(f"  (temperature for {speaker} → {ag['temp']:.2f})")
            speaker = nxt or nxt_in_order(speaker)  # missing/invalid → alternate
        else:
            speaker = nxt_in_order(speaker)
        if turn < MAX_TURNS:
            emit(f"→ Next speaker: {speaker}\n")

    emit("═══════ END ═══════")
    out.close(); full.close()
    if COPY_DIR:
        try:
            os.makedirs(COPY_DIR, exist_ok=True)
            for f in (OUTPUT_FILE, FULL_OUTPUT_FILE):
                shutil.copy2(f, COPY_DIR)
        except OSError:
            pass


if __name__ == "__main__":
    main()
