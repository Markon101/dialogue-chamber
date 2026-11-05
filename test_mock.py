"""Smoke test with mocked API responses (no network). Run: python3 test_mock.py"""
import os
os.environ["OPENROUTER_API_KEY"] = "dummy-secret"
import dialogue_chamber as dc

dc.MAX_TURNS, dc.MODE, dc.COPY_DIR = 5, "self_directed", ""
dc.OUTPUT_FILE, dc.FULL_OUTPUT_FILE = "mock_transcript.txt", "mock_transcript_full.txt"
replies = iter([
    ("Hello!\nNEXT: LOCAL\nTEMP: 9", "private thought"),  # picks itself; temp clamped; reasoning hidden
    ("I continue.\nNEXT: nobody", ""),                     # invalid NEXT -> alternate
    ("HERMES: Echoed label.\nTEMP: 0.5", ""),              # label stripped; missing NEXT -> alternate
    ("<think>hmm</think>Back again. NEXT: HERMES", ""),    # inline think + trailing NEXT
    ("Last word.\nNEXT: HERMES", ""),                      # request ignored: turn limit wins
])
calls = []
def fake_chat(ep, model, key, system, user, t, stops=None):
    calls.append((ep, key, t, system, user)); return next(replies)
dc.chat = fake_chat
dc.main()

clean, full = open(dc.OUTPUT_FILE).read(), open(dc.FULL_OUTPUT_FILE).read()
order = [l.split()[-1] for l in clean.splitlines() if l.startswith("[0")]
assert len(calls) == 5, "turn limit enforced"
assert order == ["LOCAL", "LOCAL", "HERMES", "LOCAL", "HERMES"], order
assert calls[1][2] == dc.TEMP_MAX, "temperature clamped"
assert calls[0][1] == "" and calls[2][1] == "dummy-secret", "key only sent to the OpenRouter agent"
assert all("dummy-secret" not in c[3] + c[4] for c in calls), "key never in model-visible text"
assert "dummy-secret" not in clean + full, "key never in transcripts"
assert "private thought" not in clean and "private thought" in full, "reasoning only in full file"
assert "hmm" not in clean and "<think>" not in clean
assert "NEXT:" not in clean.split("Topic")[1] and "TEMP:" not in clean.split("Topic")[1]
assert "LOCAL: I continue." in calls[2][4], "shared transcript passed to the next speaker"
print("MOCK TEST OK")
