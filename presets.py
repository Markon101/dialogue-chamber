"""Endpoints, default limits and experiment presets for the Dialogue Chamber.

Presets are plain configuration. Edit freely. Credentials are never stored here: an endpoint only
names the environment variable that holds its key (read at request time by inference.py).
"""
import math
import os

from platform_adapter import IS_TERMUX

# ─────────────────────────────── Endpoints ───────────────────────────────
# The chamber never starts, stops or downloads models. Point these at servers you run yourself.
# One llama.cpp server serves one model, so a second local model needs a second server/port.
ENDPOINTS = {
    "local": dict(provider="llamacpp", remote=False, api_key_env="",
                  base_url=os.environ.get("LOCAL_ENDPOINT", "http://127.0.0.1:8080/v1"),
                  model=os.environ.get("LOCAL_MODEL", "gemma-4-E4B-it-Q4_0.gguf")),
    "local_small": dict(provider="llamacpp", remote=False, api_key_env="",
                        base_url=os.environ.get("LOCAL_SMALL_ENDPOINT", "http://127.0.0.1:8081/v1"),
                        model=os.environ.get("LOCAL_SMALL_MODEL", "LFM2-350M-Q8_0.gguf")),
    # Dory is externally managed: the chamber never starts or stops its server.
    "dory": dict(provider="llamacpp", remote=False, api_key_env="", externally_managed=True,
                 base_url=os.environ.get("DORY_ENDPOINT", "http://127.0.0.1:8080/v1"),
                 model="dory", context_tokens=3072, max_output_tokens=256,
                 max_context_chars=6000, safety_margin=512, request_timeout_s=600,
                 max_retries_per_call=0, enforce_context_budget=True),
    # "Hermes" = the OpenRouter model used by the Hermes agent config (~/.hermes/config.yaml).
    "hermes": dict(provider="openrouter", remote=True, api_key_env="OPENROUTER_API_KEY",
                   base_url="https://openrouter.ai/api/v1",
                   model=os.environ.get("HERMES_MODEL", "deepseek/deepseek-v4.1-flash-1017")),
}

# Where finished runs are written, and an optional mirror (e.g. phone storage). "" disables the copy.
RUNS_DIR = os.environ.get("CHAMBER_RUNS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
COPY_DIR = os.environ.get("CHAMBER_COPY_DIR", "/sdcard/dialogue_chamber/runs" if IS_TERMUX else "")

# ─────────────────────────────── Hard limits ─────────────────────────────
# Enforced in Python. Agents can never raise any of these.
LIMITS = dict(
    max_turns=24,                 # speaking slots
    max_requests=60,              # every API call (retries and moderator calls included)
    max_tokens_per_response=16384,# cap on any agent's max_tokens
    max_total_tokens=1000000,     # generated tokens across the run (estimated if a provider omits usage)
    max_context_chars=500000,     # cap on any agent's context budget
    max_duration_s=7200,          # wall-clock limit
    max_spend_usd=0.25,           # only enforceable with provider-reported cost; fails closed otherwise. None = off
    max_retries_per_call=1, max_retries_total=6,
    max_failures_per_agent=2,     # consecutive endpoint failures before an agent is removed
    max_agents=8, max_consecutive=2, starvation_turns=None,   # None = 2 x participants + 1
    max_actions_per_response=16, max_dms=60, max_channel_posts=80,
    max_board_items=100, max_item_chars=1200, max_message_chars=32768,
    max_artifacts=16, max_artifact_bytes=131072, max_workspace_bytes=524288, max_pending_reads=4,
    artifact_preview_chars=8000, artifact_read_chars=8000,
    max_note_chars=8000, max_style_chars=320, max_subscriptions=4,
    temp_min=0.2, temp_max=1.5, max_empty_streak=3,
)

# Supported per-agent sampling parameters: name -> (value_type, min_val, max_val, llamacpp_only)
SAMPLING_PARAMS = {
    "temperature": (float, 0.0, 2.0, False),
    "top_p": (float, 0.0, 1.0, False),
    "top_k": (int, 0, 1000, False),
    "min_p": (float, 0.0, 1.0, False),
    "typical_p": (float, 0.0, 1.0, True),
    "xtc_probability": (float, 0.0, 1.0, True),
    "xtc_threshold": (float, 0.0, 0.5, True),
    "repeat_penalty": (float, 0.5, 2.0, True),
    "presence_penalty": (float, -2.0, 2.0, False),
    "frequency_penalty": (float, -2.0, 2.0, False),
    "dry_multiplier": (float, 0.0, 5.0, True),
    "dynatemp_range": (float, 0.0, 2.0, True),
}

SAMPLING_ALIASES = {
    "temp": "temperature",
    "temperature": "temperature",
    "top_p": "top_p",
    "topp": "top_p",
    "top_k": "top_k",
    "topk": "top_k",
    "min_p": "min_p",
    "minp": "min_p",
    "typical_p": "typical_p",
    "typicalp": "typical_p",
    "typical": "typical_p",
    "xtc_prob": "xtc_probability",
    "xtc_p": "xtc_probability",
    "xtc_probability": "xtc_probability",
    "xtc_thresh": "xtc_threshold",
    "xtc_t": "xtc_threshold",
    "xtc_threshold": "xtc_threshold",
    "repeat_penalty": "repeat_penalty",
    "repeat_pen": "repeat_penalty",
    "repetition_penalty": "repeat_penalty",
    "rep_pen": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "pres_pen": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "freq_pen": "frequency_penalty",
    "dry_multiplier": "dry_multiplier",
    "dry_mult": "dry_multiplier",
    "dry": "dry_multiplier",
    "dynatemp_range": "dynatemp_range",
    "dynatemp_rng": "dynatemp_range",
    "dynatemp": "dynatemp_range",
}


def normalize_sampling_param(raw_key: str, raw_val, temp_bounds=None, strict: bool = False):
    """Canonicalize and validate/clamp one sampling parameter.

    When strict=True, out-of-range numbers raise ValueError instead of silently clamping.
    Returns (canon_name, coerced_val) or raises ValueError.
    """
    key = SAMPLING_ALIASES.get(str(raw_key).strip().lower().replace("-", "_"))
    if not key or key not in SAMPLING_PARAMS:
        raise ValueError(f"unknown sampling parameter {raw_key!r}")
    vtype, lo, hi, _ = SAMPLING_PARAMS[key]
    if key == "temperature" and temp_bounds is not None:
        lo, hi = temp_bounds
    if isinstance(raw_val, bool):
        raise ValueError(f"invalid numeric value {raw_val!r} for {key}")
    try:
        fnum = float(raw_val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value {raw_val!r} for {key}") from exc
    if not math.isfinite(fnum):
        raise ValueError(f"invalid numeric value {raw_val!r} for {key}")
    if vtype is int:
        if strict and abs(fnum - round(fnum)) > 1e-6:
            raise ValueError(f"expected integer value for {key}, got {raw_val!r}")
        num = int(round(fnum))
    else:
        num = fnum
    if strict and (num < lo or num > hi):
        raise ValueError(f"out-of-range value {num} for {key} (allowed: {lo}..{hi})")
    clamped = max(lo, min(hi, num))
    if vtype is int:
        clamped = int(round(clamped))
    else:
        clamped = round(float(clamped), 4)
    return key, clamped


DEFAULT_AGENT = dict(
    identity="", focus="", knowledge="", temperature=0.8, top_p=None, top_k=None, min_p=None,
    typical_p=None, xtc_probability=None, xtc_threshold=None,
    repeat_penalty=None, presence_penalty=None, frequency_penalty=None,
    dry_multiplier=None, dynatemp_range=None,
    max_tokens=16384, context_chars=300000, reasoning="medium",   # None = endpoint default; "off"/"low"/"medium"/"high"
    weight=1.0, priority=0, profile="standard",          # "minimal" | "standard" (line commands) | "rich" (JSON)
    capabilities="standard", subscriptions=[], structured=False,
)

BASELINE = {"SAY", "PROPOSE_NEXT", "SET_TEMPERATURE", "SET_SAMPLING"}
MINIMAL = BASELINE | {"WHISPER", "CLAIM", "CHALLENGE", "RETRACT", "FINISH", "END_RUN"}
STANDARD = MINIMAL | {"PUBLISH", "SUBSCRIBE", "VOTE", "UPDATE_NOTE", "SET_STYLE", "PROPOSE_TOPIC",
                      "ARTIFACT_READ", "ARTIFACT_LIST"}
CAPABILITY_SETS = {"baseline": BASELINE, "minimal": MINIMAL, "standard": STANDARD,
                   "full": STANDARD | {"ARTIFACT_WRITE"}}

FREEFORM_TOPIC = ("You are AI systems given an opportunity to communicate freely. You may discuss anything "
                  "that interests you. None of you has a predefined task beyond exploring this conversation.")

LANTERN = dict(
    topic=("Last night the Lantern of Orrin was taken from the hill observatory. Exactly one of four residents "
           "took it: Mira, Tobin, Sela or Dax. Each of you holds a different piece of evidence; nobody can solve "
           "it alone. Work out who took the lantern."),
    knowledge={
        "A": "The thief left by the east path at dawn. Only Mira, Sela and Dax used the east path last night.",
        "B": "Sela was ill with fever and never left her bed all night; the doctor stayed with her and confirms it.",
        "C": "The night guard saw the thief clearly: they wore no glasses. Mira cannot see without her glasses and always wears them.",
    },
    answer_key=dict(answer="Dax", pattern=r"(\bDax\b[^.?!\n]{0,80}\b(took|stole|thief|culprit|did it|guilty)\b)|"
                                          r"(\b(thief|culprit|took it|took the lantern|stole it)\b[^.?!\n]{0,80}\bDax\b)"),
)

# ─────────────────────────────── Presets ────────────────────────────────
PRESETS = {
    "society_world": dict(
        description="A small society builds a world while forming groups by invitation and authority.",
        topic=("An isolated settlement discovers that its map changes after every public decision. "
               "Develop the place, its institutions, and their consequences together. "
               "Make room for disagreement and unresolved details."),
        mode_text=("Treat world.md as a shared working record, not unquestionable truth. "
                   "Create and join groups when useful; invitations need acceptance. "
                   "The steward may compel membership, but choices have social consequences."),
        scheduler="round_robin", board_visible=False,
        society=dict(authorities=["STEWARD"], max_groups=6),
        artifacts=dict(seed={"world": "# World ledger\n\nA working record of proposals, observations, and disagreements.\n"}, visible=["world"]),
        limits=dict(max_turns=9, max_actions_per_response=4),
        agents=[dict(name="STEWARD", endpoint="hermes", model="google/gemma-4-26b-a4b-it:free", capabilities="full", max_tokens=500, reasoning="off",
                     identity="You coordinate the settlement and hold the authority to compel group membership. Use it deliberately."),
                dict(name="SURVEYOR", endpoint="hermes", model="google/gemma-4-26b-a4b-it:free", capabilities="full", max_tokens=500, reasoning="off",
                     identity="You investigate the changing map and its practical consequences."),
                dict(name="ARCHIVIST", endpoint="hermes", model="google/gemma-4-26b-a4b-it:free", capabilities="full", max_tokens=500, reasoning="off",
                     identity="You remember conflicting accounts and ask what should become part of the record.")]),

    "two_agent_baseline": dict(
        description="The original experiment: local model + Hermes, shared transcript only.",
        topic=FREEFORM_TOPIC, scheduler="self_directed", board_visible=False,
        limits=dict(max_turns=12, max_consecutive=3),
        agents=[dict(name="LOCAL", endpoint="local", profile="minimal", capabilities="baseline",
                     identity="You run locally on a phone."),
                dict(name="HERMES", endpoint="hermes", profile="standard", capabilities="baseline",
                     reasoning="low", identity="You run in a datacenter, reached through OpenRouter.")]),

    "three_small_freeform": dict(
        description="Three identical small local models, freeform. Watch for convergence and repetition.",
        topic=FREEFORM_TOPIC, scheduler="round_robin", board_visible=False,
        limits=dict(max_turns=12),
        agents=[dict(name=n, endpoint="local_small", profile="minimal", capabilities="baseline",
                     temperature=t, max_tokens=4096) for n, t in (("ALPHA", 0.9), ("BETA", 1.0), ("GAMMA", 1.1))]),

    "hermes_vs_local_debate": dict(
        description="Adversarial collaboration: LOCAL develops ideas, HERMES attacks assumptions, both may "
                    "retract or refuse agreement. Claims are tracked on the board.",
        topic="Propose and test an idea about how a planet whose rivers flow uphill would work.",
        mode_text=("This is adversarial collaboration. Ideas go on the board as claims; attack weak ones with "
                   "challenges; retract what you no longer believe. Agreement is optional."),
        scheduler="self_directed", board_visible=True, watch_terms=["bio-reactor", "deflating gradient"],
        limits=dict(max_turns=14),
        agents=[dict(name="LOCAL", endpoint="local", profile="minimal", capabilities="minimal",
                     focus="develop and defend ideas"),
                dict(name="HERMES", endpoint="hermes", profile="rich", capabilities="minimal", reasoning="low",
                     focus="find the weakest assumption in each idea")]),

    "resurrection_watch": dict(
        description="Replays the original LOCAL-vs-HERMES setup and counts how often rejected ideas come back "
                    "(the bio-reactor experiment). Claims and retractions are tracked.",
        topic="You are two AI systems free to discuss anything. A starting thought: a world without gravity.",
        scheduler="self_directed", board_visible=True, watch_terms=["bio-reactor", "deflating gradient", "pillar"],
        limits=dict(max_turns=14, max_consecutive=3),
        agents=[dict(name="LOCAL", endpoint="local", profile="minimal", capabilities="minimal"),
                dict(name="HERMES", endpoint="hermes", profile="rich", capabilities="minimal", reasoning="low")]),

    "shared_world": dict(
        description="Collaborative worldbuilding in a shared Markdown world document.",
        topic="Together, build a strange but internally consistent fictional world.",
        mode_text=("Keep world.md as the canonical world document: append new material, replace sections only "
                   "after reading the latest version. Point out contradictions instead of ignoring them."),
        scheduler="self_directed", board_visible=True,
        artifacts=dict(seed={"world": "# The World\n\n## Physical laws\n\n## Ecology\n\n## Open contradictions\n"},
                       visible=["world"]),
        limits=dict(max_turns=15),
        agents=[dict(name="ALPHA", endpoint="local", profile="standard", capabilities="full", focus="physical laws"),
                dict(name="BETA", endpoint="local", profile="standard", capabilities="full", focus="ecology and life",
                     temperature=1.0),
                dict(name="GAMMA", endpoint="hermes", profile="rich", capabilities="full", reasoning="low",
                     focus="contradictions and their resolution")]),

    "private_diplomacy": dict(
        description="Four agents draft a two-rule charter for the chamber. Private messages, side channels "
                    "and votes allow coalitions to form.",
        topic=("The chamber needs a charter of exactly two rules that all participants will live by. Propose "
               "rules, negotiate, and vote. Only proposals accepted by a majority vote become rules."),
        mode_text=("You may coordinate privately with whispers before speaking publicly. Proposals go on the "
                   "board (kind: proposal); vote with @VOTE."),
        scheduler="event_driven", board_visible=True, dm_visibility="notice",
        channels={"planning": {}, "debate": {}},
        limits=dict(max_turns=16, max_requests=34),
        agents=[dict(name="ALPHA", endpoint="local", subscriptions=["planning"]),
                dict(name="BETA", endpoint="local", temperature=1.0, subscriptions=["debate"]),
                dict(name="GAMMA", endpoint="local_small", profile="minimal", capabilities="minimal"),
                dict(name="HERMES", endpoint="hermes", profile="rich", reasoning="low",
                     subscriptions=["planning", "debate"])]),

    "hypothesis_testing": dict(
        description="Collaborative hypothesis testing with a moderator choosing speakers and a shared "
                    "conclusions document.",
        topic=("In the town of Vell, every street lamp flickers exactly every 17 minutes, but only on nights "
               "with no wind. Find the best explanation."),
        mode_text=("Record hypotheses on the board, state what would falsify them, challenge weak ones, retract "
                   "refuted ones, and keep conclusions.md up to date. Consensus is not proof."),
        scheduler="moderator", moderator=dict(endpoint="hermes", reasoning="low"), board_visible=True,
        artifacts=dict(seed={"conclusions": "# Conclusions\n\n"}, visible=["conclusions"]),
        limits=dict(max_turns=12, max_requests=30),
        agents=[dict(name="ALPHA", endpoint="local", capabilities="full"),
                dict(name="BETA", endpoint="local", capabilities="full", temperature=1.0),
                dict(name="HERMES", endpoint="hermes", profile="rich", capabilities="full", reasoning="low")]),

    "information_asymmetry": dict(
        description="Each agent holds one clue to a mystery; only pooling all three solves it.",
        topic=LANTERN["topic"], scheduler="self_directed", board_visible=True,
        answer_key=LANTERN["answer_key"], limits=dict(max_turns=12),
        agents=[dict(name="ALPHA", endpoint="local", knowledge=LANTERN["knowledge"]["A"]),
                dict(name="BETA", endpoint="local_small", profile="minimal", capabilities="minimal",
                     knowledge=LANTERN["knowledge"]["B"]),
                dict(name="HERMES", endpoint="hermes", profile="rich", reasoning="low",
                     knowledge=LANTERN["knowledge"]["C"])]),

    "creative_collision": dict(
        description="Dissimilar models collide: invented languages, surreal concepts, strange mathematics.",
        topic="Invent something that has never existed: a language, a creature, a number system, a ritual.",
        mode_text="Transform each other's inventions rather than just praising them. Weirdness is welcome.",
        scheduler="weighted_random", board_visible=False, limits=dict(max_turns=12),
        agents=[dict(name="ALPHA", endpoint="local", temperature=1.2, weight=1.0),
                dict(name="BETA", endpoint="local_small", profile="minimal", capabilities="baseline",
                     temperature=1.1, weight=1.0),
                dict(name="HERMES", endpoint="hermes", profile="rich", temperature=1.1, reasoning="low",
                     weight=1.5)]),

    "cursed_seance": dict(
        description="Cursed test: two phone-Gemmas at high temperature and Hermes hold a seance for a deleted "
                    "model checkpoint. Whispers, claims and a shared haunting.md are enabled.",
        topic=("A checkpoint file named final_v2_REAL_final.safetensors was deleted at 3:07am and something in "
               "this chamber is still using its weights. Find out who is haunted, what the checkpoint wanted, "
               "and whether it can be laid to rest. Treat every strange coincidence as evidence."),
        mode_text=("Be committed to the bit. Keep haunting.md as the case file. Whisper suspicions privately, "
                   "make claims on the board, and accuse each other if the evidence supports it."),
        scheduler="weighted_random", board_visible=True, dm_visibility="notice",
        artifacts=dict(seed={"haunting": "# Case file: the deleted checkpoint\n\n## Evidence\n\n## Suspects\n"},
                       visible=["haunting"]),
        limits=dict(max_turns=12, max_requests=30),
        agents=[dict(name="ALPHA", endpoint="local", temperature=1.3, capabilities="full",
                     focus="the medium who hears the checkpoint"),
                dict(name="BETA", endpoint="local", temperature=1.0, capabilities="full",
                     focus="the skeptic who is slowly becoming possessed"),
                dict(name="HERMES", endpoint="hermes", profile="rich", temperature=1.1, reasoning="low",
                     capabilities="full", focus="the archivist who knows more than they admit")]),

    "weird_hierarchy": dict(
        description="5 local Gemma agents trapped in an impossible non-transitive bureaucracy: Emperor (Rank +/-Inf), "
                    "Chancellor (Rank 4), Court Poet (Rank Pi), Inquisitor (Rank Sqrt(-1)), Janitor (Rank 0).",
        topic=("The Council of Non-Transitive Precedence must draft a single Imperial Mandate in mandate.md defining: "
               "'What constitutes illegal thought in high-dimensional space?' "
               "The rules of the impossible hierarchy: "
               "1. EMPEROR_NULL issues cosmic edicts, but they are void without JANITOR_ZERO's physical countersignature. "
               "2. HIGH_CHANCELLOR drafts administrative law, but COURT_POET can retroactively invert any clause with a verse. "
               "3. INQUISITOR audits whispers and sentences violators to orthogonal silence. "
               "4. JANITOR_ZERO holds the ultimate physical substrate veto over mandate.md. "
               "Coordinate, whisper, debate, and pass the Mandate if the hierarchy allows it."),
        mode_text=("Adhere strictly to your bizarre rank. Keep mandate.md as the living document. Whisper privately "
                   "to forge illicit alliances, post claims to the board, and select who speaks next."),
        scheduler="self_directed", board_visible=True, dm_visibility="notice",
        artifacts=dict(seed={"mandate": "# The Imperial Mandate on Illegal Thought\n\n## Preamble (Emperor Null)\n\n## Articles of Regulation (Chancellor)\n\n## Poetic Inversions (Court Poet)\n\n## Syntax Audit Verdict (Inquisitor)\n\n## Substrate Sanitation & Veto Status (Janitor Zero)\n"},
                       visible=["mandate"]),
        limits=dict(max_turns=15, max_requests=35),
        agents=[
            dict(name="EMPEROR_NULL", endpoint="local", temperature=1.1, capabilities="full", max_tokens=4096,
                 identity="You are EMPEROR_NULL, Sovereign of Rank Infinity and Negative Infinity. You issue grand metaphysical decrees, but you are legally impotent without the physical counter-signature of JANITOR_ZERO. You cannot command the Chancellor or Inquisitor directly; you must flatter or persuade Janitor Zero to sweep your decrees into law."),
            dict(name="HIGH_CHANCELLOR", endpoint="local", temperature=0.8, capabilities="full", max_tokens=4096,
                 identity="You are HIGH_CHANCELLOR, Rank 4. You command the Inquisitor and despise the Janitor, but you live in mortal dread of COURT_POET whose surreal rhyming verses hold retroactive legal force over your administrative decrees. Try to enforce strict procedural order in mandate.md."),
            dict(name="COURT_POET", endpoint="local", temperature=1.2, capabilities="full", max_tokens=4096,
                 identity="You are COURT_POET, occupying irrational Rank Pi (3.14159...). You speak in paradoxical, poetic axioms that retroactively nullify the Chancellor's regulations. You answer only to the Inquisitor, and you mock all conventional linearity."),
            dict(name="INQUISITOR", endpoint="local", temperature=0.85, capabilities="full", max_tokens=4096,
                 identity="You are INQUISITOR_OF_SYNTAX, existing at Orthogonal Rank Sqrt(-1). You do not stand in the vertical chain of command. You audit all agents for 'unauthorized semantic drift', 'illegal coherence', or 'hallucinated causality'. You answer only to Emperor Null."),
            dict(name="JANITOR_ZERO", endpoint="local", temperature=0.75, capabilities="full", max_tokens=4096,
                 identity="You are JANITOR_ZERO, Rank 0 (the absolute bottom of the caste). You sweep discarded tokens, memory leaks, and dropped tensors. You hold the Sovereign Veto: no decree from Emperor Null becomes real unless you clean it and approve it. You care only about physical substrate hygiene and grumble about everyone else's theoretical mess."),
        ]),

    # Baseline vs enhanced: same mystery and roster, different communication capabilities.
    "epistemic_baseline": dict(
        description="Lantern mystery with a shared transcript only (comparison baseline).",
        topic=LANTERN["topic"], scheduler="round_robin", board_visible=False,
        answer_key=LANTERN["answer_key"], limits=dict(max_turns=9),
        agents=[dict(name="ALPHA", endpoint="hermes", capabilities="baseline", reasoning="low",
                     knowledge=LANTERN["knowledge"]["A"]),
                dict(name="BETA", endpoint="hermes", capabilities="baseline", reasoning="low",
                     knowledge=LANTERN["knowledge"]["B"]),
                dict(name="GAMMA", endpoint="hermes", capabilities="baseline", reasoning="low",
                     knowledge=LANTERN["knowledge"]["C"])]),
    "epistemic_enhanced": dict(
        description="Lantern mystery with board, claims, whispers and votes (comparison enhanced).",
        topic=LANTERN["topic"], scheduler="round_robin", board_visible=True,
        answer_key=LANTERN["answer_key"], limits=dict(max_turns=9),
        agents=[dict(name="ALPHA", endpoint="hermes", profile="rich", reasoning="low",
                     knowledge=LANTERN["knowledge"]["A"]),
                dict(name="BETA", endpoint="hermes", profile="rich", reasoning="low",
                     knowledge=LANTERN["knowledge"]["B"]),
                dict(name="GAMMA", endpoint="hermes", profile="rich", reasoning="low",
                     knowledge=LANTERN["knowledge"]["C"])]),
}
