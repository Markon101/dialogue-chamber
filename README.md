# The Dialogue Chamber — Evolution Edition

A research harness for bounded multi-agent simulations, emergent institutional dynamics, adversarial collaboration, and context-aware LLM collective intelligence.

Runs locally on **Android Termux** (optimized for Snapdragon 8 Elite NPU/CPU with `llama.cpp`) and connects dynamically to remote models via **OpenRouter**.

---

## Key Features

- **Python-First Orchestration**: Complete experiment lifecycle, concurrency locks, wake-lock power management, server PID/PGID tracking, and signal handling.
- **Tokenizer-Aware Context Intelligence**: Exact token counting via server `/tokenize` endpoint with conservative estimation fallback. Separate accounting for system instructions, topic, scratchpads, blackboards, whispers, artifacts, output reservation, and safety headroom.
- **Dynamic OpenRouter Multi-Model Support**: Any agent can run any OpenRouter model (free or paid) alongside local models. Features live model discovery, local catalog caching, search, filtering, and per-agent fallback chains.
- **Atomic Checkpointing & Interrupted-Run Recovery**: Turn-by-turn state checkpoints (`checkpoint.json`), append-only event stream (`events.jsonl`), immutable configuration manifests (`manifest.json`), and deterministic event log replay without re-running inference.
- **Action Protocol Reliability**: Canonical agent IDs, schema validation, deterministic repair of unambiguous model formatting errors (trailing commas, markdown fences, unclosed brackets from truncation).
- **Comprehensive Observability**: Full markdown and JSON telemetry (`metrics.md`, `metrics.json`, `events.md`, `transcript.md`, `transcript_full.md`), token rates, Jaccard repetition heuristics, idea resurrection tracking, and device thermal/memory probes.

---

## Quick Start

### Programmable societies and worlds

For proposed additions supporting freely organizing and evolving societies, see the
[society evolution design](SOCIETY_EVOLUTION_DESIGN.md), including current limitations,
revisable institutions, persistent history, and suggested experiments.

The `society_world` preset starts three agents in a settlement whose map changes
after public decisions. They can create working groups, invite other agents,
accept invitations, post privately within groups, and add proposals to a shared
world ledger. The steward has a configured power to compel group membership.
The rules are enforced by the harness; whether agents choose to use them and
what kind of world they create depend on the model and the conversation.

Try it without any model server or network requests:

```bash
CHAMBER_COPY_DIR='' python3 chamber.py --preset society_world --mock --turns 6
```

For a live run, the preset defaults to a free OpenRouter model and requires
`OPENROUTER_API_KEY` in the environment. Free-model availability and rate limits
can change. No local model is used by this preset:

```bash
CHAMBER_COPY_DIR='' python3 chamber.py --preset society_world --free-only --turns 6
```

To select another model for every participant, add for example
`--agent-model STEWARD=provider/model` and corresponding overrides for SURVEYOR
and ARCHIVIST. Paid models require `--allow-paid`; check their pricing first.

To make your own society, copy and edit [examples/society.json](examples/society.json),
then run `python3 chamber.py --config examples/society.json --mock`. Set the
topic, identities, models, turn limit, world seed, and `society.authorities` in
that file. Use `--free-only` for live runs restricted to free models. Agent
names in `authorities` must match the roster.

Agents use `@CREATE_GROUP name`, `@INVITE_GROUP name AGENT`,
`@JOIN_GROUP name`, `@LEAVE_GROUP name`, `@PUBLISH name: message`, and
`@WORLD: text` (a one-line addition to the world ledger). A configured authority who belongs to a group can also
use `@COMPEL_GROUP name AGENT` and `@RELEASE_GROUP name AGENT`. Invitation does
not add a member until they join; compulsion adds them and prevents leaving until
release. It does not force them to speak or agree. Group creation, invitations,
and compulsion respect the configured limits.

Group names and membership are visible to all agents; posts are visible only to
current members, who receive up to six recent posts including those written before
they joined. Agents can still share what they know in public. The run operator can
inspect all group posts and whispers in the output files. The transcript is public
speech, `society.md` summarizes groups and their posts, `events.md` records accepted
and rejected actions, and `artifacts/world.md` is the shared, agent-written ledger.
Writing to the ledger does not automatically make a claim true or settled.

When `world.md` grows, prompts show a bounded excerpt of its beginning and latest
changes, with an omission marker. Use `@READ world OFFSET` to inspect any 1,500
character window; offsets start at zero and count characters, not UTF-8 bytes.
`@READ world` still starts at the beginning. A JSON `ARTIFACT_READ` action accepts
`"offset": 1200` for the same operation. Accepted reads appear in that agent's next
prompt with the file version and character range. Unread portions remain in the
artifact and in saved checkpoints. These windows are bounded for the phone's context
budget; they are not a summary of omitted material.

Runs save their state automatically, including society membership, private posts,
notes, and the world ledger. See **Saving and continuing runs** below. Event replay
displays recorded events; it does not recreate the complete simulation. Mock replies
test the plumbing, not social or creative quality. Inspect the transcript, action
outcomes, and truncation counts in live runs.

### Defining starting organizations and communication paths in JSON

Experiment JSON can define an initial organization, let it evolve, or freeze its
structure for a control experiment. See [examples/defined_society.json](examples/defined_society.json)
and the editor-compatible [JSON Schema](schemas/society_config.schema.json).

```bash
CHAMBER_COPY_DIR='' python3 chamber.py --config examples/defined_society.json --mock
```

The example starts a council and a workers' group, seeds membership, restricts
speech and whispers to directed routes, and provides a coordinator-only announcement
channel. It uses local endpoints for live inference; `--mock` needs no model server.

The relevant configuration fields are:

```json
{
  "society": {
    "authorities": [],
    "frozen": false,
    "groups": {
      "council": {"members": ["A", "B"], "frozen": true},
      "commons": {"members": ["C"], "invited": ["B"], "admission": "open"}
    }
  },
  "communication": {
    "public": {"A": ["B"], "B": ["A", "C"], "C": ["B"]},
    "whispers": {"A": ["B"], "B": ["A"]}
  }
}
```

This is a fragment to merge into a full config with matching agents. Groups can
specify `members`, `invited`, `compelled`, `creator`, `admission`, and `frozen`.
Compelled participants must already be members; a specified creator must be a member.
Members automatically subscribe to their group's channel within subscription limits.
Names, references, duplicate members, and inconsistent memberships are validated
before generation. The schema covers these definition fields; runtime validation
also checks roster references and limits without requiring a schema package.

`admission` defaults to `invite`; `open` allows joining without an invitation.
`society.frozen: true` rejects all group creation, invitations, joining, leaving,
compulsion, and release. A group's own `frozen: true` freezes just that group's
membership and invitations. Posting and conversation continue. With both flags
false, the initial organization is a starting condition that participants can change
using the implemented membership actions. Neither flag freezes the world ledger or
participants' speech. Authority is still configured rather than elected.

`communication.public` maps a speaker to recipients of their ordinary speech.
Speakers also see their own speech. `communication.whispers` maps senders to permitted
recipients; prohibited whispers are rejected. In a supplied mapping, an omitted
sender has no recipients. If the entire mapping is omitted, existing unrestricted
behavior remains. These routes stay fixed for the run and survive continuation.

`communication.channels` can restrict an existing channel with `readers` and
`writers` arrays. Declare ordinary channels in top-level `channels`; group channels
are created by `society.groups`. Writers must also be readers. Reader permission is
required to subscribe; writer permission is required to publish. Group membership
is additionally required for group posts. Unspecified channels retain their normal
permissions. Channel readers still need a subscription to receive posts.

These controls govern conversation paths, not all information in the experiment:
shared artifacts and the blackboard remain shared, memberships remain public, and
`dm_visibility: "notice"` still exposes whisper metadata. Participants can relay
received information along allowed paths. The run operator can inspect all messages.
Transcripts are an operator's combined view; agent prompts enforce recipient filters.

Local runs, transcript outputs, generated anomaly reports/specimens, caches, and
locks are ignored by Git. The public repository contains source, definitions, tests,
and documentation rather than recorded local runs.

### Using Dory as an agent

The `dory` endpoint connects to the separate Dory-enabled llama.cpp server in
`~/gitapps/dory-test`. The chamber will not start, stop, or switch that server.
When you choose to run it, read the local [Dory handoff](../../gitapps/dory-test/README.md)
and start it through the chamber's guarded launcher:

```bash
python3 start_dory.py
```

The guard refuses to start Dory if it sees another `llama-server`. Its current launcher
uses model alias `dory` at `http://127.0.0.1:8080/v1`. The chamber verifies that
alias before a live run; if the server is unavailable or has a different model,
the run stops instead of starting another model.

For example, after the Dory server is ready:

```bash
CHAMBER_COPY_DIR='' python3 chamber.py --preset society_world \
  --agent-endpoint STEWARD=dory --free-only --turns 6
```

Use `--mock` on the same command to exercise routing without a running server.
`--agent-endpoint` works with the other roster names too; it replaces that
agent's prior endpoint and model. Select Dory directly rather than as a fallback,
so its identity and resource checks are applied. Other agents in this example
use the preset's free OpenRouter model and require `OPENROUTER_API_KEY`.

The Dory endpoint limits each chamber prompt to a **3,072-token working budget**,
including a 256-token output reservation and a 512-token safety margin; it also
limits output to 256 tokens and uses a 600-second HTTP timeout with no automatic
retry. Oversized prompts are rejected before generation. Token counts may be
estimated when the server tokenizer is unavailable; the cap is not a guarantee of
memory use or model accuracy. If the server reports a smaller allocated context,
the chamber uses that smaller limit. These client limits do **not** change the server's
allocated context: the existing Dory launcher allocates 8,192 tokens and loads a
large model. Keep other models stopped and watch memory on the phone. The local
Dory handoff records one 6,466-token test with an arithmetic error and does not
establish a maximum safe context.

The Dory launcher and the chamber's managed local endpoint both default to port
8080. The chamber also rejects a run that combines Dory with a different managed
local server, even on separate ports: use one local server at a time. Before local
inference and managed launches it checks visible `llama-server` processes and
refuses a conflict without killing them. `start_dory.py` and the managed launcher
also share a launch lock. Independently starting servers outside these launchers
can bypass the lock, and process visibility is limited by the operating system;
do not start another server separately while one is running. `DORY_ENDPOINT` may
point to a different port if the single Dory server is configured to listen there.

### 1. Diagnostics & Health Check
Inspect system health, endpoints, Termux environment, and storage:
```bash
./run_prompt.sh --doctor
```

Inspect context limits and token budgets for the selected preset:
```bash
./run_prompt.sh --context-info
./run_prompt.sh -p weird_hierarchy --context-info
```

Run a zero-cost simulation (dry run) showing budgets and planning:
```bash
./run_prompt.sh -p weird_hierarchy --dry-run "Draft an Imperial Mandate"
```

### 2. Running Experiments

**Freeform prompt (default two-agent baseline):**
```bash
./run_prompt.sh "What is the nature of latent space?"
```

**Weird Hierarchy (5 local Gemma agents on Snapdragon NPU):**
```bash
./run_hierarchy.sh
# Or via CLI:
./run_prompt.sh --preset weird_hierarchy -t 15 "Draft a mandate on illegal thought"
```

**Mixed Local & Remote Debate (Gemma vs Hermes with claims & challenges):**
```bash
./run_prompt.sh --preset hermes_vs_local_debate -t 10 "How would uphill-flowing rivers work?"
```

**Per-Agent Model Customization:**
```bash
# Assign specific OpenRouter models to agents
./run_prompt.sh --preset two_agent_baseline \
  --agent-model "LOCAL=google/gemma-4-31b-it:free" \
  --agent-model "HERMES=deepseek/deepseek-v4.1-flash-1017" \
  -t 6 "Explore emergent cooperation"
```

**Enforce Free Models Only:**
```bash
./run_prompt.sh --preset two_agent_baseline --free-only -t 6 "Free tier experiment"
```

**Batch Suite Execution:**
```bash
./run_batch.sh
# Or with offline mock double:
./run_batch.sh --mock
```

### Saving and continuing runs

The chamber writes a self-contained, versioned `checkpoint.json` before generation,
after each processed turn, and on a graceful stop. Press **Ctrl+C** to request a stop;
the current inference call must return before the chamber can finish saving. Copying
a completed run folder preserves its checkpoint, events, artifacts, and reports.
The optional storage mirror also includes the checkpoint and event stream.

Continue a saved run using its saved roster, models, configuration, and history:

```bash
CHAMBER_COPY_DIR='' python3 chamber.py --resume runs/YOUR_RUN \
  --turns 24 --requests 60 --run-dir runs/YOUR_RUN_continued
```

`--turns` and `--requests` are **cumulative totals**, including work already saved.
Raise the turn limit to extend a completed run; an interrupted run can continue
under its existing limits. Token, time, and spend budgets also carry forward.
The time budget counts active run time, excluding the gap between sessions.
Saved mock runs stay offline; `--mock` can also continue a live run with demo replies.

Continuation writes into a **new directory**, leaving the source run intact. Omit
`--run-dir` to create one automatically. The new manifest records the parent
checkpoint's path, turn, and SHA-256 hash. No preset or original config file is
required; `--resume` also accepts the path to `checkpoint.json` itself.

Saved state includes group membership and authority, channel history, whispers,
claims and votes, notes, speaking styles, artifact contents/versions/read markers,
finished status, pending speaker selection, scheduler randomness, and usage counters.
Finished agents remain finished. A forced termination can lose an in-flight response
and usage since the last checkpoint; continuation starts from the saved boundary.
Live model replies may differ after continuation even when the saved state matches.

Checkpoints made before this complete-saving format cannot be safely continued and
are rejected without changing the original run. Missing or corrupt saves also fail
instead of silently starting a fresh experiment.

**Replay Recorded Event Stream Deterministically:**
```bash
python3 chamber.py --replay-events runs/two_agent_baseline_20261103_035000/events.jsonl
```

---

## OpenRouter Model Discovery

Discover, search, and inspect available OpenRouter models:

```bash
# List free models
python3 chamber.py models list --free --limit 20

# Search models by keyword
python3 chamber.py models search deepseek
python3 chamber.py models search gemma --free

# View model context, pricing, and capabilities
python3 chamber.py models info "google/gemma-4-31b-it:free"

# Force refresh the local catalog cache
python3 chamber.py models refresh
```

---

## Anomaly Laboratory & Living Catalog

The harness treats runtime anomalies, protocol slips, and emergent phenomena as structured experimental material. It deterministically scans run artifacts without LLM calls, classifies specimens into a 5-category taxonomy, and maintains a living catalog:

```bash
# Scan runs directory, extract reproducible specimens, and update catalog:
python3 chamber.py --anomalies runs

# Output specimens in JSON format:
python3 chamber.py --anomalies runs --json
```

Living catalog: [anomalies/CATALOG.md](file:///data/data/com.termux/files/home/projects/dialogue_chamber/anomalies/CATALOG.md)  
Archived specimens: `anomalies/specimens/*.json`

### Taxonomy Categories
1. **`harness_defect`**: Parser, prompt-template, or state-tracking bug.
2. **`model_behavior`**: Specific capability limitation of the model family or scale.
3. **`experimental_artifact`**: Side-effect of task framing, proxy metrics, or limits.
4. **`emergent_phenomenon`**: Unprogrammed multi-agent dynamic arising from interaction constraints.
5. **`unknown`**: Insufficient evidence to isolate a cause; preserved for falsification.

---

## Empirical Benchmarks & Epistemic Metrics

### Asymmetric Lantern Benchmark (R1)
Compares `round_robin` vs `self_directed` scheduling under distributed information:

```bash
# Run 2-trial comparison on local Gemma 4 E4B on Snapdragon NPU:
python3 bench_lantern.py --trials 2 --turns 9 --max-tokens 400 --reasoning off --model gemma4-htp --stop
```

### Epistemic Metrics
- **`solved_turn`**: Fast regex proxy matching culprit mentions. Subject to Turn-1 false positives on initial suspect enumerations.
- **`verified_synthesis_turn`**: Rigorous epistemic metric requiring that all clue-holding agents have spoken and non-enumerative deductive reasoning has eliminated alternatives.

---

## Context Budget Formula

For every agent request, usable context is budgeted as:

$$\text{Total Prompt} + \text{Reserved Output} + \text{Safety Margin} \le \text{Effective Usable Context}$$

Separate accounting tracks:
1. System instructions & operational boundary
2. Agent personality, role, and style rules
3. Current topic & active objectives
4. Private scratchpads & initial knowledge
5. Workspace artifacts (read and visible)
6. Shared blackboard claims, challenges, and votes
7. Private whispers (visible only to sender/recipient)
8. Tool definitions & protocol instructions
9. Public conversation transcript (trimmed oldest-first under context pressure)
10. Reserved output tokens (default 4096 or agent limit)
11. Configured safety margin (default 512–2048 tokens)

---

## Testing & Verification

Run the comprehensive unit and integration test suite:

```bash
# Run all verification suites
python3 -m unittest \
  test_anomaly_lab.py \
  test_context.py \
  test_openrouter_models.py \
  test_checkpoints.py \
  test_protocol.py \
  test_chamber.py \
  test_phase_a.py \
  test_mock.py
```

All shell scripts pass syntax validation:
```bash
for f in *.sh; do bash -n "$f" && echo "$f: OK"; done
```
