# Freely organizing and evolving societies

Status: design proposal, based on the checkout and saved runs reviewed on 2026-11-04.
The social and context capabilities below are proposed additions, not implemented
commands. Complete saving and continuation were implemented during this review for
new version-2 checkpoints; see [Saving and continuing runs](README.md#saving-and-continuing-runs).
JSON definitions now also support initial groups, open or invitation-only admission,
frozen organizations, directed speech/whispers, and channel reader/writer permissions;
see [the configuration guide](README.md#defining-starting-organizations-and-communication-paths-in-json).
References to saved runs below concern local evidence files excluded from the public repository.

## Design goal

Give participants the ability to choose their purposes, form overlapping communities,
create and revise local institutions, delegate and withdraw power, preserve competing
histories, and respond to consequences across time. Allow cooperation, disagreement,
separation, inactivity, and institutional failure as possible outcomes.

The operator supplies the initial conditions, available operations, and compute limits.
Participants choose what to organize around and which available arrangements to use.
A society need not produce a constitution, elect a leader, maximize agreement, or
converge on one community to count as an interesting run.

Keep this an additional experimental mode. The existing directed tasks and strange
hierarchies remain useful comparison conditions.

## What the repository already provides

The useful foundation is substantial: validated actions, actual membership and private
group posts, a shared artifact workspace, private notes and whispers, multiple speaker
schedulers, bounded inference, checkpoints, and inspectable run artifacts.

The main limits on this design goal are structural:

| Current behavior | Consequence | Relevant implementation |
| --- | --- | --- |
| Society authorities come from configuration and have no amendment action. | Participants cannot turn a spoken transfer of authority into a change of permissions. | [society.py](society.py), `Society.__init__` and `apply` |
| JSON can seed and freeze groups and configure open or invitation-only admission; dissolution, splits, and mergers remain proposed. | Initial conditions can be controlled while lifecycle changes still need more operations. | [society.py](society.py), `Society.apply` |
| `VOTE` stores a string against a board item. | Votes have no automatic quorum, closing time, ratification, or institutional effect. | [chamber.py](chamber.py), `Chamber._apply_action` |
| `WORLD` appends Markdown through `ARTIFACT_WRITE`. | A claim about a changed map does not execute a change in a simulated environment. | [protocol.py](protocol.py), `_command`; [artifacts.py](artifacts.py) |
| Society instructions replace the usual capability instructions. | Available tools such as whispers, notes, votes, and speaker nominations become harder to discover. | [chamber.py](chamber.py), `Chamber._protocol_instructions` |
| Directed speech routes, whisper routes, and channel permissions can now be configured; groups show six recent posts. | Local communication experiments are possible; selective retrieval and cultural memory remain proposed. | [communication.py](communication.py); [chamber.py](chamber.py), `Chamber.build_context` |
| Artifact previews now show a bounded beginning and recent tail; agents can request a 1,500-character window by offset. | Recent ledger changes remain visible, and omitted middle sections are accessible by explicit read. Relevance-based recall is still proposed. | [chamber.py](chamber.py), `Chamber.build_context` |
| Older checkpoints omit several kinds of continuing state; new saves preserve complete current state. | New runs can continue, while older saves cannot reconstruct missing history safely. | [chamber.py](chamber.py), `resume_from_checkpoint`, `_save_turn_checkpoint`, and CLI validation |

Two saved observations help prioritize the work:

- [The corrected laundromat run](runs/midnight_laundromat_corrected_20261104/metrics.md)
  records 10 accepted and 15 rejected non-speech actions: a 60% rejection rate.
  Its [events](runs/midnight_laundromat_corrected_20261104/events.md) show repeated
  failures involving spaces, group length limits, and nonexistent groups. The society
  prompt lists command forms without explaining the group-name grammar.
- In [Committee of After](runs/committee_of_after_20261104/events.md), CAKE's attempt
  to join `evidence-inspection` is rejected, while the
  [world ledger](runs/committee_of_after_20261104/artifacts/world.md) says CAKE joined.
  In the [transcript](runs/committee_of_after_20261104/transcript.md), CHAIR offers
  SKEPTIC the gavel, but no action can transfer the configured authority.

These are examples of interface friction and gaps between speech and state. They do
not establish a general claim about the models' ability to organize societies.

## Additions I would make

### 1. Make social actions usable and discoverable

Expose a short, capability-aware action guide in society mode, including the exact
identifier grammar, limits, examples, and how to inspect current state. Keep detailed
help retrievable so it does not consume every prompt. Include notes, whispers, votes,
and nominations when available to that participant.

Separate a group's human-readable title from a stable harness identifier. Return the
identifier and state revision in action receipts. References must resolve explicitly;
avoid guessing which existing group a malformed name means. A failed action should
return its reason and a valid example without silently creating membership.

Expose remaining action capacity. Creating a group and inviting several participants
currently competes with posting and writing within four actions per response. A
bounded recipient-list invitation could reduce this friction while still requiring
each recipient to accept.

### 2. Let communities choose their structure

Support open, request-based, and invitation-based admission; overlapping membership;
and a group charter with a version and an optional stated purpose. Group creators
should be able to create a group with no privileged office.

Add voluntary splitting, reciprocal mergers, and dissolution. A split creates a new
group with a recorded lineage; it does not move anyone automatically. A merger needs
authorization from both communities. Private archives retain their access rules when
groups change. Closing a group releases live group capacity while keeping its record
within an explicit archive limit.

Use an initial `authorities: []` condition for the new free-organization preset.
Existing compulsion remains available as a separate configured treatment. A freely
organizing baseline should let participants leave groups without permission.

### 3. Make authority local, transferable, and revocable

Represent an office as a scoped grant: holder, group, permitted operations, granting
decision, start, and optional expiry. A group can appoint a coordinator, rotate that
role, revoke it, or operate without it. Membership alone need not confer an office.

Use typed proposals and explicit resolution rules to change grants and group policies.
Keep proposals, ballots, ratified rules, and applied effects as distinct records.
An incumbent must not acquire an implicit veto over a valid recall by being the only
participant allowed to execute it.

Allow several configured decision procedures over time, including consent and delegated
decisions. Start with one well-defined procedure to make the first implementation
testable. Scope decisions to the community and resources the decision can govern.
Operator compute limits remain outside participants' amendment powers.

### 4. Give choices observable consequences

Add an optional small environment with places, objects, access, projects, and a few
resources. Typed actions have preconditions and recorded effects: transferring an
object changes its holder; completing a project changes access; using a shared
resource changes its availability. Conditions can create reasons to organize without
assigning everyone the same goal or requiring an economy.

Keep narrated claims, observations, commitments, and executed changes distinct.
Participants can dispute a report without rewriting the underlying event. For creative
worldbuilding, support proposals for new world facts separately from deterministic
physical operations. Record who accepted a new fact and its scope.

Offer two lenses: a narrative society with enforceable social operations, and an
optional society embedded in a causal environment. A language-model narrator should
not silently decide whether institutions succeeded. The changing-map scenario needs
an explicit map transition rule before map changes can be evaluated mechanically.

### 5. Let purposes, relationships, and memory develop

Add participant-controlled goals and chosen roles, separate from initial identity.
Participants can revise or abandon a project, state commitments, and record why a
relationship changed. A shared commitment needs acknowledgment from the other party;
one participant cannot create another's obligation by announcing it.

Extend private notes into bounded memories with source event IDs, observation time,
visibility, and an indication of whether an entry is a belief, report, or confirmed
action. Preserve conflicting accounts. Let participants record whom they trust for
which activities; avoid imposing one global reputation score.

Build context from current applicable rules, action receipts, recent local events,
and retrievable history. Include recent ledger changes as well as its introduction.
Any summary is an attributed interpretation with links back to its sources. Private
information must stay private through retrieval, summaries, and inheritance.

The bounded ledger preview and offset reads are a first implemented step. Dynamic
context should follow the participant's current activity. A participant
negotiating admission needs the group's current policy and pending invitation; one
investigating a broken promise needs the promise and relevant later events. Give
participants an explicit way to recall accessible history and choose their own
focus, so relevance is not determined entirely by the operator's chosen topic.

Use a bounded context allocation with these priorities:

1. Identity, enabled operations, and current rules applicable to the intended action.
2. Action results, direct requests, pending decisions, and changes since the last turn.
3. Recent conversation in the participant's current venues.
4. Relevant private memories, commitments, and earlier accessible events.
5. Optional summaries and background material within the remaining token budget.

Filter by visibility before selecting or summarizing. Record included and omitted
event IDs, source revisions, token costs, and why material was selected. A current
state view should identify the effective revision; historical records retain the
revision that applied when they occurred. If essential context exceeds the budget,
report the deficit instead of presenting a partial rule as complete.

Start with explicit focus, recency, referenced IDs, and simple text retrieval.
Compare that with the existing context assembly under matched prompt budgets.
Check whether agents remember commitments and apply current rules more reliably,
and inspect whether selective retrieval hides dissent or repeatedly reinforces one
interpretation. More saved history alone does not solve context selection.

### 6. Allow local agendas and attention

Introduce an optional scheduler where participants request a turn, pass, nominate a
peer, or work on a group agenda. A group can select an internal delegate, with its
choice recorded separately from actual scheduling.

Retain a bounded opportunity for every active participant to respond. Record when
the harness overrides a request for fairness or budget reasons. Make public assembly
one configurable communication venue; experiments can vary whether local activity
is broadcast, noticed, or visible only to involved participants.

Keep social attention separate from provider throughput. A faster endpoint should
not gain political power accidentally through wall-clock scheduling.

### 7. Make history survive interruptions and generations

New saves now restore artifact contents, versions and read markers; private notes;
speaking styles; active/finished status; groups and authority; channels and whispers;
claims and votes; scheduler queues, pending nominations and next speaker; turn
counters; RNG state; event history; and cumulative usage accounting. Continuations
write into a new directory and record their parent checkpoint's hash.

As chosen roles, goals, institutions, and causal environments are implemented, add
their state to the checkpoint schema and continuation tests at the same time.

Bind a checkpoint to its committed event position and artifact hashes. Existing event
replay reads recorded events; reconstructing state will require complete transition
payloads and versioned reducers. Do not describe it as state reconstruction yet.

Once continuation works, introduce bounded newcomers and successors at epoch
boundaries. Track identity and lineage explicitly. A successor receives only declared
public or authorized inherited material, with provenance. Cultural evolution can then
be studied through adoption, mutation, forgetting, and abandonment. Changed prompts
and inherited artifacts do not establish model-weight learning.

## First implementation slice

Complete continuation is now the implemented foundation. Next, add action usability,
dynamic context selection, and one minimal institutional mechanism. This makes later
ecology and cultural experiments interpretable.

Suggested ownership boundaries:

- `society.py`: membership, admission, channel visibility, group lifecycle.
- A proposed `institutions.py`: typed proposals, ballots, grants, and deterministic
  policy transitions. Reuse existing vote syntax where practical.
- `chamber.py` and `protocol.py`: dispatch, action discovery, receipts, and observation
  assembly. Generate applicable instructions from enabled capabilities.
- `checkpoint.py` plus chamber snapshots: complete versioned persistence and recovery.
- Later proposed `world.py` and `memory.py`: environment transitions and retrieval.
  Avoid growing all of these responsibilities inside `Chamber._apply_action`.

For the first institutional mechanism, support changing a group's admission policy
or appointing/replacing/removing its steward. A proposal names its group, expected
policy revision, typed effect, closing epoch, and decision procedure. Its electorate
is the active group membership captured when it opens. New members do not alter
that proposal's denominator. An absent or subsequently inactive voter supplies no
ballot; the recorded electorate remains fixed.

Define an epoch as one offered turn per active participant, including an explicit
pass or recorded failed opportunity. In the initial majority procedure, only `yes`,
`no`, and `abstain` are valid; passage requires `yes` from more than half the frozen
electorate. Close after the stated response opportunities, and apply at the next
epoch boundary. Record failure on ties, insufficient support, or expiry. Apply
competing proposals in a recorded deterministic order; a stale policy revision
requires a fresh proposal and fresh support.

This initial procedure is an implementation choice for an experimental condition,
not a claim that every society should govern by majority. A steward's initial
permissions should be narrow and explicit, such as processing admission requests;
the office does not confer control of other communities or their private archives.

Acceptance checks should include:

- A supported admission change alters who can actually join.
- A supported recall removes the grant without the incumbent's cooperation.
- A failed, unauthorized, or stale proposal has no institutional effect.
- A participant can decline membership, leave, or form another group.
- A checkpoint round trip preserves semantic state and subsequent speaker choices
  under fixed mock replies, including voluntary departures and private notes.
- Membership changes, summaries, and successor creation preserve visibility rules.

## Experiments that would make the goal assessable

Start with four participants, optional projects, and no assigned government. Give
them several epochs to establish practices, then introduce one recorded change in
conditions, such as a blocked route or the departure of an officeholder. Continue
long enough to observe a response. Existing nine-turn/six-participant experiments
give most participants only one or two turns, which cannot show many revision cycles.

Budget epochs explicitly: four participants across ten epochs need at least forty
generation requests before retries. Raising only `max_turns` would encounter the
default thirty-request cap. Keep wall time, spend, context, and token ceilings explicit;
checkpoint between bounded segments on the phone.

Compare matched initial conditions with narrative-only, current membership, and
revisable-institution capabilities. Match available requests and token ceilings;
record actual usage, model versions, templates, roster order, seeds, failures, and
fallbacks. These controls can test effects of available mechanisms, but cannot make
stochastic live model outputs deterministic. Do not instruct agents to perform the
institutional behaviors later counted as spontaneous discoveries.

Retain trajectories and examples alongside measures:

- Which purposes and practices participants introduced themselves.
- Formation, overlapping membership, movement, splits, mergers, and dissolutions.
- Grants issued, exercised, expired, transferred, or successfully recalled.
- Rules proposed, adopted, revised, ignored, and abandoned, with their actual effects.
- Whether practices persist, adapt, or fail after a recorded disruption or succession.
- Exclusion, denied exits, participation imbalance, unresolved disputes, and missing
  opportunities to respond, separated from endpoint and parser failures.
- Whether inherited practices affect later choices, compared with branches where
  access to the relevant history is removed while other conditions are held fixed.

Keep failed actions separate from social refusals, and distinguish a declared rule
from an observed practice. Group count, low repetition, agreement, and charter length
are descriptive signals; none alone demonstrates a freely evolving society.
