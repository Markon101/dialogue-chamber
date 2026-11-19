# Private context editing and turn awareness

The harness exposes the current turn, turn cap, remaining scheduled turns, and
end-vote eligibility in each participant's prompt. Turn slots can be consumed by
failed requests, so `turns_completed` is the count to use for contributions.

Private context editing is opt-in for each agent and each source. An agent can hide
a message, substitute a labeled working interpretation, or summarize the earlier
messages of a source while continuing to see new ones. The operator can choose the
allowed operations independently for public speech, delivered whispers, and named
channels. See the [configuration guide](README.md#let-participants-edit-their-own-conversation-context).

## Validation

- The scripted regression verifies a public replacement, whispered-message hiding,
  and channel summarization. It checks that another agent still sees the original,
  blocked routes cannot be crossed, loss of a channel subscription hides its
  summary, new messages survive summarization, and a continuation restores the
  editor's view. Source restoration makes the original visible again.
- A six-turn Free OpenRouter check used Nemotron 3.5 Lightning and Particle 1.
  The editor was given a specific replacement on turn 2 and a summary on turn 4.
  In the recorded prompt for turn 4, the editor saw its replacement while the
  archivist retained the original. On turn 6, the editor saw its summary through
  turn 3 alongside a later message. All six requests that produced turns carried
  the correct turn number. The run used eight completion attempts, including two
  fallbacks, with a reported cost of $0.00.
- The collaborative and opposing-team princess workshops each used a six-round
  minimum before end voting and a nine-round cap. The collaborative run produced
  35 replies in 36 slots and saved 20 song updates. The team run used the new turn
  cues and produced 36 replies, 15 song updates, and eight accepted autonomy
  updates. On turn 35 a participant told its teammate there was one turn left.
  Both workshops reached the cap and reported $0.00 in completion cost.

The live context check measured whether edits reached later requests under explicit
cues. How often agents elect to edit their memory unaided, and whether the edits
improve a longer discussion, remain open questions.

```sh
CHAMBER_COPY_DIR='' python3 -m unittest -v test_context_edits
CHAMBER_COPY_DIR='' python3 -m unittest discover
```
