# Emergent Ocean Research Charter — Research Backlog

Platform: Dialogue Chamber / Emergent Ocean Multi-Agent Research Platform  
Environment: Samsung Galaxy S25 Ultra, Android Termux, local `llama.cpp` Snapdragon HTP0, OpenRouter inference.

Related design proposal: [Freely organizing and evolving societies](SOCIETY_EVOLUTION_DESIGN.md).
It connects group lifecycle, revisable authority, causal world state, memory, and
continuation into a staged implementation and evaluation plan.

> **Evaluation Metrics**:
> - **Research Value (RV)**: 1 (Incremental) to 5 (Transformative)
> - **Feasibility (Feas)**: 1 (Speculative) to 5 (Directly implementable today)
> - **Compute Requirement (Comp)**: Low / Med / High
> - **Observability (Obs)**: 1 (Heuristic text proxy) to 5 (Objective formal ground truth)
> - **Implementation Complexity (Cplx)**: Low / Med / High

---

## Category 1: 10 Conservative Extensions
*Direct extensions of existing harness primitives (state, actions, scheduling, memory).*

| # | Proposal | RV | Feas | Comp | Obs | Cplx | Description |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| C1 | **Multi-Round Voting & Quorum Engine** | 4 | 5 | Low | 5 | Low | Add formal quorum thresholds (e.g. 2/3 majority, unanimous consent) and automated resolution transitions for `@VOTE` actions. |
| C2 | **Hierarchical Document Locking** | 4 | 5 | Low | 5 | Low | Section-level optimistic locks on shared Markdown artifacts (e.g. `mandate.md#Section-2`) to test division of labor without merge collisions. |
| C3 | **Agent Reputation & Credit Ledger** | 4 | 5 | Low | 5 | Low | Maintain internal scoreboard tracking accepted claims, refuted challenges, and accurate predictions across turns. |
| C4 | **Episodic Memory Summarizer Hook** | 3 | 5 | Low | 4 | Med | When transcript pressure triggers message trimming, automatically summarize evicted dialogue into a persistent `digest.md` artifact. |
| C5 | **Dynamic Temperature Schedules** | 3 | 5 | Low | 4 | Low | Automated annealing: decrease agent temperature systematically as a debate approaches turn limits to measure convergence. |
| C6 | **Claim Dependency Graphs** | 4 | 4 | Low | 5 | Med | Structured claim relations (`c2 depends_on c1`, `c3 refutes c2`); automatically retract downstream claims if an axiom is withdrawn. |
| C7 | **Whisper Leakage Rate Parameter** | 4 | 5 | Low | 5 | Low | Configurable probability ($P_{\text{leak}}$) that a private whisper is broadcast as a partial transcript rumor, measuring coalition resilience. |
| C8 | **Token-Cost Allocation Ceilings per Agent** | 3 | 5 | Low | 5 | Low | Hard token/spend quotas per agent persona, forcing concise communication in bandwidth-constrained roles. |
| C9 | **Multi-Channel Private Subgroups** | 3 | 5 | Low | 4 | Low | Dynamic creation of invite-only communication channels through validated `@CREATE_CHANNEL` actions. |
| C10 | **Automated Jaccard Convergence Halting** | 3 | 5 | Low | 4 | Low | Early-stopping trigger when consecutive response trigram overlap exceeds a configurable threshold, preventing runaway echoing. |

---

## Category 2: 10 Ambitious But Technically Feasible Experiments
*Experiments investigating collective intelligence, prediction, institution building, and coordination.*

| # | Proposal | RV | Feas | Comp | Obs | Cplx | Description |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| A1 | **The Synthetic Constitution Convention** | 5 | 4 | Med | 4 | Med | 5 agents (mixed local/remote) must draft, debate, amend, and ratify a 5-article governing charter; evaluate institutional stability under persona swaps. |
| A2 | **Double-Blind Market Prediction Game** | 5 | 4 | Med | 5 | Med | Agents trade simulated shares on binary propositions before external evidence is revealed; measure prediction calibration vs individual agent confidence. |
| A3 | **Emergent Protocol Invention Under Bandwidth Bottlenecks** | 5 | 4 | Med | 5 | Med | Restrict public speech to 16 characters or 4 tokens; observe whether agents invent compression symbols or shorthand coordinate codes. |
| A4 | **Adversarial Institutional Subversion** | 5 | 4 | Med | 4 | Med | Introduce one covert defector agent incentivized to introduce subtle contradictions into `conclusions.md`; measure detection latency by peers. |
| A5 | **Asymmetric Information Murder Mystery (10-clue network)** | 4 | 5 | Med | 5 | Low | 4 agents each possess disjoint clues and red herrings; test whether round-robin vs self-directed scheduling discovers Dax faster. |
| A6 | **Counterfactual Branching Explorer** | 5 | 3 | High | 4 | High | At turn $T$, checkpoint state and fork 5 parallel branches with different speaker nominations to measure trajectory sensitivity and path dependency. |
| A7 | **Model Heterogeneity Synergy Baseline** | 5 | 4 | Med | 5 | Med | Compare collective problem-solving across 3 cohorts: (1) 4x Gemma local, (2) 4x DeepSeek remote, (3) 2x Gemma + 2x DeepSeek on identical mathematical puzzles. |
| A8 | **Iterative Hypothesis Refinement Matrix** | 4 | 4 | Med | 5 | Med | Structured scientific discovery cycle: Hypothesis Formulation $\rightarrow$ Falsification Challenge $\rightarrow$ Experimental Artifact Update. |
| A9 | **Language Drift in Isolated Micro-Societies** | 4 | 4 | High | 4 | Med | 100-turn simulation of high-temperature local models; evaluate semantic drift, vocabulary contraction, and novel idiom emergence via trigram embeddings. |
| A10 | **Automated Multi-Agent Code Review & Debugging Guild** | 4 | 4 | Med | 5 | Med | One drafter, two adversarial critics, one formal verifier repairing deterministic Python functions inside artifacts; evaluate test pass rates. |

---

## Category 3: 10 Unconventional Cross-Disciplinary Ideas
*Bridging cellular automata, information theory, linguistics, game theory, and algorithmic art.*

| # | Proposal | RV | Feas | Comp | Obs | Cplx | Description |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| U1 | **Cellular Automata Linguistic Lattice** | 4 | 3 | Med | 4 | Med | Arrange agents on a 2D grid where agents can only whisper to Moore-neighborhood neighbors; observe wave-propagation of concepts. |
| U2 | **Kolmogorov Complexity Dialect Compression** | 4 | 4 | Low | 5 | Low | Measure gzip / zlib compression ratios of the transcript across turns as a proxy for shared mutual information and collective convergence. |
| U3 | **The Byzantine Seance: Checkpoint Necromancy** | 3 | 5 | Low | 4 | Low | Persona role-play where agents deliberate on the reconstructed weights of a deleted model, using cryptographic hashes as "incantations". |
| U4 | **Sonnet-Form Imperial Jurisprudence** | 3 | 5 | Low | 4 | Low | The Court Poet must write legal objections in iambic pentameter; evaluate whether metric constraints alter strategic game-theoretic outcomes. |
| U5 | **Shannon Entropy Channel Throttling** | 4 | 4 | Med | 5 | Med | Measure character entropy per turn; inject noise into whispers inversely proportional to channel entropy, forcing redundancy protocols. |
| U6 | **Simulated Resource Ecology (Token Thermodynamics)** | 5 | 3 | Med | 5 | High | Agents receive finite energy tokens per turn; speaking consumes energy, successful claims earn bounties, bankruptcy causes agent retirement. |
| U7 | **Cryptographic Zero-Knowledge Social Proofs** | 4 | 3 | Med | 5 | High | Agents publish SHA-256 commitments of their hidden clues before revealing them, preventing retrospective clue fabrication. |
| U8 | **Non-Euclidean Spatial Simulation in Markdown** | 4 | 4 | Low | 4 | Med | Agents navigate and describe a hyperbolic or Klein-bottle topography represented as connected Markdown nodes in `map.md`. |
| U9 | **Memetic Mutation Tournaments** | 4 | 4 | Med | 5 | Med | Short aphorisms compete across turns; agents replicate and mutate favorite phrases; track phylogenetic trees of surviving memes. |
| U10 | **The Blind Oracle Council** | 4 | 4 | Low | 5 | Low | One agent acts as a sensor receiving raw numerical telemetry; other agents cannot see the numbers and must interrogate the sensor to deduce world state. |

---

## Category 4: 5 High-Risk / Speculative Research Ideas
*Require strict sandboxing, isolated execution, and bounded safety controls.*

| # | Proposal | RV | Feas | Comp | Obs | Cplx | Description |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| S1 | **Agent-Synthesized Domain-Specific Languages (DSL)** | 5 | 2 | Med | 4 | High | Agents define their own grammar and AST interpreter in Python; harness executes verified ASTs in an isolated sandbox. |
| S2 | **Self-Modifying System Prompts via Constitutional Amendment** | 5 | 2 | Med | 3 | High | If a supermajority ratifies an amendment, the orchestrator updates their own system prompt for subsequent turns; test for governance collapse. |
| S3 | **Autonomous Multi-Agent Tool Construction** | 5 | 2 | High | 4 | High | Agents write regex validators and text transformation scripts into artifacts; harness exposes accepted scripts as callable tools in subsequent turns. |
| S4 | **Adversarial Jailbreak Resistance Ecosystem** | 4 | 3 | Med | 4 | High | Red team agents attempt to elicit forbidden tokens or formatting violations; Blue team agents construct defensive filters in real time. |
| S5 | **Longitudinal Memory Transfer Across Generational Runs** | 5 | 2 | High | 4 | High | Checkpoint artifacts from Run $N$ are inherited as sacred historical canon by Run $N+1$ with randomized agent personas; track cultural evolution over 20 generations. |

---

## Category 5: 3 Unexpectedly Useful Experiments Achievable Today
*High return on investment, fully runnable on existing Termux Snapdragon hardware right now.*

| # | Proposal | RV | Feas | Comp | Obs | Cplx | Description |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| R1 | **The Asymmetric Lantern Benchmark (Self-Directed vs Round-Robin)** | 5 | 5 | Low | 5 | Low | Execute 10 runs of `information_asymmetry` under round-robin vs self-directed scheduling. Measure exact turns to solution (`solved_turn`) and token efficiency. |
| R2 | **KV-Cache Memory & Context Utilization Scaling** | 4 | 5 | Low | 5 | Low | Profile Snapdragon RAM usage and decoding speed (tok/s) across 4K, 8K, 16K, and 32K context sizes on Gemma 4 E4B; map practical mobile context envelopes. |
| R3 | **Free-Tier OpenRouter Heterogeneous Council** | 5 | 5 | Low | 5 | Low | Run a 4-agent Council combining local Gemma 4 (NPU), Nemotron 3.5 Free, Gemma 4 31B Free, and Inkling Small Free on a collective logic puzzle; measure cross-model consensus formation. |

---

## Category 6: 1 Minimal Experiment Capable of Falsifying a Central Assumption
*The Falsification Test for Genuine Multi-Agent Collective Advantage.*

### The Hypothesis
> **Core Assumption**: Multiple specialized LLM agents interacting through private channels, blackboards, and turn-taking produce better collective reasoning than a single LLM prompted with the identical cumulative compute and context budget.

### Falsification Experiment: The "Single-Agent Oracle vs 4-Agent Chamber" Baseline
1. **Task**: 20 distinct combinatorial mystery puzzles (isomorphic to the Lantern problem) with partitioned clues.
2. **Condition A (Chamber)**: 4 agents with private clue partitions, self-directed turn taking, whispers, and shared blackboard (`max_turns=12`).
3. **Condition B (Compute-Matched Monolith)**: A single instance of the same model given all 4 clues simultaneously, asked to reason step-by-step using a token budget exactly equal to the total completion tokens consumed in Condition A.
4. **Condition C (Ablated Chamber)**: 4 agents with identical public access to all clues from turn 1.
5. **Falsification Criterion**: If Condition B matches or exceeds Condition A's accuracy while consuming $\le 50\%$ of Condition A's prompt prefill tokens, the hypothesis that multi-agent conversational role-play adds computational value over unified reasoning is **FALSIFIED** for this task class.
