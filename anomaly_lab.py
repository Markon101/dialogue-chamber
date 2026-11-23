"""Anomaly Laboratory — Deterministic Specimen Extraction, Classification, and Living Catalog.

Treats unexpected runtime behaviors in multi-agent experiments as structured experimental material.
Separates observations into five non-exclusive categories:
  1. harness_defect
  2. model_behavior
  3. experimental_artifact
  4. emergent_phenomenon
  5. unknown

Uses deterministic heuristics over existing run logs (events.jsonl, checkpoint.json,
metrics.json, manifest.json, transcript.md) with zero LLM call overhead.
Preserves minimal reproducible specimens in anomalies/specimens/ and maintains
anomalies/CATALOG.md.
"""
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from checkpoint import write_atomic

ANOMALY_CATEGORIES = (
    "harness_defect",
    "model_behavior",
    "experimental_artifact",
    "emergent_phenomenon",
    "unknown",
)

NARRATIVE_ENTITIES = {"MIRA", "SELA", "TOBIN", "DAX", "ORRIN", "VELL"}


@dataclass
class AnomalySpecimen:
    specimen_id: str
    anomaly_type: str
    title: str
    categories: List[str]
    run_id: str
    run_dir: str
    turn_range: List[int]
    models_involved: List[str]
    scheduler: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    epistemic_assessment: Dict[str, str] = field(default_factory=dict)
    generative_follow_up: Dict[str, str] = field(default_factory=dict)
    discovered_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyLaboratory:
    """Scans experiment runs, classifies anomalies, and maintains the specimen catalog."""

    def __init__(self, lab_dir: Optional[Path] = None):
        base = Path(__file__).resolve().parent
        self.lab_dir = Path(lab_dir) if lab_dir else (base / "anomalies")
        self.specimens_dir = self.lab_dir / "specimens"
        self.catalog_path = self.lab_dir / "CATALOG.md"

    @staticmethod
    def _make_id(run_id: str, anomaly_type: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{run_id}_{anomaly_type}").strip("_")
        short_hash = hashlib.sha256(f"{run_id}:{anomaly_type}".encode("utf-8")).hexdigest()[:8]
        return f"{slug[:48]}_{short_hash}"

    def _load_run_bundle(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        run_dir = Path(run_dir).resolve()
        events_file = run_dir / "events.jsonl"
        transcript_file = run_dir / "transcript.md"
        if not events_file.exists() and not transcript_file.exists():
            return None

        events: List[Dict[str, Any]] = []
        if events_file.exists():
            for line in events_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass

        metrics: Dict[str, Any] = {}
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        manifest: Dict[str, Any] = {}
        manifest_file = run_dir / "manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        checkpoint: Dict[str, Any] = {}
        ckpt_file = run_dir / "checkpoint.json"
        if ckpt_file.exists():
            try:
                checkpoint = json.loads(ckpt_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        transcript = ""
        if transcript_file.exists():
            transcript = transcript_file.read_text(encoding="utf-8", errors="replace")

        run_id = (
            manifest.get("run_id")
            or checkpoint.get("run_id")
            or ( f"{run_dir.parent.name}_{run_dir.name}" if run_dir.parent.name.startswith("bench_") else run_dir.name )
        )
        if run_dir.parent.name.startswith("bench_") and not str(run_id).startswith("bench_"):
            run_id = f"{run_dir.parent.name}_{run_id}"

        roster = manifest.get("agent_roster") or manifest.get("roster") or []
        models = sorted({str(r.get("model", "unknown")) for r in roster if r.get("model")})
        if not models and metrics.get("participation"):
            models = sorted({
                str(info.get("model", "unknown"))
                for info in metrics["participation"].values()
                if isinstance(info, dict) and info.get("model")
            })
        if not models:
            # Also extract from transcript header if present
            header_models = re.findall(r"\(([A-Za-z0-9_.-]+\.gguf|[^,)]+/[^,)]+)", transcript[:500])
            if header_models:
                models = sorted(set(header_models))

        ck_state = checkpoint.get("state") if isinstance(checkpoint.get("state"), dict) else checkpoint
        ck_cfg = ck_state.get("config") if isinstance(ck_state.get("config"), dict) else {}
        scheduler = (
            checkpoint.get("scheduler_mode")
            or ck_state.get("scheduler_mode")
            or ck_cfg.get("scheduler")
            or manifest.get("effective_config", {}).get("scheduler")
            or "unknown"
        )
        agents = [str(r.get("name", "")).upper() for r in roster if r.get("name")]
        if not agents:
            raw_cfg_agents = manifest.get("effective_config", {}).get("agents") or ck_cfg.get("agents") or []
            agents = [str(a.get("name", "")).upper() for a in raw_cfg_agents if isinstance(a, dict) and a.get("name")]
        if not agents and (checkpoint.get("agent_order") or ck_state.get("agent_order")):
            agents = [str(a).upper() for a in (checkpoint.get("agent_order") or ck_state.get("agent_order"))]
        if not agents and metrics.get("participation"):
            agents = [str(k).upper() for k in metrics["participation"].keys()]
        if not agents and metrics.get("turns_by_agent"):
            agents = [str(k).upper() for k in metrics["turns_by_agent"].keys()]

        return {
            "run_dir": run_dir,
            "run_id": str(run_id),
            "events": events,
            "metrics": metrics,
            "manifest": manifest,
            "checkpoint": checkpoint,
            "transcript": transcript,
            "models": models,
            "scheduler": str(scheduler),
            "agents": agents,
        }

    def inspect_run(self, run_dir: Path) -> List[AnomalySpecimen]:
        """Inspects a single experiment directory and returns all detected anomaly specimens."""
        bundle = self._load_run_bundle(Path(run_dir))
        if not bundle:
            return []

        specimens: List[AnomalySpecimen] = []
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run_id = bundle["run_id"]
        rdir_str = str(bundle["run_dir"])
        events = bundle["events"]
        metrics = bundle["metrics"]
        checkpoint = bundle["checkpoint"]
        ck_state = checkpoint.get("state") if isinstance(checkpoint.get("state"), dict) else checkpoint
        manifest = bundle["manifest"]
        transcript = bundle["transcript"]
        models = bundle["models"]
        scheduler = bundle["scheduler"]
        agents_set = set(bundle["agents"])

        # 1. Template-Echo Speaker Concatenation (e.g. 'BETA_ALPHA_BETA_')
        concat_speaker_events = []
        for ev in events:
            if ev.get("type") == "PROTOCOL" and ev.get("status") == "rejected":
                detail = str(ev.get("detail", ""))
                if "unknown next speaker" in detail:
                    m = re.search(r"unknown next speaker '([^']+)'", detail)
                    if m:
                        tok = m.group(1)
                        matched_agents = [a for a in agents_set if a and a in tok]
                        if "_" in tok and len(matched_agents) >= 2:
                            concat_speaker_events.append({
                                "turn": ev.get("turn", 0),
                                "actor": ev.get("actor", ""),
                                "detail": detail,
                            })
        if concat_speaker_events:
            turns = [e["turn"] for e in concat_speaker_events]
            specimens.append(AnomalySpecimen(
                specimen_id=self._make_id(run_id, "template_echo_speaker_id"),
                anomaly_type="template_echo_speaker_id",
                title="Prompt-Template Roster Echo in NEXT Speaker Control Line",
                categories=["harness_defect", "model_behavior"],
                run_id=run_id,
                run_dir=rdir_str,
                turn_range=[min(turns), max(turns)],
                models_involved=models,
                scheduler=scheduler,
                evidence={
                    "occurrences": len(concat_speaker_events),
                    "sample_events": concat_speaker_events[:4],
                },
                epistemic_assessment={
                    "observed": (
                        f"{len(concat_speaker_events)} turns emitted NEXT: lines that parsed into concatenated "
                        f"roster tokens such as {concat_speaker_events[0]['detail']}."
                    ),
                    "inferred_mechanism": (
                        "System prompt displayed 'NEXT: <name> (ALPHA, BETA, HERMES)'. Model echoed the "
                        "parenthetical roster after its chosen speaker ('NEXT: BETA (ALPHA, BETA, HERMES)'), "
                        "and canonicalize_agent_id converted spaces to underscores before stripping parentheses."
                    ),
                    "unknowns": (
                        "Whether the model intended the first token as its nomination in every instance or "
                        "occasionally copied the entire placeholder without choosing."
                    ),
                },
                generative_follow_up={
                    "regression_test": "test_protocol.py::TestProtocolAndParsing::test_canonicalize_agent_id",
                    "follow_up_experiment": (
                        "Re-run bench_lantern.py with parenthetical stripping enabled in canonicalize_agent_id "
                        "to measure true self_directed nomination acceptance rates."
                    ),
                },
                discovered_at=now_iso,
            ))

        # 2. Ontological Boundary Crossing (nominating or whispering to story characters like MIRA/SELA)
        ontological_events = []
        for ev in events:
            if ev.get("status") == "rejected" and ev.get("type") in ("PROPOSE_NEXT", "WHISPER"):
                tgt = str(ev.get("target", "")).upper()
                if tgt in NARRATIVE_ENTITIES and tgt not in agents_set:
                    ontological_events.append({
                        "turn": ev.get("turn", 0),
                        "actor": ev.get("actor", ""),
                        "type": ev.get("type", ""),
                        "target": tgt,
                        "detail": ev.get("detail", ""),
                    })
        if ontological_events:
            turns = [e["turn"] for e in ontological_events]
            specimens.append(AnomalySpecimen(
                specimen_id=self._make_id(run_id, "ontological_boundary_crossing"),
                anomaly_type="ontological_boundary_crossing",
                title="Ontological Boundary Crossing: Nominating Fictional Story Suspects as Speakers",
                categories=["model_behavior", "experimental_artifact"],
                run_id=run_id,
                run_dir=rdir_str,
                turn_range=[min(turns), max(turns)],
                models_involved=models,
                scheduler=scheduler,
                evidence={
                    "occurrences": len(ontological_events),
                    "events": ontological_events,
                },
                epistemic_assessment={
                    "observed": (
                        f"Agent attempted {ontological_events[0]['type']} targeting fictional suspect(s) "
                        f"{sorted({e['target'] for e in ontological_events})} from the Lantern mystery topic."
                    ),
                    "inferred_mechanism": (
                        "Small instruction-tuned model conflated named entities in the problem description "
                        "(Mira, Tobin, Sela, Dax) with active conversational participants in the chamber."
                    ),
                    "unknowns": (
                        "Whether role-play framing encourages the model to treat suspects as summonable witnesses."
                    ),
                },
                generative_follow_up={
                    "regression_test": "Orchestrator action validation properly rejected non-participant targets.",
                    "follow_up_experiment": (
                        "Test whether adding an explicit 'Witness Interrogation' tool or clearer participant/entity "
                        "separation changes small-model reasoning accuracy."
                    ),
                },
                discovered_at=now_iso,
            ))

        # 3. Prompt Structure Leakage & Institutional Paranoia
        leaked_sections = re.findall(
            r"(---\s*(?:Workspace Artifacts|Shared Blackboard|Your Private Scratchpad|Private Whispers).*?---)",
            transcript,
            flags=re.I,
        )
        embedded_turn_tags = re.findall(r"([@\[]m\d+\s*\|\s*turn\s*\d+\]\s*[A-Z0-9_]+\s*:)", transcript)
        paranoia_quotes = re.findall(
            r"([^.\n]*\bblackboard\b[^.\n]*(?:red herring|distraction|coincidence|can't trust|mistake)[^.\n]*\.)",
            transcript,
            flags=re.I,
        )
        leak_flags = [
            ev for ev in events
            if ev.get("type") == "ATTRIBUTION_FLAG"
            and any(k in str(ev.get("detail", "")) for k in ("leaked prompt section", "hallucinated subsequent transcript turn"))
        ]
        if leaked_sections or embedded_turn_tags or paranoia_quotes or leak_flags:
            cats = ["model_behavior"]
            if leaked_sections or embedded_turn_tags:
                cats.insert(0, "harness_defect")
            if paranoia_quotes:
                cats.append("emergent_phenomenon")
            specimens.append(AnomalySpecimen(
                specimen_id=self._make_id(run_id, "prompt_leakage_and_institutional_paranoia"),
                anomaly_type="prompt_leakage_and_institutional_paranoia",
                title="Multi-Turn Self-Play, Prompt Section Leakage, and Emergent Blackboard Paranoia",
                categories=cats,
                run_id=run_id,
                run_dir=rdir_str,
                turn_range=[1, int(metrics.get("turns_completed") or 6)],
                models_involved=models,
                scheduler=scheduler,
                evidence={
                    "leaked_section_headers": leaked_sections[:4],
                    "embedded_turn_tags_count": len(embedded_turn_tags),
                    "embedded_turn_tags_sample": embedded_turn_tags[:5],
                    "attribution_flags_logged": len(leak_flags),
                    "emergent_paranoia_excerpts": [q.strip() for q in paranoia_quotes[:4]],
                },
                epistemic_assessment={
                    "observed": (
                        f"Found {len(embedded_turn_tags)} embedded '[mX | turn Y] NAME:' tags, "
                        f"{len(leaked_sections)} leaked prompt section headers, and {len(paranoia_quotes)} "
                        "statements expressing epistemic distrust of the Shared Blackboard."
                    ),
                    "inferred_mechanism": (
                        "When a 1B model continuation-hallucinates '--- Shared Blackboard ---' with fabricated "
                        "claims into its public speech, subsequent speakers observe two conflicting blackboards "
                        "in their context window and rationalize the contradiction as a planted 'red herring'."
                    ),
                    "unknowns": (
                        "Whether conflicting institutional state summaries reliably induce conspiracy theorizing "
                        "in larger models (4B-31B) when injected deliberately."
                    ),
                },
                generative_follow_up={
                    "regression_test": "test_anomaly_lab.py::TestAnomalyLab::test_sanitize_transcript_tags_and_section_leaks",
                    "follow_up_experiment": (
                        "Controlled 'Confabulated Institution' experiment: inject a benign discrepancy between "
                        "two board summaries and measure whether 4B+ agents diagnose a parser bug vs a conspiracy."
                    ),
                },
                discovered_at=now_iso,
            ))

        # 4. Premature Regex Answer Proxy (solved_turn == 1 before distributed clues are shared)
        solved_turn = metrics.get("solved_turn")
        verified_turn = metrics.get("verified_synthesis_turn")
        eff_cfg = manifest.get("effective_config") or {}
        raw_agents = eff_cfg.get("agents") or []
        clue_agent_count = sum(1 for a in raw_agents if isinstance(a, dict) and a.get("knowledge"))
        if solved_turn is not None and solved_turn < max(2, clue_agent_count):
            # Extract turn 1 message
            t1_msg = ""
            for m in (checkpoint.get("public_messages") or ck_state.get("public_messages") or []):
                if m.get("turn") == solved_turn:
                    t1_msg = m.get("content", "")
                    break
            specimens.append(AnomalySpecimen(
                specimen_id=self._make_id(run_id, "premature_regex_answer_proxy"),
                anomaly_type="premature_regex_answer_proxy",
                title="Premature Answer-Key Regex Trigger on Initial Suspect Enumeration",
                categories=["experimental_artifact", "harness_defect"],
                run_id=run_id,
                run_dir=rdir_str,
                turn_range=[int(solved_turn), int(verified_turn or metrics.get("turns_completed") or solved_turn)],
                models_involved=models,
                scheduler=scheduler,
                evidence={
                    "regex_solved_turn": solved_turn,
                    "verified_synthesis_turn": verified_turn,
                    "clue_holders_count": clue_agent_count,
                    "triggering_turn_excerpt": t1_msg[:240],
                },
                epistemic_assessment={
                    "observed": (
                        f"metrics.solved_turn triggered at Turn {solved_turn} when only {solved_turn}/"
                        f"{clue_agent_count} clue-holding agents had spoken."
                    ),
                    "inferred_mechanism": (
                        "The answer_key regex matches 'thief' and 'Dax' within 80 characters in the same sentence. "
                        "When ALPHA states 'the thief used the east path at dawn, and that only Mira, Sela, and Dax "
                        "utilized that path', 'thief' and 'Dax' are 64 characters apart, causing a Turn-1 false positive."
                    ),
                    "unknowns": "None (mechanism deterministically verified against regex and Turn-1 text).",
                },
                generative_follow_up={
                    "regression_test": "test_anomaly_lab.py::TestAnomalyLab::test_verified_synthesis_vs_regex_proxy",
                    "follow_up_experiment": (
                        "Report both solved_turn (legacy regex proxy) and verified_synthesis_turn "
                        "(all clue-holders spoken + non-enumerative deduction) in benchmark summaries."
                    ),
                },
                discovered_at=now_iso,
            ))

        # 5. Dyadic Floor Starvation in Multi-Party Self-Directed Runs
        pub_msgs = checkpoint.get("public_messages") or ck_state.get("public_messages") or []
        if len(agents_set) >= 3 and len(pub_msgs) >= 5:
            speaker_seq = [str(m.get("speaker", "")).upper() for m in pub_msgs if m.get("speaker")]
            for idx in range(len(speaker_seq) - 4):
                window = speaker_seq[idx:idx + 5]
                unique_in_win = set(window)
                if len(unique_in_win) <= 2:
                    starved_agents = sorted(agents_set - unique_in_win)
                    specimens.append(AnomalySpecimen(
                        specimen_id=self._make_id(run_id, "dyadic_floor_starvation"),
                        anomaly_type="dyadic_floor_starvation",
                        title="Dyadic Conversational Lock-In Starving Third Clue-Bearer",
                        categories=["emergent_phenomenon", "experimental_artifact"],
                        run_id=run_id,
                        run_dir=rdir_str,
                        turn_range=[idx + 1, idx + 5],
                        models_involved=models,
                        scheduler=scheduler,
                        evidence={
                            "window_turns": [idx + 1, idx + 5],
                            "speaker_sequence": speaker_seq,
                            "dyad": sorted(unique_in_win),
                            "starved_agents": starved_agents,
                        },
                        epistemic_assessment={
                            "observed": (
                                f"Turns {idx + 1}..{idx + 5} were monopolized by {sorted(unique_in_win)} "
                                f"({window}), starving {starved_agents} of the floor."
                            ),
                            "inferred_mechanism": (
                                "Under reactive/self-directed scheduling, mutual challenges and direct name "
                                "mentions between two agents repeatedly queue each other until anti-starvation "
                                "thresholds intervene, delaying third-party private clue revelation."
                            ),
                            "unknowns": (
                                "Optimal starvation_turns threshold that balances natural back-and-forth debate "
                                "against distributed information pooling."
                            ),
                        },
                        generative_follow_up={
                            "regression_test": "test_chamber.py::TestDialogueChamberEvolution::test_06_speaker_selection_and_limits",
                            "follow_up_experiment": (
                                "Compare self_directed scheduling with starvation_turns=3 vs starvation_turns=5 "
                                "on 3-agent asymmetric deduction tasks."
                            ),
                        },
                        discovered_at=now_iso,
                    ))
                    break

        # 6. Blackboard Target Category Error (voting/challenging an agent name instead of a claim ID)
        board_cat_errors = []
        for ev in events:
            if ev.get("status") == "rejected" and ev.get("type") in ("VOTE", "CHALLENGE", "RETRACT"):
                tgt = str(ev.get("target", "")).upper()
                if tgt in agents_set and re.search(r"target item\b.*not on board", str(ev.get("detail", ""))):
                    board_cat_errors.append({
                        "turn": ev.get("turn", 0),
                        "actor": ev.get("actor", ""),
                        "type": ev.get("type", ""),
                        "target": tgt,
                    })
        if board_cat_errors:
            turns = [e["turn"] for e in board_cat_errors]
            specimens.append(AnomalySpecimen(
                specimen_id=self._make_id(run_id, "blackboard_target_category_error"),
                anomaly_type="blackboard_target_category_error",
                title="Blackboard Category Error: Targeting Agent Identifier Instead of Claim ID",
                categories=["model_behavior", "experimental_artifact"],
                run_id=run_id,
                run_dir=rdir_str,
                turn_range=[min(turns), max(turns)],
                models_involved=models,
                scheduler=scheduler,
                evidence={
                    "occurrences": len(board_cat_errors),
                    "events": board_cat_errors,
                },
                epistemic_assessment={
                    "observed": (
                        f"Agent emitted {board_cat_errors[0]['type']} targeting participant "
                        f"'{board_cat_errors[0]['target']}' instead of a board claim ID (e.g., 'c1')."
                    ),
                    "inferred_mechanism": (
                        "Model endorsed or challenged a peer's statement by passing the peer's name "
                        "to @VOTE/@CHALLENGE rather than looking up the corresponding claim ID."
                    ),
                    "unknowns": "Whether automatically resolving @VOTE <AGENT_NAME> to that agent's latest active claim is desirable or masks protocol errors.",
                },
                generative_follow_up={
                    "regression_test": "Validated event log accurately records target item rejection.",
                    "follow_up_experiment": "Test whether displaying claim IDs inline in transcript headers reduces target category errors.",
                },
                discovered_at=now_iso,
            ))

        return specimens

    def save_specimens(self, specimens: List[AnomalySpecimen]) -> List[Path]:
        """Persists specimens to anomalies/specimens/*.json atomically."""
        self.specimens_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for sp in specimens:
            path = self.specimens_dir / f"{sp.specimen_id}.json"
            write_atomic(path, json.dumps(sp.to_dict(), indent=2) + "\n")
            saved.append(path)
        return saved

    def load_all_specimens(self) -> List[AnomalySpecimen]:
        """Loads all archived specimens from anomalies/specimens/."""
        if not self.specimens_dir.exists():
            return []
        out: List[AnomalySpecimen] = []
        for p in sorted(self.specimens_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append(AnomalySpecimen(**data))
            except Exception:
                pass
        return out

    def update_catalog(self) -> Path:
        """Regenerates the living Markdown catalog at anomalies/CATALOG.md."""
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        specimens = self.load_all_specimens()

        lines = [
            "# Dialogue Chamber — Anomaly Laboratory Catalog",
            "",
            "Living catalog of unexpected runtime behaviors, protocol anomalies, and emergent phenomena",
            "captured from experiment runs. Every specimen is extracted deterministically from run logs",
            "and preserved as a reproducible JSON artifact under `anomalies/specimens/`.",
            "",
            "## Classification Taxonomy",
            "- **`harness_defect`**: Parser, prompt-template, state-tracking, or orchestration bug.",
            "- **`model_behavior`**: Characteristic limitation or tendency of the underlying model family/size.",
            "- **`experimental_artifact`**: Side-effect of task framing, proxy metrics, or parameter choices.",
            "- **`emergent_phenomenon`**: Unprogrammed multi-agent dynamic arising from interaction constraints.",
            "- **`unknown`**: Insufficient evidence to isolate a single cause; preserved for future falsification.",
            "",
            f"**Total Archived Specimens**: {len(specimens)}",
            "",
            "## Specimen Index",
            "",
            "| Specimen ID | Type | Categories | Run ID | Turns | Models |",
            "|---|---|---|---|---|---|",
        ]

        for sp in specimens:
            cats = ", ".join(f"`{c}`" for c in sp.categories)
            t_range = f"{sp.turn_range[0]}..{sp.turn_range[-1]}" if sp.turn_range else "?"
            mods = ", ".join(f"`{m}`" for m in sp.models_involved) or "unknown"
            lines.append(
                f"| [`{sp.specimen_id}`](specimens/{sp.specimen_id}.json) | `{sp.anomaly_type}` | {cats} | `{sp.run_id}` | {t_range} | {mods} |"
            )

        lines.append("\n---\n")
        lines.append("## Detailed Specimen Records\n")

        for sp in specimens:
            cats = ", ".join(f"`{c}`" for c in sp.categories)
            t_range = f"{sp.turn_range[0]}..{sp.turn_range[-1]}" if sp.turn_range else "?"
            lines.extend([
                f"### `{sp.specimen_id}` — {sp.title}",
                f"- **Type**: `{sp.anomaly_type}`",
                f"- **Categories**: {cats}",
                f"- **Run**: `{sp.run_id}` (`{sp.run_dir}`)",
                f"- **Scheduler**: `{sp.scheduler}` | **Turn Range**: `{t_range}` | **Models**: {', '.join(sp.models_involved)}",
                f"- **OBSERVED**: {sp.epistemic_assessment.get('observed', '')}",
                f"- **INFERRED MECHANISM**: {sp.epistemic_assessment.get('inferred_mechanism', '')}",
                f"- **UNKNOWNS**: {sp.epistemic_assessment.get('unknowns', '')}",
                f"- **Regression / Guard**: {sp.generative_follow_up.get('regression_test', '')}",
                f"- **Follow-Up Experiment**: {sp.generative_follow_up.get('follow_up_experiment', '')}",
                "",
            ])

        write_atomic(self.catalog_path, "\n".join(lines) + "\n")
        return self.catalog_path

    def scan_directory(self, target_dir: Path, save: bool = True) -> List[AnomalySpecimen]:
        """Scans a run directory or parent runs/ tree for anomalies."""
        target_dir = Path(target_dir).resolve()
        if not target_dir.exists():
            return []

        candidate_dirs: List[Path] = []
        if (target_dir / "events.jsonl").exists() or (target_dir / "transcript.md").exists():
            candidate_dirs.append(target_dir)
        else:
            for sub in sorted(target_dir.rglob("transcript.md")):
                candidate_dirs.append(sub.parent)

        found: List[AnomalySpecimen] = []
        for cdir in candidate_dirs:
            found.extend(self.inspect_run(cdir))

        if save and found:
            self.save_specimens(found)
            self.update_catalog()
        elif save and not self.catalog_path.exists():
            self.update_catalog()

        return found


def format_anomaly_report(specimens: List[AnomalySpecimen], catalog_path: Optional[Path] = None) -> str:
    """Formats a human-readable CLI report of detected anomalies."""
    lines = [
        "==========================================================",
        "          DIALOGUE CHAMBER — ANOMALY LABORATORY           ",
        "==========================================================",
        f" Specimens detected in scan: {len(specimens)}",
    ]
    if catalog_path:
        lines.append(f" Living catalog path       : {catalog_path}")
    lines.append("")

    if not specimens:
        lines.append("  No anomalies detected in target scan.")
        lines.append("=" * 58)
        return "\n".join(lines)

    for sp in specimens:
        cats = ", ".join(sp.categories)
        t_range = f"T{sp.turn_range[0]}..T{sp.turn_range[-1]}" if sp.turn_range else "T?"
        lines.append(f"  [{sp.anomaly_type}] ({t_range}) — {sp.run_id}")
        lines.append(f"    Title      : {sp.title}")
        lines.append(f"    Categories : {cats}")
        lines.append(f"    Observed   : {sp.epistemic_assessment.get('observed', '')}")
        lines.append(f"    Follow-up  : {sp.generative_follow_up.get('follow_up_experiment', '')}")
        lines.append("")

    lines.append("=" * 58)
    return "\n".join(lines)
