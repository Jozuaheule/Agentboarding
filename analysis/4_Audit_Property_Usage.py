from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulation import (
    BoardingSimulation,
    CabinEnvironment,
    EDGES_FILE,
    MANIFEST_FILE,
    NODES_FILE,
    PassengerAgent,
)


RUNS_PER_POLICY = 35
POLICIES = ["std", "pyramid", "random"]
SEED_START = 5000


def _static_passenger_field_usage(repo_root: Path) -> dict[str, Any]:
    simulation_path = repo_root / "simulation.py"
    tree = ast.parse(simulation_path.read_text(encoding="utf-8"))

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PassengerAgent":
            class_node = node
            break
    if class_node is None:
        raise RuntimeError("PassengerAgent class not found")

    init_assigned = set()
    loads_anywhere = set()

    for node in ast.walk(class_node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, ast.Load):
                loads_anywhere.add(node.attr)

    for fn in class_node.body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "self"
                ):
                    init_assigned.add(node.targets[0].attr)

    never_read_in_class = sorted(init_assigned - loads_anywhere)

    # Lightweight whole-repo external usage check for each candidate.
    py_files = [p for p in repo_root.rglob("*.py") if ".venv" not in p.parts and "site-packages" not in p.parts]
    external_refs = {}
    for attr in never_read_in_class:
        token = f".{attr}"
        refs = []
        for py_file in py_files:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            count = text.count(token)
            if count > 0:
                refs.append({"path": str(py_file.relative_to(repo_root)), "count": count})
        external_refs[attr] = refs

    return {
        "init_assigned_fields": sorted(init_assigned),
        "loaded_fields_anywhere_in_class": sorted(loads_anywhere),
        "never_read_in_class": never_read_in_class,
        "external_dot_refs": external_refs,
    }


def _dynamic_coverage(repo_root: Path) -> dict[str, Any]:
    coverage: dict[str, Any] = {
        "runs_total": 0,
        "runs_by_policy": Counter(),
        "intents": Counter(),
        "actions": Counter(),
        "luggage_statuses": Counter(),
        "flag_true_counts": Counter(),
        "event_counters_total": Counter(),
        "max_tick_by_policy": defaultdict(int),
    }

    original_evaluate = PassengerAgent.evaluate_intent
    original_execute = PassengerAgent.execute_action

    def wrapped_evaluate(self: PassengerAgent, env, occupied, all_agents, agent_at, all_agent_at):
        original_evaluate(self, env, occupied, all_agents, agent_at, all_agent_at)
        coverage["intents"][self.intent] += 1
        coverage["luggage_statuses"][self.luggage_status] += 1
        if self.row_blocker:
            coverage["flag_true_counts"]["row_blocker"] += 1
        if self.row_blocked:
            coverage["flag_true_counts"]["row_blocked"] += 1
        if self.row_shift_complete:
            coverage["flag_true_counts"]["row_shift_complete"] += 1

    def wrapped_execute(self: PassengerAgent, env, occupied, agent_at, all_agents, next_positions, rng):
        action = original_execute(self, env, occupied, agent_at, all_agents, next_positions, rng)
        coverage["actions"][action] += 1
        return action

    PassengerAgent.evaluate_intent = wrapped_evaluate
    PassengerAgent.execute_action = wrapped_execute

    try:
        for p_idx, policy in enumerate(POLICIES):
            for run_idx in range(RUNS_PER_POLICY):
                seed = SEED_START + p_idx * 1000 + run_idx
                with contextlib.redirect_stdout(io.StringIO()):
                    env = CabinEnvironment(NODES_FILE, EDGES_FILE)
                    sim = BoardingSimulation(
                        env,
                        MANIFEST_FILE,
                        seed=seed,
                        boarding_policy=policy,
                        log_summary=False,
                    )
                    sim.run(verbose=False, enforce_completion=True)

                coverage["runs_total"] += 1
                coverage["runs_by_policy"][policy] += 1
                for k, v in sim.event_counters.items():
                    coverage["event_counters_total"][k] += v
                coverage["max_tick_by_policy"][policy] = max(coverage["max_tick_by_policy"][policy], sim.tick)
    finally:
        PassengerAgent.evaluate_intent = original_evaluate
        PassengerAgent.execute_action = original_execute

    all_defined_intents = {
        "none",
        "sit",
        "enterRow",
        "stow",
        "advance",
        "wait",
        "resolveSeatBlock",
        "switchAisle",
    }
    all_defined_action_prefixes = {
        "none",
        "sit",
        "startStow",
        "stowComplete",
        "wait",
        "waitBlocker",
        "startShuffle",
        "shufflingSeat",
        "finishShuffle",
        "moveTo:",
    }

    seen_intents = set(coverage["intents"].keys())
    seen_actions = set(coverage["actions"].keys())

    unseen_intents = sorted(all_defined_intents - seen_intents)

    # Action coverage with prefix handling for moveTo:
    seen_action_tokens = set()
    for action in seen_actions:
        if action.startswith("moveTo:"):
            seen_action_tokens.add("moveTo:")
        else:
            seen_action_tokens.add(action)
    unseen_action_tokens = sorted(all_defined_action_prefixes - seen_action_tokens)

    return {
        "runs_total": coverage["runs_total"],
        "runs_by_policy": dict(coverage["runs_by_policy"]),
        "intents": dict(coverage["intents"]),
        "actions": dict(coverage["actions"]),
        "luggage_statuses": dict(coverage["luggage_statuses"]),
        "flag_true_counts": dict(coverage["flag_true_counts"]),
        "event_counters_total": dict(coverage["event_counters_total"]),
        "max_tick_by_policy": dict(coverage["max_tick_by_policy"]),
        "unseen_intents": unseen_intents,
        "unseen_action_tokens": unseen_action_tokens,
    }


def _build_likely_unused(static_data: dict[str, Any], dynamic_data: dict[str, Any]) -> list[dict[str, Any]]:
    findings = []

    for field in static_data["never_read_in_class"]:
        findings.append(
            {
                "kind": "field_never_read_in_passenger_class",
                "name": field,
                "evidence": {
                    "external_dot_refs": static_data["external_dot_refs"].get(field, []),
                },
                "confidence": "high" if not static_data["external_dot_refs"].get(field) else "medium",
            }
        )

    for intent in dynamic_data["unseen_intents"]:
        findings.append(
            {
                "kind": "intent_never_observed_in_runtime",
                "name": intent,
                "evidence": {"runs_total": dynamic_data["runs_total"]},
                "confidence": "high",
            }
        )

    for token in dynamic_data["unseen_action_tokens"]:
        findings.append(
            {
                "kind": "action_never_observed_in_runtime",
                "name": token,
                "evidence": {"runs_total": dynamic_data["runs_total"]},
                "confidence": "high",
            }
        )

    for flag in ["row_shift_complete"]:
        if dynamic_data["flag_true_counts"].get(flag, 0) == 0:
            findings.append(
                {
                    "kind": "belief_flag_never_true_in_runtime",
                    "name": flag,
                    "evidence": {"runs_total": dynamic_data["runs_total"]},
                    "confidence": "medium",
                }
            )

    return findings


def _write_markdown(report_path: Path, payload: dict[str, Any]) -> None:
    dyn = payload["dynamic"]
    static_data = payload["static"]
    findings = payload["likely_unused"]

    lines = []
    lines.append("# Property Usage Audit")
    lines.append("")
    lines.append("## Audit Scope")
    lines.append(f"- Runs: {dyn['runs_total']}")
    lines.append(f"- Policies: {', '.join(sorted(dyn['runs_by_policy'].keys()))}")
    lines.append("")

    lines.append("## Runtime Coverage")
    lines.append(f"- Seen intents: {', '.join(sorted(dyn['intents'].keys()))}")
    lines.append(f"- Unseen intents: {', '.join(dyn['unseen_intents']) if dyn['unseen_intents'] else 'None'}")
    lines.append("")

    lines.append("- Seen action tokens:")
    action_tokens = sorted({"moveTo:" if a.startswith("moveTo:") else a for a in dyn["actions"].keys()})
    lines.append(f"  {', '.join(action_tokens)}")
    lines.append(f"- Unseen action tokens: {', '.join(dyn['unseen_action_tokens']) if dyn['unseen_action_tokens'] else 'None'}")
    lines.append("")

    lines.append("- Belief flags true-counts:")
    for k in ["row_blocker", "row_blocked", "row_shift_complete"]:
        lines.append(f"  - {k}: {dyn['flag_true_counts'].get(k, 0)}")
    lines.append("")

    lines.append("## Static Field Usage (PassengerAgent)")
    lines.append("- Fields assigned in __init__ but never read in PassengerAgent methods:")
    if static_data["never_read_in_class"]:
        for field in static_data["never_read_in_class"]:
            refs = static_data["external_dot_refs"].get(field, [])
            lines.append(f"  - {field} (external .field refs: {sum(r['count'] for r in refs) if refs else 0})")
    else:
        lines.append("  - None")
    lines.append("")

    lines.append("## Likely Unused Candidates")
    if findings:
        for item in findings:
            lines.append(f"- [{item['confidence']}] {item['kind']}: {item['name']}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Notes")
    lines.append("- Runtime non-observation is empirical evidence, not a formal proof of impossibility.")
    lines.append("- Keep candidates as deprecation targets first, then remove incrementally and rerun the verification suite.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / "results" / "property_usage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    static_data = _static_passenger_field_usage(repo_root)
    dynamic_data = _dynamic_coverage(repo_root)
    likely_unused = _build_likely_unused(static_data, dynamic_data)

    payload = {
        "config": {
            "runs_per_policy": RUNS_PER_POLICY,
            "policies": POLICIES,
            "seed_start": SEED_START,
        },
        "static": static_data,
        "dynamic": dynamic_data,
        "likely_unused": likely_unused,
    }

    json_path = out_dir / "property_usage_audit.json"
    md_path = out_dir / "property_usage_audit.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(f"Audit complete. JSON: {json_path}")
    print(f"Audit complete. Markdown: {md_path}")


if __name__ == "__main__":
    main()
