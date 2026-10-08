"""Command-line access to planning, risk evaluation, and entry-risk estimation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from serverless_dag.core.workflow import load_workflow
from serverless_dag.forecast.entry_cold import expected_entry_cold_fraction
from serverless_dag.planner.artifacts import PlannerArtifacts
from serverless_dag.planner.risk_model import compute_jit_mixture_risk
from serverless_dag.planner.search import EvalContext, PlannerConfig, repair_prune_plan


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "risk"):
        command = commands.add_parser(name)
        command.add_argument("--workflow", type=Path, required=True)
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--slo-ms", type=float, required=True)
        command.add_argument("--p-entry-cold", type=float, required=True)
        command.add_argument("--first-hop-sync", action=argparse.BooleanOptionalAction, default=True)
        if name == "plan":
            command.add_argument("--tiers", type=int, nargs="+", required=True)
            command.add_argument("--max-violation-rate", type=float, default=0.05)
        else:
            command.add_argument("--memory-json", type=Path, required=True)
    entry = commands.add_parser("entry-risk")
    entry.add_argument("--mean-demand", type=float, required=True)
    entry.add_argument("--available-capacity", type=float, required=True)
    entry.add_argument("--alpha", type=float, required=True)
    return root


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        if args.command == "entry-risk":
            result = {"p_entry_cold": expected_entry_cold_fraction(
                args.mean_demand, args.available_capacity, alpha=args.alpha)}
        else:
            workflow = load_workflow(args.workflow)
            artifacts = PlannerArtifacts.from_model_dir(args.model_dir, require_stage_correlation=True)
            if args.command == "plan":
                if args.tiers != sorted(set(args.tiers)) or any(tier <= 0 for tier in args.tiers):
                    raise ValueError("tiers must be distinct positive integers in increasing order")
                if not 0.0 <= args.max_violation_rate <= 1.0:
                    raise ValueError("max-violation-rate must be in [0, 1]")
                config = PlannerConfig(
                    slo_ms=args.slo_ms, max_violation_rate=args.max_violation_rate,
                    tiers=tuple(args.tiers), stages=workflow.topological_order(),
                    p_entry_cold=args.p_entry_cold, rho=0.0,
                    stage_log_correlation=artifacts.stage_log_correlation,
                    entry_cold_model="lognormal_overhead", include_first_hop_sync=args.first_hop_sync,
                )
                selected = repair_prune_plan(EvalContext(workflow=workflow, artifacts=artifacts, config=config))
                result = {
                    "memory_by_stage": selected.memory_by_stage,
                    "feasible": selected.feasible,
                    "violation_risk": selected.evaluation.violation_rate,
                    "execution_cost_gbsec": selected.evaluation.cost_gbsec,
                    "risk_evaluations": selected.states_evaluated,
                    "planner_time_sec": selected.wall_time_sec,
                }
            else:
                memory = json.loads(args.memory_json.read_text(encoding="utf-8"))
                if not isinstance(memory, dict) or set(memory) != set(workflow.nodes):
                    raise ValueError("memory-json must map every stage name to its memory in MB")
                if any(type(value) is not int or value <= 0 for value in memory.values()):
                    raise ValueError("memory values must be positive integers in MB")
                risk = compute_jit_mixture_risk(
                    workflow=workflow, artifacts=artifacts, memory_by_stage=memory,
                    slo_ms=args.slo_ms, p_entry_cold=args.p_entry_cold,
                    stage_log_correlation=artifacts.stage_log_correlation,
                    entry_cold_model="lognormal_overhead", include_first_hop_sync=args.first_hop_sync,
                )
                result = {
                    "violation_risk": risk.p_violation_total,
                    "warm_p50_ms": risk.e2e_warm_params.quantile(0.50),
                    "warm_p95_ms": risk.e2e_warm_params.quantile(0.95),
                    "cold_p50_ms": risk.e2e_cold_entry_params.quantile(0.50),
                    "cold_p95_ms": risk.e2e_cold_entry_params.quantile(0.95),
                }
    except (ValueError, KeyError, FileNotFoundError) as error:
        argument_parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
