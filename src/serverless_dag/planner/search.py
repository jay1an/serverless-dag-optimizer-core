"""Search algorithms for static memory-tier plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable

from serverless_dag.core.workflow import WorkflowSpec
from serverless_dag.planner.artifacts import PlannerArtifacts
from serverless_dag.planner.cost import plan_cost_gbsec
from serverless_dag.planner.risk_model import PlanRiskResult, compute_jit_mixture_risk


EPS = 1e-12


@dataclass(frozen=True)
class PlannerConfig:
    slo_ms: float
    max_violation_rate: float
    tiers: tuple[int, ...]
    stages: tuple[str, ...]
    p_entry_cold: float
    rho: float = 0.0
    stage_log_correlation: dict[tuple[str, str], float] | None = None
    sync_shift_warm_ms: float = 0.0
    sync_shift_cold_ms: float = 0.0
    include_first_hop_sync: bool = False
    first_hop_entry_quantile: float = 0.5
    entry_cold_model: str = "cold_like"
    fixed_memory_by_stage: dict[str, int] | None = None


@dataclass(frozen=True)
class StateEval:
    state_key: tuple[int, ...]
    memory_by_stage: dict[str, int]
    risk_result: PlanRiskResult
    cost_gbsec: float

    @property
    def violation_rate(self) -> float:
        return float(self.risk_result.p_violation_total)

    @property
    def expected_e2e_ms(self) -> float:
        return float(self.risk_result.expected_e2e_ms)


@dataclass(frozen=True)
class SearchResult:
    method: str
    memory_by_stage: dict[str, int]
    evaluation: StateEval
    feasible: bool
    iterations: int
    states_evaluated: int
    wall_time_sec: float
    trace: tuple[dict[str, Any], ...] = ()


class EvalContext:
    def __init__(
        self,
        *,
        workflow: WorkflowSpec,
        artifacts: PlannerArtifacts,
        config: PlannerConfig,
    ):
        self.workflow = workflow
        self.artifacts = artifacts
        self.config = config
        self.cache: dict[tuple[int, ...], StateEval] = {}

    def memory(self, state_key: tuple[int, ...]) -> dict[str, int]:
        memory = {
            stage_name: int(self.config.tiers[state_key[index]])
            for index, stage_name in enumerate(self.config.stages)
        }
        if self.config.fixed_memory_by_stage:
            for stage_name, tier in self.config.fixed_memory_by_stage.items():
                memory[stage_name] = int(tier)
        return memory

    def evaluate(self, state_key: tuple[int, ...]) -> StateEval:
        if state_key in self.cache:
            return self.cache[state_key]
        memory = self.memory(state_key)
        risk = compute_jit_mixture_risk(
            workflow=self.workflow,
            artifacts=self.artifacts,
            memory_by_stage=memory,
            slo_ms=self.config.slo_ms,
            p_entry_cold=self.config.p_entry_cold,
            rho=self.config.rho,
            stage_log_correlation=self.config.stage_log_correlation,
            sync_shift_warm_ms=self.config.sync_shift_warm_ms,
            sync_shift_cold_ms=self.config.sync_shift_cold_ms,
            include_first_hop_sync=self.config.include_first_hop_sync,
            first_hop_entry_quantile=self.config.first_hop_entry_quantile,
            entry_cold_model=self.config.entry_cold_model,
        )
        cost = plan_cost_gbsec(self.workflow, self.artifacts, memory)
        out = StateEval(
            state_key=state_key,
            memory_by_stage=memory,
            risk_result=risk,
            cost_gbsec=cost,
        )
        self.cache[state_key] = out
        return out


def format_memory_config(memory_by_stage: dict[str, int], stages: Iterable[str]) -> str:
    return ",".join(f"{stage}:{int(memory_by_stage[stage])}" for stage in stages)


def is_feasible(evaluation: StateEval, config: PlannerConfig) -> bool:
    return evaluation.violation_rate <= config.max_violation_rate + EPS


def _initial_state(config: PlannerConfig) -> tuple[int, ...]:
    fixed = config.fixed_memory_by_stage or {}
    state: list[int] = []
    for stage_name in config.stages:
        if stage_name in fixed:
            tier = int(fixed[stage_name])
            if tier not in config.tiers:
                raise ValueError(f"fixed tier {tier} for {stage_name} is not in tiers")
            state.append(config.tiers.index(tier))
        else:
            state.append(0)
    return tuple(state)


def _candidate_state(
    state_key: tuple[int, ...], group_index: int, value_index: int
) -> tuple[int, ...]:
    out = list(state_key)
    out[group_index] = int(value_index)
    return tuple(out)


def _candidate_actions(
    ctx: EvalContext,
    state_key: tuple[int, ...],
    *,
    all_higher: bool,
) -> list[dict[str, Any]]:
    current = ctx.evaluate(state_key)
    config = ctx.config
    actions: list[dict[str, Any]] = []
    fixed = config.fixed_memory_by_stage or {}
    for group_index, stage_name in enumerate(config.stages):
        if stage_name in fixed:
            continue
        current_index = state_key[group_index]
        if current_index >= len(config.tiers) - 1:
            continue
        value_range = (
            range(current_index + 1, len(config.tiers))
            if all_higher
            else [current_index + 1]
        )
        for value_index in value_range:
            next_key = _candidate_state(state_key, group_index, value_index)
            evaluation = ctx.evaluate(next_key)
            actions.append(
                {
                    "stage": stage_name,
                    "value_index": value_index,
                    "value": int(config.tiers[value_index]),
                    "state_key": next_key,
                    "evaluation": evaluation,
                    "risk_delta": current.violation_rate - evaluation.violation_rate,
                    "expected_delta": current.expected_e2e_ms - evaluation.expected_e2e_ms,
                    "cost_delta": evaluation.cost_gbsec - current.cost_gbsec,
                }
            )
    return actions


def _efficiency_key(action: dict[str, Any]) -> tuple[float, float, float, str, int]:
    risk_delta = float(action["risk_delta"])
    expected_delta = float(action["expected_delta"])
    cost_delta = float(action["cost_delta"])
    if risk_delta > EPS:
        score = math.inf if cost_delta <= 0.0 else risk_delta / cost_delta
        primary = 0.0
        secondary = -score
    elif expected_delta > 1e-9:
        score = math.inf if cost_delta <= 0.0 else expected_delta / cost_delta
        primary = 1.0
        secondary = -score
    else:
        primary = 2.0
        secondary = 0.0
    return (
        primary,
        secondary,
        abs(cost_delta),
        str(action["stage"]),
        int(action["value_index"]),
    )


def _repair_until_feasible(
    ctx: EvalContext,
    state_key: tuple[int, ...],
    *,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[int, ...]:
    current = ctx.evaluate(state_key)
    max_steps = (len(ctx.config.tiers) - 1) * len(ctx.config.stages)
    steps = 0
    while not is_feasible(current, ctx.config) and steps < max_steps:
        actions = [
            item
            for item in _candidate_actions(ctx, state_key, all_higher=True)
            if item["risk_delta"] > EPS
            or (
                item["expected_delta"] > 1e-9
                and item["evaluation"].violation_rate <= current.violation_rate + EPS
            )
        ]
        if not actions:
            break
        chosen = sorted(actions, key=_efficiency_key)[0]
        before_key = state_key
        state_key = chosen["state_key"]
        current = chosen["evaluation"]
        steps += 1
        if trace is not None:
            trace.append(
                _transition_trace_row(
                    ctx,
                    step=len(trace),
                    phase="repair",
                    before_key=before_key,
                    after_key=state_key,
                )
            )
    return state_key


def _local_cost_prune(
    ctx: EvalContext,
    state_key: tuple[int, ...],
    *,
    pairwise: bool = True,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[int, ...]:
    current = ctx.evaluate(state_key)
    if not is_feasible(current, ctx.config):
        return state_key

    fixed = ctx.config.fixed_memory_by_stage or {}
    improved = True
    while improved:
        improved = False
        best_key = state_key
        best_eval = current

        for group_index in range(len(ctx.config.stages)):
            if ctx.config.stages[group_index] in fixed:
                continue
            for value_index in range(len(ctx.config.tiers)):
                if value_index == state_key[group_index]:
                    continue
                candidate = _candidate_state(state_key, group_index, value_index)
                evaluation = ctx.evaluate(candidate)
                if not is_feasible(evaluation, ctx.config):
                    continue
                if evaluation.cost_gbsec < best_eval.cost_gbsec - EPS:
                    best_key = candidate
                    best_eval = evaluation

        if best_key != state_key:
            before_key = state_key
            state_key = best_key
            current = best_eval
            improved = True
            if trace is not None:
                trace.append(
                    _transition_trace_row(
                        ctx,
                        step=len(trace),
                        phase="single_prune",
                        before_key=before_key,
                        after_key=state_key,
                    )
                )
            continue

        if not pairwise:
            break

        for left in range(len(ctx.config.stages)):
            if ctx.config.stages[left] in fixed:
                continue
            for right in range(left + 1, len(ctx.config.stages)):
                if ctx.config.stages[right] in fixed:
                    continue
                for left_value in range(len(ctx.config.tiers)):
                    if left_value == state_key[left]:
                        continue
                    for right_value in range(len(ctx.config.tiers)):
                        if right_value == state_key[right]:
                            continue
                        candidate_list = list(state_key)
                        candidate_list[left] = left_value
                        candidate_list[right] = right_value
                        candidate = tuple(candidate_list)
                        evaluation = ctx.evaluate(candidate)
                        if not is_feasible(evaluation, ctx.config):
                            continue
                        if evaluation.cost_gbsec < best_eval.cost_gbsec - EPS:
                            best_key = candidate
                            best_eval = evaluation

        if best_key != state_key:
            before_key = state_key
            state_key = best_key
            current = best_eval
            improved = True
            if trace is not None:
                trace.append(
                    _transition_trace_row(
                        ctx,
                        step=len(trace),
                        phase="pairwise_prune",
                        before_key=before_key,
                        after_key=state_key,
                    )
                )
    return state_key


def repair_prune_ablation(ctx: EvalContext) -> dict[str, SearchResult]:
    """Run Repair-only, Single-Prune, and Full-Prune from one repair result.

    Repair is executed exactly once. The resulting evaluation cache is copied
    into two independent contexts so both prune branches start with the same
    state, candidate set, and already-evaluated repair states. This keeps the
    component comparison deterministic while preserving per-branch state and
    wall-time accounting.
    """

    repair_start = time.monotonic()
    start_key = _initial_state(ctx.config)
    repair_trace = [
        _ablation_trace_row(
            ctx,
            step=0,
            phase="init",
            before_key=start_key,
            after_key=start_key,
        )
    ]
    repaired_key = _repair_until_feasible(ctx, start_key, trace=repair_trace)
    repair_wall_time = time.monotonic() - repair_start
    repair_cache = dict(ctx.cache)
    repair_iterations = len(repair_trace) - 1

    results: dict[str, SearchResult] = {
        "repair_only": _ablation_result(
            method="repair_only",
            ctx=ctx,
            state_key=repaired_key,
            wall_time_sec=repair_wall_time,
            trace=repair_trace,
        )
    }

    for method, pairwise in (
        ("repair_single_prune", False),
        ("repair_full_prune", True),
    ):
        branch_ctx = _fork_eval_context(ctx, repair_cache)
        branch_trace = list(repair_trace)
        branch_start = time.monotonic()
        final_key = _local_cost_prune(
            branch_ctx,
            repaired_key,
            pairwise=pairwise,
            trace=branch_trace,
        )
        branch_wall_time = time.monotonic() - branch_start
        results[method] = _ablation_result(
            method=method,
            ctx=branch_ctx,
            state_key=final_key,
            wall_time_sec=repair_wall_time + branch_wall_time,
            trace=branch_trace,
        )

    repaired_memory = results["repair_only"].memory_by_stage
    fixed = ctx.config.fixed_memory_by_stage or {}
    for method, result in results.items():
        for stage_name, tier in fixed.items():
            if result.memory_by_stage[stage_name] != int(tier):
                raise AssertionError(
                    f"{method} changed fixed stage {stage_name}: "
                    f"expected {tier}, got {result.memory_by_stage[stage_name]}"
                )
        if any(tier not in ctx.config.tiers for tier in result.memory_by_stage.values()):
            raise AssertionError(f"{method} returned a tier outside the candidate set")

    if repaired_memory != ctx.memory(repaired_key):
        raise AssertionError("repair-only result does not match the shared repair state")
    if repair_iterations != results["repair_only"].iterations:
        raise AssertionError("repair iteration accounting is inconsistent")
    return results


def repair_prune_plan(ctx: EvalContext) -> SearchResult:
    """Risk-certified repair search with exact local/pairwise cost pruning."""

    start = time.monotonic()
    start_key = _initial_state(ctx.config)
    repaired = _repair_until_feasible(ctx, start_key)
    pruned = _local_cost_prune(ctx, repaired, pairwise=True)
    trace = (
        _trace_row(0, "init", ctx.evaluate(start_key)),
        _trace_row(1, "risk repair", ctx.evaluate(repaired)),
        _trace_row(2, "single/pairwise cost prune", ctx.evaluate(pruned)),
    )
    return _result("repair_prune", ctx, pruned, 2, start, trace)


def _trace_row(step: int, action: str, evaluation: StateEval) -> dict[str, Any]:
    return {
        "step": int(step),
        "action": action,
        "violation_rate": evaluation.violation_rate,
        "cost_gbsec": evaluation.cost_gbsec,
        "expected_e2e_ms": evaluation.expected_e2e_ms,
    }


def _ablation_trace_row(
    ctx: EvalContext,
    *,
    step: int,
    phase: str,
    before_key: tuple[int, ...],
    after_key: tuple[int, ...],
) -> dict[str, Any]:
    evaluation = ctx.evaluate(after_key)
    changes = []
    for index, stage_name in enumerate(ctx.config.stages):
        if before_key[index] == after_key[index]:
            continue
        changes.append(
            {
                "stage": stage_name,
                "from_tier": int(ctx.config.tiers[before_key[index]]),
                "to_tier": int(ctx.config.tiers[after_key[index]]),
            }
        )
    return {
        "step": int(step),
        "phase": phase,
        "action": ";".join(
            f"{item['stage']}:{item['from_tier']}->{item['to_tier']}"
            for item in changes
        )
        or "initial state",
        "changed_stages": ";".join(str(item["stage"]) for item in changes),
        "from_tiers": ";".join(str(item["from_tier"]) for item in changes),
        "to_tiers": ";".join(str(item["to_tier"]) for item in changes),
        "state_key": ";".join(str(value) for value in after_key),
        "memory_config": format_memory_config(
            evaluation.memory_by_stage, ctx.config.stages
        ),
        "violation_rate": evaluation.violation_rate,
        "cost_gbsec": evaluation.cost_gbsec,
        "expected_e2e_ms": evaluation.expected_e2e_ms,
        "warm_p95_ms": evaluation.risk_result.e2e_warm_params.quantile(0.95),
        "entry_cold_p95_ms": evaluation.risk_result.e2e_cold_entry_params.quantile(
            0.95
        ),
    }


def _transition_trace_row(
    ctx: EvalContext,
    *,
    step: int,
    phase: str,
    before_key: tuple[int, ...],
    after_key: tuple[int, ...],
) -> dict[str, Any]:
    return _ablation_trace_row(
        ctx,
        step=step,
        phase=phase,
        before_key=before_key,
        after_key=after_key,
    )


def _fork_eval_context(
    ctx: EvalContext,
    cache: dict[tuple[int, ...], StateEval],
) -> EvalContext:
    fork = EvalContext(
        workflow=ctx.workflow,
        artifacts=ctx.artifacts,
        config=ctx.config,
    )
    fork.cache = dict(cache)
    return fork


def _ablation_result(
    *,
    method: str,
    ctx: EvalContext,
    state_key: tuple[int, ...],
    wall_time_sec: float,
    trace: Iterable[dict[str, Any]],
) -> SearchResult:
    evaluation = ctx.evaluate(state_key)
    trace_tuple = tuple(trace)
    return SearchResult(
        method=method,
        memory_by_stage=evaluation.memory_by_stage,
        evaluation=evaluation,
        feasible=is_feasible(evaluation, ctx.config),
        iterations=max(0, len(trace_tuple) - 1),
        states_evaluated=len(ctx.cache),
        wall_time_sec=float(wall_time_sec),
        trace=trace_tuple,
    )


def _result(
    method: str,
    ctx: EvalContext,
    state_key: tuple[int, ...],
    iterations: int,
    start_time: float,
    trace: Iterable[dict[str, Any]],
) -> SearchResult:
    evaluation = ctx.evaluate(state_key)
    return SearchResult(
        method=method,
        memory_by_stage=evaluation.memory_by_stage,
        evaluation=evaluation,
        feasible=is_feasible(evaluation, ctx.config),
        iterations=int(iterations),
        states_evaluated=len(ctx.cache),
        wall_time_sec=time.monotonic() - start_time,
        trace=tuple(trace),
    )
