"""Conditional-risk evaluation and UP-only repair for online control."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Mapping

from serverless_dag.core.workflow import WorkflowSpec
from serverless_dag.planner.artifacts import PlannerArtifacts
from serverless_dag.planner.cost import plan_cost_gbsec
from serverless_dag.risk.aggregation import aggregate_dag
from serverless_dag.risk.distributions import LogNormalParams


EPS = 1e-12


@dataclass(frozen=True)
class ConditionalRiskEval:
    risk: float
    e2e_params: LogNormalParams


@dataclass(frozen=True)
class DynamicUpgradeCandidate:
    stage_name: str
    from_tier: int
    to_tier: int
    predicted_start_ms: float
    warmup_deadline_ms: float


@dataclass(frozen=True)
class DynamicRepairResult:
    feasible: bool
    timed_out: bool
    memory_by_stage: dict[str, int]
    pre_risk: float
    post_risk: float
    states_evaluated: int
    wall_time_ms: float
    tier_changes: tuple[tuple[str, int, int], ...]
    trace: tuple[dict[str, object], ...]


def conditional_warm_risk(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
    completed_finish_ms: dict[str, float],
    slo_ms: float,
    rho: float = 0.0,
    stage_log_correlation: Mapping[tuple[str, str], float] | None = None,
) -> ConditionalRiskEval:
    """Condition on measured completions and model every remainder stage as warm."""

    stage_dists = {
        stage_name: artifacts.dist(stage_name, int(memory_by_stage[stage_name]), "warm")
        for stage_name in workflow.topological_order()
    }
    e2e = aggregate_dag(
        workflow=workflow,
        stage_dists=stage_dists,
        fixed_finish=completed_finish_ms,
        rho=float(rho),
        stage_correlation=stage_log_correlation,
    )
    return ConditionalRiskEval(risk=e2e.survival(float(slo_ms)), e2e_params=e2e)


def predicted_stage_times_ms(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
    completed_finish_ms: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Predict warm start/finish times from measured completed-stage finishes."""

    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for stage_name in workflow.topological_order():
        parents = workflow.parents_of(stage_name)
        starts[stage_name] = max((finishes[parent] for parent in parents), default=0.0)
        if stage_name in completed_finish_ms:
            finishes[stage_name] = float(completed_finish_ms[stage_name])
        else:
            tier = int(memory_by_stage[stage_name])
            finishes[stage_name] = starts[stage_name] + artifacts.dist(
                stage_name, tier, "warm"
            ).mean
    return starts, finishes


def jit_safe_upgrade_candidates(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
    completed_finish_ms: dict[str, float],
    started_stages: set[str],
    old_warmup_issued_stages: set[str],
    old_warmup_pending_stages: set[str],
    now_ms: float,
    tiers: tuple[int, ...],
) -> tuple[DynamicUpgradeCandidate, ...]:
    """Return UP-only stages whose old task is queued and new H95 fits."""

    predicted_start, _ = predicted_stage_times_ms(
        workflow=workflow,
        artifacts=artifacts,
        memory_by_stage=memory_by_stage,
        completed_finish_ms=completed_finish_ms,
    )
    completed = set(completed_finish_ms)
    candidates: list[DynamicUpgradeCandidate] = []
    for stage_name in workflow.topological_order():
        if (
            stage_name in completed
            or stage_name in started_stages
            or stage_name in old_warmup_issued_stages
            or stage_name not in old_warmup_pending_stages
        ):
            continue
        current_tier = int(memory_by_stage[stage_name])
        for to_tier in tiers:
            if to_tier <= current_tier:
                continue
            deadline_ms = predicted_start[stage_name] - artifacts.jit_lead_ms(
                stage_name, int(to_tier)
            )
            if deadline_ms + EPS < float(now_ms):
                continue
            candidates.append(
                DynamicUpgradeCandidate(
                    stage_name=stage_name,
                    from_tier=current_tier,
                    to_tier=int(to_tier),
                    predicted_start_ms=predicted_start[stage_name],
                    warmup_deadline_ms=deadline_ms,
                )
            )
    return tuple(candidates)


def repair_up_only(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    current_memory_by_stage: dict[str, int],
    completed_finish_ms: dict[str, float],
    slo_ms: float,
    tiers: tuple[int, ...],
    eligible_candidates: Iterable[DynamicUpgradeCandidate],
    stage_log_correlation: Mapping[tuple[str, str], float] | None = None,
    rho: float = 0.0,
    max_violation_rate: float = 0.05,
    max_decision_ms: float = 300.0,
) -> DynamicRepairResult:
    """Greedily buy the greatest conditional-risk reduction per added GB-s.

    This is deliberately Repair-only: every accepted transition is UP-only and
    no local, pairwise, or downward pruning routine is called.
    """

    start = time.perf_counter()
    deadline = start + max(0.0, float(max_decision_ms)) / 1000.0
    memory = {stage: int(tier) for stage, tier in current_memory_by_stage.items()}
    candidates = tuple(eligible_candidates)
    allowed: dict[str, set[int]] = {}
    for candidate in candidates:
        if candidate.to_tier <= candidate.from_tier:
            raise ValueError("dynamic candidates must be UP-only")
        if candidate.to_tier not in tiers or candidate.from_tier not in tiers:
            raise ValueError("dynamic candidate tier is outside the configured set")
        allowed.setdefault(candidate.stage_name, set()).add(int(candidate.to_tier))

    cache: dict[tuple[int, ...], ConditionalRiskEval] = {}
    stages = workflow.topological_order()

    def state_key(plan: dict[str, int]) -> tuple[int, ...]:
        return tuple(int(plan[stage]) for stage in stages)

    def evaluate(plan: dict[str, int]) -> ConditionalRiskEval:
        key = state_key(plan)
        if key not in cache:
            cache[key] = conditional_warm_risk(
                workflow=workflow,
                artifacts=artifacts,
                memory_by_stage=plan,
                completed_finish_ms=completed_finish_ms,
                slo_ms=slo_ms,
                rho=rho,
                stage_log_correlation=stage_log_correlation,
            )
        return cache[key]

    current = evaluate(memory)
    pre_risk = current.risk
    trace: list[dict[str, object]] = []
    changes: list[tuple[str, int, int]] = []
    timed_out = False

    while current.risk > max_violation_rate + EPS:
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        current_cost = plan_cost_gbsec(workflow, artifacts, memory)
        actions: list[tuple[float, float, float, str, int, ConditionalRiskEval]] = []
        for stage_name in stages:
            current_tier = int(memory[stage_name])
            for to_tier in sorted(allowed.get(stage_name, ())):
                if to_tier <= current_tier:
                    continue
                trial = dict(memory)
                trial[stage_name] = int(to_tier)
                evaluation = evaluate(trial)
                risk_delta = current.risk - evaluation.risk
                if risk_delta <= EPS:
                    continue
                cost_delta = plan_cost_gbsec(workflow, artifacts, trial) - current_cost
                efficiency = float("inf") if cost_delta <= EPS else risk_delta / cost_delta
                actions.append(
                    (
                        -efficiency,
                        cost_delta,
                        -risk_delta,
                        stage_name,
                        int(to_tier),
                        evaluation,
                    )
                )
                if time.perf_counter() >= deadline:
                    timed_out = True
                    break
            if timed_out:
                break
        if timed_out or not actions:
            break

        _, cost_delta, neg_risk_delta, stage_name, to_tier, chosen = min(actions)
        from_tier = int(memory[stage_name])
        memory[stage_name] = int(to_tier)
        changes.append((stage_name, from_tier, int(to_tier)))
        trace.append(
            {
                "step": len(trace) + 1,
                "stage_name": stage_name,
                "from_tier": from_tier,
                "to_tier": int(to_tier),
                "risk_before": current.risk,
                "risk_after": chosen.risk,
                "risk_delta": -neg_risk_delta,
                "cost_delta_gbsec": cost_delta,
            }
        )
        current = chosen

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    feasible = current.risk <= max_violation_rate + EPS and not timed_out
    return DynamicRepairResult(
        feasible=feasible,
        timed_out=timed_out,
        memory_by_stage=memory,
        pre_risk=pre_risk,
        post_risk=current.risk,
        states_evaluated=len(cache),
        wall_time_ms=elapsed_ms,
        tier_changes=tuple(changes),
        trace=tuple(trace),
    )
