"""JIT-aware entry-mixture risk model for static workflow plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from serverless_dag.core.workflow import WorkflowSpec
from serverless_dag.planner.artifacts import PlannerArtifacts
from serverless_dag.risk.aggregation import (
    add_deterministic_shift,
    aggregate_dag,
    clark_max,
    fenton_wilkinson_sum,
)
from serverless_dag.risk.distributions import LogNormalParams


@dataclass(frozen=True)
class PlanRiskResult:
    p_entry_cold: float
    entry_cold_model: str
    e2e_warm_params: LogNormalParams
    e2e_cold_entry_params: LogNormalParams
    p_violation_warm: float
    p_violation_cold_entry: float
    p_violation_total: float
    expected_e2e_ms: float
    first_hop_sync_mean_warm_ms: float = 0.0
    first_hop_sync_mean_cold_ms: float = 0.0
    entry_cold_overhead_shift_ms: float = 0.0
    entry_cold_overhead_mean_ms: float = 0.0
    entry_cold_overhead_p95_ms: float = 0.0

    @property
    def first_hop_sync_shift_warm_ms(self) -> float:
        """Compatibility alias for the modeled mean warm-entry wait."""

        return self.first_hop_sync_mean_warm_ms

    @property
    def first_hop_sync_shift_cold_ms(self) -> float:
        """Compatibility alias for the modeled mean cold-entry wait."""

        return self.first_hop_sync_mean_cold_ms


def estimate_first_hop_sync_shift_ms(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
    entry_latency_class: str,
    entry_quantile: float = 0.5,
) -> float:
    """Estimate JIT sync wait for the first hop from measured lead and entry time.

    The first downstream warmup is scheduled at workflow start when
    ``needed_at - lead`` lies in the past. In that common case the sync wait at
    the first real downstream invoke is approximately:

    ``max(0, first_hop_warmup_lead - entry_dispatch_time)``.

    We approximate the entry dispatch time with a configurable quantile of the
    entry distribution. Multiple entry children use the maximum estimated wait,
    because a slow first-hop branch can still hold the DAG's join path.
    """

    if not 0.0 < entry_quantile < 1.0:
        raise ValueError(f"entry_quantile must be in (0, 1), got {entry_quantile}")
    if entry_latency_class not in {"warm", "cold_like"}:
        raise ValueError(
            f"entry_latency_class must be 'warm' or 'cold_like', got {entry_latency_class}"
        )

    children = workflow.children_of(workflow.entry)
    if not children:
        return 0.0

    entry_tier = int(memory_by_stage[workflow.entry])
    entry_time_ms = artifacts.dist(
        workflow.entry,
        entry_tier,
        entry_latency_class,
    ).quantile(entry_quantile)
    waits = []
    for child in children:
        child_tier = int(memory_by_stage[child])
        waits.append(max(0.0, artifacts.jit_lead_ms(child, child_tier) - entry_time_ms))
    return float(max(waits, default=0.0))


def first_hop_warmup_barriers(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
) -> dict[str, LogNormalParams]:
    """Build first-hop warmup completion distributions relative to request start.

    Runtime JIT schedules a direct child's warmup at
    ``predicted_entry_completion - lead`` and clamps a past fire time to now.
    The returned barrier therefore models ``issue_offset + warmup_duration``.
    """

    entry_tier = int(memory_by_stage[workflow.entry])
    predicted_entry_ms = artifacts.warm_mean_ms(workflow.entry, entry_tier)
    barriers: dict[str, LogNormalParams] = {}
    for child in workflow.children_of(workflow.entry):
        if tuple(workflow.parents_of(child)) != (workflow.entry,):
            continue
        child_tier = int(memory_by_stage[child])
        lead_ms = artifacts.jit_lead_ms(child, child_tier)
        issue_offset_ms = max(0.0, predicted_entry_ms - lead_ms)
        barriers[child] = add_deterministic_shift(
            artifacts.jit_warmup_dist(child, child_tier),
            issue_offset_ms,
        )
    return barriers


def _first_hop_mean_wait_ms(
    *,
    entry_dist: LogNormalParams,
    barriers: Mapping[str, LogNormalParams],
) -> float:
    """Return the largest Clark-estimated mean wait among first-hop stages."""

    return float(
        max(
            (
                max(0.0, clark_max(entry_dist, barrier, rho=0.0).mean - entry_dist.mean)
                for barrier in barriers.values()
            ),
            default=0.0,
        )
    )


def compute_jit_mixture_risk(
    *,
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
    slo_ms: float,
    p_entry_cold: float,
    rho: float = 0.0,
    stage_log_correlation: Mapping[tuple[str, str], float] | None = None,
    sync_shift_warm_ms: float = 0.0,
    sync_shift_cold_ms: float = 0.0,
    include_first_hop_sync: bool = False,
    first_hop_entry_quantile: float = 0.5,
    entry_cold_model: str = "cold_like",
) -> PlanRiskResult:
    """Compute JIT-aware ``P(E2E > SLO)`` for a static memory plan.

    Scenario definitions:
    - entry-warm: every stage uses its measured warm distribution.
    - entry-cold+JIT, ``entry_cold_model="cold_like"``: the entry stage uses
      measured cold_like; downstream stages use warm distributions.
    - entry-cold+JIT, ``entry_cold_model="additive_h95"``: the entry warm
      distribution receives the empirical cold-overhead p95 before DAG
      propagation.
    - entry-cold+JIT, ``entry_cold_model="lognormal_overhead"``: the entry warm
      distribution is summed with an independent fitted cold platform-overhead
      distribution using Fenton--Wilkinson moment matching.

    ``p_entry_cold`` is explicit. It is not inferred from a safety factor.
    Optional sync shifts are deterministic offsets and default to zero. When
    ``include_first_hop_sync`` is set, each direct child receives a stochastic
    ready barrier. Its start time becomes the maximum of entry completion and
    warmup completion, preserving the overlap instead of adding a fixed delay.
    """

    if entry_cold_model not in {
        "cold_like",
        "additive_h95",
        "lognormal_overhead",
    }:
        raise ValueError(
            "entry_cold_model must be 'cold_like', 'additive_h95', or "
            "'lognormal_overhead', "
            f"got {entry_cold_model!r}"
        )
    if slo_ms <= 0.0 or not math.isfinite(float(slo_ms)):
        raise ValueError(f"slo_ms must be finite and positive, got {slo_ms}")
    p_entry_cold = float(p_entry_cold)
    if not 0.0 <= p_entry_cold <= 1.0 or not math.isfinite(p_entry_cold):
        raise ValueError(f"p_entry_cold must be finite in [0, 1], got {p_entry_cold}")

    missing = sorted(set(workflow.nodes).difference(memory_by_stage))
    unknown = sorted(set(memory_by_stage).difference(workflow.nodes))
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing memory tiers for stages: {missing}")
        if unknown:
            parts.append(f"unknown memory tier stages: {unknown}")
        raise ValueError("; ".join(parts))

    warm_dists: dict[str, LogNormalParams] = {}
    cold_entry_dists: dict[str, LogNormalParams] = {}
    for stage_name in workflow.topological_order():
        tier = int(memory_by_stage[stage_name])
        warm = artifacts.dist(stage_name, tier, "warm")
        warm_dists[stage_name] = warm
        cold_entry_dists[stage_name] = (
            artifacts.dist(stage_name, tier, "cold_like")
            if stage_name == workflow.entry and entry_cold_model == "cold_like"
            else warm
        )

    barriers = (
        first_hop_warmup_barriers(
            workflow=workflow,
            artifacts=artifacts,
            memory_by_stage=memory_by_stage,
        )
        if include_first_hop_sync
        else {}
    )
    entry_cold_overhead_shift = 0.0
    entry_cold_overhead_mean = 0.0
    entry_cold_overhead_p95 = 0.0
    overhead: LogNormalParams | None = None
    if entry_cold_model == "additive_h95":
        entry_cold_overhead_shift = artifacts.entry_cold_overhead_p95(
            workflow.entry,
            int(memory_by_stage[workflow.entry]),
        )
        entry_cold_overhead_mean = entry_cold_overhead_shift
        entry_cold_overhead_p95 = entry_cold_overhead_shift
    elif entry_cold_model == "lognormal_overhead":
        overhead = artifacts.entry_cold_overhead_dist(
            workflow.entry,
            int(memory_by_stage[workflow.entry]),
        )
        entry_cold_overhead_mean = overhead.mean
        entry_cold_overhead_p95 = overhead.quantile(0.95)

    if barriers:
        if entry_cold_model == "additive_h95":
            cold_entry_dists[workflow.entry] = add_deterministic_shift(
                warm_dists[workflow.entry],
                entry_cold_overhead_shift,
            )
        elif overhead is not None:
            cold_entry_dists[workflow.entry] = fenton_wilkinson_sum(
                [warm_dists[workflow.entry], overhead],
                rho=0.0,
            )

    warm_base = aggregate_dag(
        workflow,
        warm_dists,
        rho=rho,
        stage_correlation=stage_log_correlation,
        ready_barrier_dists=barriers,
    )
    warm_e2e = add_deterministic_shift(warm_base, sync_shift_warm_ms)
    if barriers or entry_cold_model == "cold_like":
        cold_base = aggregate_dag(
            workflow,
            cold_entry_dists,
            rho=rho,
            stage_correlation=stage_log_correlation,
            ready_barrier_dists=barriers,
        )
        cold_entry_e2e = add_deterministic_shift(cold_base, sync_shift_cold_ms)
    elif entry_cold_model == "additive_h95":
        cold_entry_e2e = add_deterministic_shift(
            warm_base,
            sync_shift_cold_ms + entry_cold_overhead_shift,
        )
    else:
        assert overhead is not None
        cold_entry_e2e = add_deterministic_shift(
            fenton_wilkinson_sum([warm_base, overhead], rho=0.0),
            sync_shift_cold_ms,
        )
    first_hop_warm_shift = _first_hop_mean_wait_ms(
        entry_dist=warm_dists[workflow.entry],
        barriers=barriers,
    )
    first_hop_cold_shift = _first_hop_mean_wait_ms(
        entry_dist=cold_entry_dists[workflow.entry],
        barriers=barriers,
    )
    p_warm = warm_e2e.survival(float(slo_ms))
    p_cold = cold_entry_e2e.survival(float(slo_ms))
    p_total = (1.0 - p_entry_cold) * p_warm + p_entry_cold * p_cold
    expected = (1.0 - p_entry_cold) * warm_e2e.mean + p_entry_cold * cold_entry_e2e.mean
    return PlanRiskResult(
        p_entry_cold=p_entry_cold,
        entry_cold_model=entry_cold_model,
        e2e_warm_params=warm_e2e,
        e2e_cold_entry_params=cold_entry_e2e,
        p_violation_warm=float(p_warm),
        p_violation_cold_entry=float(p_cold),
        p_violation_total=float(p_total),
        expected_e2e_ms=float(expected),
        first_hop_sync_mean_warm_ms=float(first_hop_warm_shift),
        first_hop_sync_mean_cold_ms=float(first_hop_cold_shift),
        entry_cold_overhead_shift_ms=float(entry_cold_overhead_shift),
        entry_cold_overhead_mean_ms=float(entry_cold_overhead_mean),
        entry_cold_overhead_p95_ms=float(entry_cold_overhead_p95),
    )
