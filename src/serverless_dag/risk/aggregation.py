"""Closed-form DAG aggregation for lognormal stage-latency models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from scipy.stats import norm

from serverless_dag.core.workflow import WorkflowSpec
from serverless_dag.risk.distributions import LogNormalParams


def fenton_wilkinson_sum(
    distributions: list[LogNormalParams], rho: float = 0.0
) -> LogNormalParams:
    """Approximate the sum of lognormals as another lognormal.

    ``rho`` is a homogeneous pairwise linear correlation used only for series
    addition. ``rho=0`` recovers the independent Fenton-Wilkinson sum.
    """

    if not distributions:
        raise ValueError("fenton_wilkinson_sum requires at least one distribution")
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    if len(distributions) == 1:
        return distributions[0]

    mean_sum = float(sum(dist.mean for dist in distributions))
    variances = [float(dist.variance) for dist in distributions]
    var_sum = float(sum(variances))
    if rho != 0.0:
        stds = [math.sqrt(max(0.0, value)) for value in variances]
        cross = 0.0
        for i in range(len(stds)):
            for j in range(i + 1, len(stds)):
                cross += stds[i] * stds[j]
        var_sum += 2.0 * rho * cross

    if mean_sum <= 0.0 or not math.isfinite(mean_sum):
        raise ValueError(f"invalid summed mean: {mean_sum}")
    if var_sum < 0.0 or not math.isfinite(var_sum):
        raise ValueError(f"invalid summed variance: {var_sum}")
    if var_sum == 0.0:
        return LogNormalParams(mu=math.log(mean_sum), sigma=0.0)

    sigma_sq = math.log1p(var_sum / (mean_sum**2))
    mu = math.log(mean_sum) - sigma_sq / 2.0
    return LogNormalParams(mu=mu, sigma=math.sqrt(max(0.0, sigma_sq)))


def clark_max(a: LogNormalParams, b: LogNormalParams, rho: float = 0.0) -> LogNormalParams:
    """Approximate ``max(a, b)`` as lognormal using Clark in log-space."""

    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")

    mu_x, sigma_x = a.mu, a.sigma
    mu_y, sigma_y = b.mu, b.sigma
    gap_sq = sigma_x**2 + sigma_y**2 - 2.0 * rho * sigma_x * sigma_y
    gap = math.sqrt(max(0.0, gap_sq))
    if gap < 1e-12:
        return a if mu_x >= mu_y else b

    alpha = (mu_x - mu_y) / gap
    pdf_alpha = float(norm.pdf(alpha))
    cdf_alpha = float(norm.cdf(alpha))
    cdf_neg_alpha = float(norm.cdf(-alpha))

    mean_log_max = mu_x * cdf_alpha + mu_y * cdf_neg_alpha + gap * pdf_alpha
    second_log_max = (
        (mu_x**2 + sigma_x**2) * cdf_alpha
        + (mu_y**2 + sigma_y**2) * cdf_neg_alpha
        + (mu_x + mu_y) * gap * pdf_alpha
    )
    var_log_max = max(0.0, second_log_max - mean_log_max**2)
    return LogNormalParams(mu=float(mean_log_max), sigma=math.sqrt(var_log_max))


def add_deterministic_shift(dist: LogNormalParams, shift_ms: float) -> LogNormalParams:
    """Approximate ``L + constant`` as lognormal while preserving variance."""

    if shift_ms < 0.0 or not math.isfinite(shift_ms):
        raise ValueError(f"shift_ms must be finite and non-negative, got {shift_ms}")
    if shift_ms == 0.0:
        return dist

    new_mean = dist.mean + float(shift_ms)
    new_var = dist.variance
    if new_var == 0.0:
        return LogNormalParams(mu=math.log(new_mean), sigma=0.0)

    sigma_sq = math.log1p(new_var / (new_mean**2))
    mu = math.log(new_mean) - sigma_sq / 2.0
    return LogNormalParams(mu=mu, sigma=math.sqrt(max(0.0, sigma_sq)))


@dataclass(frozen=True)
class _CorrelatedVariable:
    name: str
    params: LogNormalParams


class _CorrelationWorkspace:
    """Propagate pairwise log-covariances through FW sums and Clark maxima."""

    def __init__(
        self,
        *,
        stage_dists: Mapping[str, LogNormalParams],
        stage_correlation: Mapping[tuple[str, str], float],
    ) -> None:
        self._variables: dict[str, _CorrelatedVariable] = {}
        self._log_covariance: dict[tuple[str, str], float] = {}
        self._counter = 0
        stage_names = tuple(stage_dists)
        matrix = _correlation_matrix(stage_names, stage_correlation)
        for stage_name, params in stage_dists.items():
            name = f"stage:{stage_name}"
            self._variables[name] = _CorrelatedVariable(name=name, params=params)
        for left_index, left_stage in enumerate(stage_names):
            left = self._variables[f"stage:{left_stage}"]
            for right_index, right_stage in enumerate(stage_names):
                right = self._variables[f"stage:{right_stage}"]
                self._set_log_covariance(
                    left,
                    right,
                    float(matrix[left_index, right_index])
                    * left.params.sigma
                    * right.params.sigma,
                )

    def stage(self, stage_name: str) -> _CorrelatedVariable:
        return self._variables[f"stage:{stage_name}"]

    def deterministic(self, value: float, label: str) -> _CorrelatedVariable:
        return self._new_variable(
            LogNormalParams(mu=math.log(float(value)), sigma=0.0),
            label=label,
            covariance_to_existing={},
        )

    def independent(
        self, params: LogNormalParams, label: str
    ) -> _CorrelatedVariable:
        """Add an external variable independent of profiled stage residuals."""

        return self._new_variable(
            params,
            label=label,
            log_covariance_to_existing={},
        )

    def sum(
        self,
        left: _CorrelatedVariable,
        right: _CorrelatedVariable,
        *,
        label: str,
    ) -> _CorrelatedVariable:
        covariance = self._original_covariance(left, right)
        mean = left.params.mean + right.params.mean
        variance = (
            left.params.variance
            + right.params.variance
            + 2.0 * covariance
        )
        if variance < -1e-8:
            raise ValueError(f"negative variance while aggregating {label}: {variance}")
        params = _lognormal_from_moments(mean, max(0.0, variance))
        covariance_to_existing = {
            other.name: self._original_covariance(left, other)
            + self._original_covariance(right, other)
            for other in self._variables.values()
        }
        return self._new_variable(
            params,
            label=label,
            covariance_to_existing=covariance_to_existing,
        )

    def maximum(
        self,
        left: _CorrelatedVariable,
        right: _CorrelatedVariable,
        *,
        label: str,
    ) -> _CorrelatedVariable:
        covariance = self._get_log_covariance(left, right)
        gap_sq = (
            left.params.sigma**2
            + right.params.sigma**2
            - 2.0 * covariance
        )
        gap = math.sqrt(max(0.0, gap_sq))
        if gap < 1e-12:
            return left if left.params.mu >= right.params.mu else right

        alpha = (left.params.mu - right.params.mu) / gap
        cdf_alpha = float(norm.cdf(alpha))
        cdf_neg_alpha = float(norm.cdf(-alpha))
        pdf_alpha = float(norm.pdf(alpha))
        mean_log = (
            left.params.mu * cdf_alpha
            + right.params.mu * cdf_neg_alpha
            + gap * pdf_alpha
        )
        second_log = (
            (left.params.mu**2 + left.params.sigma**2) * cdf_alpha
            + (right.params.mu**2 + right.params.sigma**2) * cdf_neg_alpha
            + (left.params.mu + right.params.mu) * gap * pdf_alpha
        )
        params = LogNormalParams(
            mu=float(mean_log),
            sigma=math.sqrt(max(0.0, second_log - mean_log**2)),
        )
        log_covariance_to_existing = {
            other.name: (
                self._get_log_covariance(left, other) * cdf_alpha
                + self._get_log_covariance(right, other) * cdf_neg_alpha
            )
            for other in self._variables.values()
        }
        return self._new_variable(
            params,
            label=label,
            log_covariance_to_existing=log_covariance_to_existing,
        )

    def _new_variable(
        self,
        params: LogNormalParams,
        *,
        label: str,
        covariance_to_existing: Mapping[str, float] | None = None,
        log_covariance_to_existing: Mapping[str, float] | None = None,
    ) -> _CorrelatedVariable:
        if (covariance_to_existing is None) == (log_covariance_to_existing is None):
            raise ValueError("provide exactly one covariance representation")
        name = f"aggregate:{self._counter}:{label}"
        self._counter += 1
        variable = _CorrelatedVariable(name=name, params=params)
        for other in tuple(self._variables.values()):
            if covariance_to_existing is not None:
                original_covariance = float(covariance_to_existing.get(other.name, 0.0))
                denominator = variable.params.mean * other.params.mean
                ratio = original_covariance / denominator if denominator > 0.0 else 0.0
                if ratio <= -1.0:
                    raise ValueError(
                        f"invalid covariance while aggregating {label}: ratio={ratio}"
                    )
                log_covariance = math.log1p(ratio)
            else:
                assert log_covariance_to_existing is not None
                log_covariance = float(
                    log_covariance_to_existing.get(other.name, 0.0)
                )
            self._set_log_covariance(variable, other, log_covariance)
        self._variables[name] = variable
        self._set_log_covariance(variable, variable, variable.params.sigma**2)
        return variable

    def _original_covariance(
        self, left: _CorrelatedVariable, right: _CorrelatedVariable
    ) -> float:
        if left.name == right.name:
            return float(left.params.variance)
        covariance = self._get_log_covariance(left, right)
        return float(left.params.mean * right.params.mean * math.expm1(covariance))

    def _get_log_covariance(
        self, left: _CorrelatedVariable, right: _CorrelatedVariable
    ) -> float:
        return float(self._log_covariance.get(_pair_key(left.name, right.name), 0.0))

    def _set_log_covariance(
        self,
        left: _CorrelatedVariable,
        right: _CorrelatedVariable,
        value: float,
    ) -> None:
        self._log_covariance[_pair_key(left.name, right.name)] = float(value)


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _lognormal_from_moments(mean: float, variance: float) -> LogNormalParams:
    if mean <= 0.0 or not math.isfinite(mean):
        raise ValueError(f"invalid mean: {mean}")
    if variance < 0.0 or not math.isfinite(variance):
        raise ValueError(f"invalid variance: {variance}")
    if variance == 0.0:
        return LogNormalParams(mu=math.log(mean), sigma=0.0)
    sigma_sq = math.log1p(variance / mean**2)
    return LogNormalParams(
        mu=math.log(mean) - sigma_sq / 2.0,
        sigma=math.sqrt(max(0.0, sigma_sq)),
    )


def _correlation_matrix(
    stage_names: tuple[str, ...],
    stage_correlation: Mapping[tuple[str, str], float],
) -> np.ndarray:
    known = set(stage_names)
    unknown = sorted(
        {
            stage
            for pair in stage_correlation
            for stage in pair
            if stage not in known
        }
    )
    if unknown:
        raise ValueError(f"stage correlation contains unknown stages: {unknown}")
    matrix = np.eye(len(stage_names), dtype=float)
    missing: list[tuple[str, str]] = []
    for left_index, left in enumerate(stage_names):
        for right_index in range(left_index + 1, len(stage_names)):
            right = stage_names[right_index]
            direct = stage_correlation.get((left, right))
            reverse = stage_correlation.get((right, left))
            if direct is None and reverse is None:
                missing.append((left, right))
                continue
            if direct is not None and reverse is not None and not math.isclose(
                float(direct), float(reverse), abs_tol=1e-8
            ):
                raise ValueError(f"asymmetric correlation for {left}/{right}")
            value = float(direct if direct is not None else reverse)
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    if missing:
        raise ValueError(f"stage correlation missing pairs: {missing}")
    if not np.isfinite(matrix).all() or np.max(np.abs(matrix)) > 1.0 + 1e-8:
        raise ValueError("stage correlation must be finite and lie in [-1, 1]")
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            "stage correlation matrix must be positive semidefinite; "
            f"minimum eigenvalue={minimum_eigenvalue:.6g}"
        )
    return matrix


def _aggregate_dag_with_stage_correlation(
    *,
    workflow: WorkflowSpec,
    stage_dists: dict[str, LogNormalParams],
    fixed_finish: dict[str, float],
    stage_correlation: Mapping[tuple[str, str], float],
    ready_barrier_dists: Mapping[str, LogNormalParams],
) -> LogNormalParams:
    workspace = _CorrelationWorkspace(
        stage_dists=stage_dists,
        stage_correlation=stage_correlation,
    )
    finish: dict[str, _CorrelatedVariable] = {}
    for node_name in workflow.topological_order():
        if node_name in fixed_finish:
            finish[node_name] = workspace.deterministic(
                float(fixed_finish[node_name]), label=f"fixed:{node_name}"
            )
            continue
        parents = workflow.parents_of(node_name)
        if not parents:
            finish[node_name] = workspace.stage(node_name)
            continue
        arrival = finish[parents[0]]
        for parent in parents[1:]:
            arrival = workspace.maximum(
                arrival,
                finish[parent],
                label=f"arrival:{node_name}",
            )
        barrier = ready_barrier_dists.get(node_name)
        if barrier is not None:
            arrival = workspace.maximum(
                arrival,
                workspace.independent(barrier, label=f"barrier:{node_name}"),
                label=f"ready:{node_name}",
            )
        finish[node_name] = workspace.sum(
            arrival,
            workspace.stage(node_name),
            label=f"finish:{node_name}",
        )

    sinks = workflow.sinks()
    e2e = finish[sinks[0]]
    for sink in sinks[1:]:
        e2e = workspace.maximum(e2e, finish[sink], label="e2e")
    return e2e.params


def aggregate_dag(
    workflow: WorkflowSpec,
    stage_dists: dict[str, LogNormalParams],
    transition_overhead_ms: float = 0.0,
    fixed_finish: dict[str, float] | None = None,
    rho: float = 0.0,
    stage_correlation: Mapping[tuple[str, str], float] | None = None,
    ready_barrier_dists: Mapping[str, LogNormalParams] | None = None,
) -> LogNormalParams:
    """Aggregate an arbitrary workflow DAG into an E2E lognormal distribution.

    A ready barrier is an external completion time measured from workflow start.
    A stage carrying one starts after both its DAG parents and the barrier finish.
    JIT uses this to represent ``max(parent completion, warmup completion)``.
    """

    if transition_overhead_ms < 0.0 or not math.isfinite(transition_overhead_ms):
        raise ValueError(
            f"transition_overhead_ms must be finite and non-negative, got {transition_overhead_ms}"
        )

    node_names = set(workflow.nodes)
    dist_names = set(stage_dists)
    missing = sorted(node_names - dist_names)
    unknown = sorted(dist_names - node_names)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing stage distributions: {missing}")
        if unknown:
            parts.append(f"unknown stage distributions: {unknown}")
        raise ValueError("; ".join(parts))

    fixed_finish = fixed_finish or {}
    ready_barrier_dists = ready_barrier_dists or {}
    unknown_fixed = sorted(set(fixed_finish) - node_names)
    if unknown_fixed:
        raise ValueError(f"unknown fixed finish nodes: {unknown_fixed}")
    for node_name, value in fixed_finish.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"fixed finish for {node_name!r} must be finite and > 0")
        incomplete = [
            parent
            for parent in workflow.parents_of(node_name)
            if parent not in fixed_finish
        ]
        if incomplete:
            raise ValueError(
                "fixed finish nodes require fixed parents; "
                f"{node_name!r} has incomplete parents {incomplete}"
            )

    unknown_barriers = sorted(set(ready_barrier_dists) - node_names)
    if unknown_barriers:
        raise ValueError(f"ready barriers contain unknown stages: {unknown_barriers}")
    root_barriers = sorted(
        name for name in ready_barrier_dists if not workflow.parents_of(name)
    )
    if root_barriers:
        raise ValueError(f"ready barriers require parent stages: {root_barriers}")

    longest_edges = {name: 0 for name in workflow.nodes}
    for node_name in workflow.topological_order():
        parents = workflow.parents_of(node_name)
        if parents:
            longest_edges[node_name] = max(longest_edges[parent] + 1 for parent in parents)

    if stage_correlation is not None:
        if rho != 0.0:
            raise ValueError("rho and stage_correlation cannot be used together")
        e2e = _aggregate_dag_with_stage_correlation(
            workflow=workflow,
            stage_dists=stage_dists,
            fixed_finish=fixed_finish,
            stage_correlation=stage_correlation,
            ready_barrier_dists=ready_barrier_dists,
        )
        sinks = workflow.sinks()
        if not sinks:
            raise ValueError("workflow has no sink nodes")
        critical_path_edges = max(longest_edges[sink] for sink in sinks)
        return add_deterministic_shift(
            e2e, critical_path_edges * transition_overhead_ms
        )

    finish: dict[str, LogNormalParams] = {}
    for node_name in workflow.topological_order():
        parents = workflow.parents_of(node_name)
        if node_name in fixed_finish:
            finish[node_name] = LogNormalParams(
                mu=math.log(float(fixed_finish[node_name])), sigma=0.0
            )
            continue

        if not parents:
            finish[node_name] = stage_dists[node_name]
            continue

        arrival = finish[parents[0]]
        for parent in parents[1:]:
            arrival = clark_max(arrival, finish[parent], rho=0.0)
        barrier = ready_barrier_dists.get(node_name)
        if barrier is not None:
            arrival = clark_max(arrival, barrier, rho=0.0)
        finish[node_name] = fenton_wilkinson_sum(
            [arrival, stage_dists[node_name]], rho=rho
        )

    sinks = workflow.sinks()
    if not sinks:
        raise ValueError("workflow has no sink nodes")
    e2e = finish[sinks[0]]
    for sink in sinks[1:]:
        e2e = clark_max(e2e, finish[sink], rho=0.0)

    critical_path_edges = max(longest_edges[sink] for sink in sinks)
    return add_deterministic_shift(e2e, critical_path_edges * transition_overhead_ms)


def conditional_risk(
    workflow: WorkflowSpec,
    stage_dists: dict[str, LogNormalParams],
    completed_finish_ms: dict[str, float],
    slo_ms: float,
    transition_overhead_ms: float = 0.0,
    rho: float = 0.0,
    stage_correlation: Mapping[tuple[str, str], float] | None = None,
) -> float:
    """Compute ``P(E2E > SLO | completed stage finish times)``."""

    e2e = aggregate_dag(
        workflow=workflow,
        stage_dists=stage_dists,
        transition_overhead_ms=transition_overhead_ms,
        fixed_finish=completed_finish_ms,
        rho=rho,
        stage_correlation=stage_correlation,
    )
    return e2e.survival(slo_ms)
