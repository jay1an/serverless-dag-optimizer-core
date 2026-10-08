"""Asynchronous trigger and Repair-only controller for one workflow request."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Callable

from serverless_dag.core.workflow import WorkflowSpec
from serverless_dag.planner.artifacts import PlannerArtifacts
from serverless_dag.planner.dynamic_repair import (
    DynamicRepairResult,
    DynamicUpgradeCandidate,
    conditional_warm_risk,
    repair_up_only,
)


@dataclass(frozen=True)
class DynamicObservation:
    request_id: str
    completed_stage: str
    elapsed_ms: float
    stage_start_ms: float
    entry_cold: bool
    completed_finish_ms: dict[str, float]
    memory_by_stage: dict[str, int]
    eligible_candidates: tuple[DynamicUpgradeCandidate, ...]
    old_warmup_issued: bool
    observation_kind: str = "completion"
    watchdog_quantile: float | None = None
    watchdog_deadline_ms: float | None = None
    stage_elapsed_ms: float | None = None
    stage_completed_at_submit: bool = True
    plan_version: int = 0


ApplyCallback = Callable[[DynamicObservation, DynamicRepairResult], tuple[bool, str]]


class DynamicRepairController:
    """Evaluate completion observations off the workflow dispatch thread."""

    def __init__(
        self,
        *,
        workflow: WorkflowSpec,
        artifacts: PlannerArtifacts,
        tiers: tuple[int, ...],
        slo_ms: float,
        max_violation_rate: float = 0.05,
        max_decision_ms: float = 300.0,
        observe_only: bool = False,
        trigger_policy: str = "excess_risk",
        delta_risk: float = 0.05,
        reference_quantile: float = 0.95,
        max_repairs_per_request: int = 2,
        repair_target_rate: float = 0.025,
    ):
        if not artifacts.stage_log_correlation:
            raise ValueError(
                "dynamic repair requires a measured stage correlation matrix"
            )
        self.workflow = workflow
        self.artifacts = artifacts
        self.tiers = tiers
        self.slo_ms = float(slo_ms)
        self.max_violation_rate = float(max_violation_rate)
        self.max_decision_ms = float(max_decision_ms)
        self.observe_only = bool(observe_only)
        if trigger_policy not in {"absolute", "excess_risk"}:
            raise ValueError("trigger_policy must be 'absolute' or 'excess_risk'")
        if not 0.0 < float(reference_quantile) < 1.0:
            raise ValueError("reference_quantile must be in (0, 1)")
        if int(max_repairs_per_request) < 1:
            raise ValueError("max_repairs_per_request must be positive")
        if not 0.0 < float(repair_target_rate) <= self.max_violation_rate:
            raise ValueError(
                "repair_target_rate must be in (0, max_violation_rate]"
            )
        self.trigger_policy = trigger_policy
        self.delta_risk = float(delta_risk)
        self.reference_quantile = float(reference_quantile)
        self.max_repairs_per_request = int(max_repairs_per_request)
        self.repair_target_rate = float(repair_target_rate)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dynamic-repair")
        self._lock = threading.Lock()
        self._futures: list[Future[dict[str, object]]] = []
        self._trace: list[dict[str, object]] = []
        self._real_tier_by_stage: dict[str, int] = {}
        self._latest_plan_version = 0
        self._applied_repairs = 0
        self._handled_observed_stages: set[str] = set()

    def submit(
        self, observation: DynamicObservation, apply_callback: ApplyCallback
    ) -> Future[dict[str, object]]:
        future = self._pool.submit(self._process, observation, apply_callback)
        with self._lock:
            self._futures.append(future)
        return future

    def mark_real_invoke(self, stage_name: str, tier: int) -> None:
        with self._lock:
            self._real_tier_by_stage[str(stage_name)] = int(tier)

    def wait(self) -> None:
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            future.result()

    def close(self) -> None:
        self.wait()
        self._pool.shutdown(wait=True)

    def trace_rows(self) -> list[dict[str, object]]:
        self.wait()
        with self._lock:
            rows = [dict(row) for row in self._trace]
            real_tiers = dict(self._real_tier_by_stage)
        for row in rows:
            changes = row.pop("_tier_changes", ())
            if not changes:
                row["upgraded_tier_used_by_real_invoke"] = ""
                continue
            row["upgraded_tier_used_by_real_invoke"] = all(
                real_tiers.get(stage_name) == to_tier
                for stage_name, _, to_tier in changes
            )
        return rows

    def _expected_stage_duration_ms(self, observation: DynamicObservation) -> float:
        stage = observation.completed_stage
        tier = int(observation.memory_by_stage[stage])
        duration = self.artifacts.dist(stage, tier, "warm").quantile(
            self.reference_quantile
        )
        if stage == self.workflow.entry and observation.entry_cold:
            duration += self.artifacts.entry_cold_overhead_dist(stage, tier).quantile(
                self.reference_quantile
            )
        return duration

    def _process(
        self,
        observation: DynamicObservation,
        apply_callback: ApplyCallback,
    ) -> dict[str, object]:
        started = time.perf_counter()
        risk_obs = conditional_warm_risk(
            workflow=self.workflow,
            artifacts=self.artifacts,
            memory_by_stage=observation.memory_by_stage,
            completed_finish_ms=observation.completed_finish_ms,
            slo_ms=self.slo_ms,
            stage_log_correlation=self.artifacts.stage_log_correlation,
        ).risk
        reference_finish = dict(observation.completed_finish_ms)
        reference_finish[observation.completed_stage] = (
            observation.stage_start_ms + self._expected_stage_duration_ms(observation)
        )
        risk_ref = conditional_warm_risk(
            workflow=self.workflow,
            artifacts=self.artifacts,
            memory_by_stage=observation.memory_by_stage,
            completed_finish_ms=reference_finish,
            slo_ms=self.slo_ms,
            stage_log_correlation=self.artifacts.stage_log_correlation,
        ).risk
        delta_r = risk_obs - risk_ref
        stale_observation = observation.plan_version != self._latest_plan_version
        already_handled = observation.completed_stage in self._handled_observed_stages
        repair_limit_reached = self._applied_repairs >= self.max_repairs_per_request
        delta_gate_passed = (
            self.trigger_policy == "absolute" or delta_r > self.delta_risk
        )
        triggered = (
            risk_obs > self.max_violation_rate
            and bool(observation.eligible_candidates)
            and delta_gate_passed
            and not stale_observation
            and not already_handled
            and not repair_limit_reached
        )
        repair: DynamicRepairResult | None = None
        applied = False
        rejected_reason = ""
        if triggered:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            remaining_ms = self.max_decision_ms - elapsed_ms
            if remaining_ms <= 0.0:
                rejected_reason = "decision_timeout_before_repair"
            else:
                repair = repair_up_only(
                    workflow=self.workflow,
                    artifacts=self.artifacts,
                    current_memory_by_stage=observation.memory_by_stage,
                    completed_finish_ms=observation.completed_finish_ms,
                    slo_ms=self.slo_ms,
                    tiers=self.tiers,
                    eligible_candidates=observation.eligible_candidates,
                    stage_log_correlation=self.artifacts.stage_log_correlation,
                    max_violation_rate=self.repair_target_rate,
                    max_decision_ms=remaining_ms,
                )
                if repair.timed_out:
                    rejected_reason = "decision_timeout"
                elif repair.post_risk > self.max_violation_rate:
                    rejected_reason = "no_feasible_repair"
                elif (time.perf_counter() - started) * 1000.0 > self.max_decision_ms:
                    rejected_reason = "decision_timeout"
                elif self.observe_only:
                    rejected_reason = "observe_only"
                else:
                    applied, rejected_reason = apply_callback(observation, repair)
                    if applied:
                        self._applied_repairs += 1
                        self._latest_plan_version = observation.plan_version + 1
                        self._handled_observed_stages.add(observation.completed_stage)

        wall_ms = (time.perf_counter() - started) * 1000.0
        if wall_ms > self.max_decision_ms and not applied and not rejected_reason:
            rejected_reason = "decision_timeout"
        changes = ()
        if repair is not None:
            changes = tuple(
                (stage_name, int(observation.memory_by_stage[stage_name]), int(final_tier))
                for stage_name, final_tier in repair.memory_by_stage.items()
                if int(final_tier) != int(observation.memory_by_stage[stage_name])
            )
        row: dict[str, object] = {
            "request_id": observation.request_id,
            "completed_stage": observation.completed_stage,
            "observation_kind": observation.observation_kind,
            "observed_stage": observation.completed_stage,
            "watchdog_quantile": (
                observation.watchdog_quantile
                if observation.watchdog_quantile is not None
                else ""
            ),
            "watchdog_deadline_ms": (
                observation.watchdog_deadline_ms
                if observation.watchdog_deadline_ms is not None
                else ""
            ),
            "stage_elapsed_ms": (
                observation.stage_elapsed_ms
                if observation.stage_elapsed_ms is not None
                else observation.elapsed_ms - observation.stage_start_ms
            ),
            "stage_completed_at_submit": observation.stage_completed_at_submit,
            "elapsed_ms": observation.elapsed_ms,
            "risk_ref": risk_ref,
            "risk_obs": risk_obs,
            "delta_r": delta_r,
            "trigger_policy": self.trigger_policy,
            "delta_risk_threshold": self.delta_risk,
            "reference_quantile": self.reference_quantile,
            "repair_target_rate": self.repair_target_rate,
            "repair_target_met": (
                repair.post_risk <= self.repair_target_rate
                if repair is not None
                else ""
            ),
            "delta_gate_passed": delta_gate_passed,
            "plan_version": observation.plan_version,
            "stale_observation": stale_observation,
            "already_handled": already_handled,
            "repair_limit_reached": repair_limit_reached,
            "repairs_applied_before": self._applied_repairs - int(applied),
            "triggered": triggered,
            "eligible_stages": ";".join(
                dict.fromkeys(candidate.stage_name for candidate in observation.eligible_candidates)
            ),
            "states_evaluated": repair.states_evaluated if repair is not None else 0,
            "decision_wall_ms": wall_ms,
            "pre_risk": repair.pre_risk if repair is not None else risk_obs,
            "post_risk": repair.post_risk if repair is not None else risk_obs,
            "tier_changes": ";".join(
                f"{stage}:{old}->{new}" for stage, old, new in changes
            ),
            "applied": applied,
            "rejected_reason": rejected_reason,
            "old_warmup_issued": observation.old_warmup_issued,
            "_tier_changes": changes,
        }
        with self._lock:
            self._trace.append(row)
        return row
