"""Execute a workflow DAG once and return per-stage measurement rows."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
import time
import uuid
from typing import Any

from serverless_dag.core.workflow import WorkflowNode, WorkflowSpec, suffix_action_name
from serverless_dag.resource.cpu import memory_to_cpu_cores
from serverless_dag.planner.dynamic_repair import jit_safe_upgrade_candidates
from serverless_dag.runtime.dynamic_controller import (
    DynamicObservation,
    DynamicRepairController,
)
from serverless_dag.runtime.jit_scheduler import JitScheduler, WarmupTask
from serverless_dag.runtime.jit_timing import (
    JitLatencyTables,
    compute_predicted_times,
    warmup_fire_time,
)
from serverless_dag.runtime.openwhisk import OpenWhiskClient, activation_annotations


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def to_float_or_none(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latency_fields(
    dispatch_start_ms: int,
    dispatch_end_ms: int,
    action_duration_ms: object = "",
    ow_wait_ms: object = "",
    ow_init_ms: object = "",
    ow_duration_ms: object = "",
) -> dict[str, object]:
    dispatch_latency_ms = dispatch_end_ms - dispatch_start_ms
    action_duration = to_float_or_none(action_duration_ms)
    wait = to_float_or_none(ow_wait_ms)
    init = to_float_or_none(ow_init_ms)
    ow_duration = to_float_or_none(ow_duration_ms)
    platform_overhead_ms = (
        dispatch_latency_ms - action_duration
        if action_duration is not None
        else ""
    )
    ow_runtime_overhead_ms = (
        ow_duration - (init or 0.0) - action_duration
        if ow_duration is not None and action_duration is not None
        else ""
    )
    client_gateway_overhead_ms = (
        dispatch_latency_ms - wait - ow_duration
        if wait is not None and ow_duration is not None
        else ""
    )
    return {
        "dispatch_latency_ms": dispatch_latency_ms,
        "platform_overhead_ms": platform_overhead_ms,
        "ow_runtime_overhead_ms": ow_runtime_overhead_ms,
        "client_gateway_overhead_ms": client_gateway_overhead_ms,
    }


def invoke_node(
    client: OpenWhiskClient,
    workflow: WorkflowSpec,
    node: WorkflowNode,
    request_id: str,
    entry_ts_ms: int,
    parent_results: dict[str, dict[str, Any]],
    allocated_memory_mb: int | None = None,
    allocated_cpu_cores: float | None = None,
    action_name: str | None = None,
    slo_class: str | None = None,
    reservation_key: str | None = None,
    action_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke one workflow stage and normalize activation/result metadata."""

    resolved_action_name = action_name or node.action
    dispatch_start_ms = now_ms()
    real_invoke_monotonic = time.monotonic()
    params: dict[str, Any] = {
        "workflow_name": workflow.workflow_name,
        "request_id": request_id,
        "entry_ts_ms": entry_ts_ms,
        "stage_name": node.name,
        "parent_stages": list(node.parents),
        "allocated_memory_mb": allocated_memory_mb,
        "allocated_cpu_cores": allocated_cpu_cores,
        "payload": {
            "parents": {
                parent: parent_results.get(parent, {})
                for parent in node.parents
            }
        },
    }
    if reservation_key:
        params["__ow_reservation_key"] = reservation_key
    if action_params:
        reserved = sorted(set(params).intersection(action_params))
        if reserved:
            raise ValueError(
                "per-stage action params cannot override executor fields: "
                + ", ".join(reserved)
            )
        params.update(action_params)
    try:
        activation = client.invoke_activation(resolved_action_name, params)
        dispatch_end_ms = now_ms()
        response = activation.get("response", {})
        result = response.get("result", {}) if isinstance(response, dict) else {}
        if not isinstance(result, dict):
            result = {"error": result}
        annotations = activation_annotations(activation)
        action_duration_ms = result.get("action_duration_ms", "")
        ow_wait_ms = annotations.get("waitTime", "")
        ow_init_ms = annotations.get("initTime", 0)
        ow_duration_ms = activation.get("duration", "")
        limits = annotations.get("limits", {})
        ow_memory_mb = limits.get("memory", "") if isinstance(limits, dict) else ""
        activation_status = response.get("status", "") if isinstance(response, dict) else ""
        activation_error = result.get("error", "") if isinstance(result, dict) else ""

        row_status = "ok"
        row_error = ""
        if not response:
            row_status = "error"
            row_error = "activation did not return a completed response"
        elif activation_status and activation_status != "success":
            row_status = "error"
            row_error = f"activation status={activation_status}: {activation_error}"
        elif action_duration_ms in ("", None):
            row_status = "error"
            row_error = "activation result is missing action_duration_ms"

        return {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": node.name,
            "parent_stages": ",".join(node.parents),
            "slo_class": slo_class or "",
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": dispatch_start_ms,
            "dispatch_end_ms": dispatch_end_ms,
            "real_invoke_monotonic": real_invoke_monotonic,
            "resolved_action_name": resolved_action_name,
            **latency_fields(
                dispatch_start_ms,
                dispatch_end_ms,
                action_duration_ms,
                ow_wait_ms,
                ow_init_ms,
                ow_duration_ms,
            ),
            "action_start_ns": result.get("action_start_ns", ""),
            "action_end_ns": result.get("action_end_ns", ""),
            "action_duration_ms": action_duration_ms,
            "container_id": result.get("container_id", ""),
            "container_invocation_index": result.get("container_invocation_index", ""),
            "container_uptime_ms": result.get("container_uptime_ms", ""),
            "previous_action_end_ns": result.get("previous_action_end_ns", ""),
            "idle_since_prev_ms": result.get("idle_since_prev_ms", ""),
            "cold_like": result.get("cold_like", ""),
            "pod_name": result.get("pod_name", ""),
            "activation_id": activation.get("activationId", ""),
            "action_version": activation.get("version", ""),
            "ow_cold_start": "initTime" in annotations,
            "ow_memory_mb": ow_memory_mb,
            "allocated_memory_mb": result.get(
                "allocated_memory_mb", allocated_memory_mb or ""
            ),
            "allocated_cpu_cores": result.get(
                "allocated_cpu_cores", allocated_cpu_cores or ""
            ),
            "detected_cpu_cores": result.get("detected_cpu_cores", ""),
            "ow_wait_ms": ow_wait_ms,
            "ow_init_ms": ow_init_ms,
            "ow_duration_ms": ow_duration_ms,
            "cpu_user_ms": result.get("cpu_user_ms", ""),
            "cpu_system_ms": result.get("cpu_system_ms", ""),
            "cpu_self_ms": result.get("cpu_self_ms", ""),
            "cpu_self_process_ms": result.get("cpu_self_process_ms", ""),
            "cpu_children_user_ms": result.get("cpu_children_user_ms", ""),
            "cpu_children_system_ms": result.get("cpu_children_system_ms", ""),
            "cpu_children_ms": result.get("cpu_children_ms", ""),
            "cpu_process_ms": result.get("cpu_process_ms", ""),
            "cpu_total_ms": result.get("cpu_total_ms", ""),
            "parallel_cpu_ms": result.get("parallel_cpu_ms", ""),
            "observed_effective_cores": result.get("observed_effective_cores", ""),
            "observed_parallel_cores": result.get("observed_parallel_cores", ""),
            "mem_rss_kb": result.get("mem_rss_kb", ""),
            "mem_peak_kb": result.get("mem_peak_kb", ""),
            "status": row_status,
            "error": row_error,
            "_result": result,
        }
    except Exception as exc:
        dispatch_end_ms = now_ms()
        return {
            "workflow_name": workflow.workflow_name,
            "request_id": request_id,
            "stage_name": node.name,
            "parent_stages": ",".join(node.parents),
            "slo_class": slo_class or "",
            "entry_ts_ms": entry_ts_ms,
            "dispatch_start_ms": dispatch_start_ms,
            "dispatch_end_ms": dispatch_end_ms,
            "real_invoke_monotonic": real_invoke_monotonic,
            "resolved_action_name": resolved_action_name,
            **latency_fields(dispatch_start_ms, dispatch_end_ms),
            "allocated_memory_mb": allocated_memory_mb or "",
            "allocated_cpu_cores": allocated_cpu_cores or "",
            "status": "error",
            "error": str(exc),
            "_result": {},
        }


def status_value(status: object, key: str, default: object = "") -> object:
    if isinstance(status, dict):
        return status.get(key, default)
    return getattr(status, key, default)


def run_one_workflow(
    workflow: WorkflowSpec,
    client: OpenWhiskClient,
    max_workers: int,
    allocated_memory_mb: int | None = None,
    allocated_cpu_cores: float | None = None,
    raise_on_error: bool = True,
    action_name_by_stage: dict[str, str] | None = None,
    slo_class: str | None = None,
    memory_by_stage: dict[str, int] | None = None,
    enable_jit: bool = False,
    jit_scheduler: JitScheduler | None = None,
    jit_latency_tables: JitLatencyTables | None = None,
    jit_margin_ms: float = 0.0,
    jit_fire_settle_ms: float = 0.0,
    enable_jit_sync: bool = False,
    jit_warmup_tracker: object | None = None,
    jit_sync_pause_grace_ms: float = 0.0,
    jit_sync_inflight_max_ms: float = 6000.0,
    enable_dynamic_repair: bool = False,
    dynamic_controller: DynamicRepairController | None = None,
    dynamic_trigger_mode: str = "completion",
    dynamic_overrun_quantile: float = 0.95,
    request_id: str | None = None,
    action_params_by_stage: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run ``workflow`` once with OpenWhisk blocking invokes."""

    request_id = str(request_id) if request_id else str(uuid.uuid4())
    entry_ts_ms = now_ms()
    workflow_start_monotonic = time.monotonic()
    workflow_start_ms = entry_ts_ms
    completed: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    running: dict[Any, str] = {}
    started_at: dict[str, float] = {}
    measured_completion_at: dict[str, float] = {}
    watchdog_timers: dict[str, threading.Timer] = {}
    dynamic_applied_stages: set[str] = set()
    active_plan_version = 0
    jit_current_fire_times: dict[str, float] = {}
    runtime_lock = threading.RLock()
    active_memory_by_stage = (
        dict(memory_by_stage)
        if enable_dynamic_repair and memory_by_stage is not None
        else memory_by_stage
    )
    entry_cold_observed = False
    jit_scheduled_count = 0
    jit_upsert_count = 0
    jit_late_count = 0
    jit_active = bool(
        enable_jit
        and jit_scheduler is not None
        and jit_latency_tables is not None
        and memory_by_stage is not None
    )
    jit_sync_active = bool(jit_active and enable_jit_sync and jit_warmup_tracker is not None)
    if dynamic_trigger_mode not in {"completion", "overrun"}:
        raise ValueError("dynamic_trigger_mode must be 'completion' or 'overrun'")
    if not 0.0 < float(dynamic_overrun_quantile) < 1.0:
        raise ValueError("dynamic_overrun_quantile must be in (0, 1)")

    def stage_memory(stage_name: str) -> int | None:
        if active_memory_by_stage is not None:
            with runtime_lock:
                return int(active_memory_by_stage[stage_name])
        return allocated_memory_mb

    def stage_cpu(stage_name: str) -> float | None:
        memory_mb = stage_memory(stage_name)
        if memory_mb is not None:
            return memory_to_cpu_cores(memory_mb)
        return allocated_cpu_cores

    def stage_action(stage_name: str) -> str:
        if action_name_by_stage and stage_name in action_name_by_stage:
            return action_name_by_stage[stage_name]
        memory_mb = stage_memory(stage_name)
        action = workflow.nodes[stage_name].action
        return suffix_action_name(action, f"_{memory_mb}") if memory_mb is not None else action

    def predicted_times() -> tuple[dict[str, float], dict[str, float]]:
        if active_memory_by_stage is None or jit_latency_tables is None:
            raise RuntimeError("JIT prediction requires memory_by_stage and jit_latency_tables")
        with runtime_lock:
            return compute_predicted_times(
                workflow=workflow,
                memory_by_stage=dict(active_memory_by_stage),
                latency_tables=jit_latency_tables,
                workflow_start_monotonic=workflow_start_monotonic,
                started_at=dict(started_at),
                measured_completion_at=dict(measured_completion_at),
            )

    def schedule_stage_warmup(stage_name: str, needed_at: float, phase: str) -> None:
        nonlocal jit_scheduled_count, jit_upsert_count, jit_late_count
        if not jit_active or active_memory_by_stage is None or jit_latency_tables is None:
            return

        now = time.monotonic()
        existing_fire_time = jit_current_fire_times.get(stage_name)
        if phase == "upsert" and existing_fire_time is not None and existing_fire_time <= now:
            return

        tier = int(stage_memory(stage_name))
        warmup_lead_ms = jit_latency_tables.jit_lead_ms(stage_name, tier)
        raw_fire_time = warmup_fire_time(
            needed_at=needed_at,
            cold_overhead_ms=warmup_lead_ms,
            margin_ms=jit_margin_ms,
            settle_ms=jit_fire_settle_ms,
        )
        late_jit = raw_fire_time <= now
        fire_time = now if late_jit else raw_fire_time
        if late_jit:
            jit_late_count += 1
        if phase == "initial":
            jit_scheduled_count += 1
        else:
            jit_upsert_count += 1

        task = WarmupTask(
            task_key=f"{request_id}:{stage_name}:{tier}",
            fire_time=fire_time,
            action_name=stage_action(stage_name),
            metadata={
                "request_id": request_id,
                "workflow_name": workflow.workflow_name,
                "stage_name": stage_name,
                "tier_mb": tier,
                "cpu_cores": stage_cpu(stage_name) or "",
                "slo_class": slo_class or "",
                "needed_at": needed_at,
                "raw_fire_time": raw_fire_time,
                "fire_time": fire_time,
                "late_jit": late_jit,
                "jit_margin_ms": jit_margin_ms,
                "jit_fire_settle_ms": jit_fire_settle_ms,
                "jit_warmup_lead_ms": warmup_lead_ms,
                "cold_overhead_ms": jit_latency_tables.cold_ms(stage_name, tier),
                "parents": list(workflow.nodes[stage_name].parents),
                "schedule_phase": phase,
            },
        )
        jit_current_fire_times[stage_name] = fire_time
        jit_scheduler.schedule(task)

    def enqueue_initial_jit_warmups() -> None:
        if not jit_active:
            return
        predicted_start, _ = predicted_times()
        for stage_name in workflow.topological_order():
            if stage_name == workflow.entry:
                continue
            schedule_stage_warmup(stage_name, predicted_start[stage_name], "initial")

    def upsert_pending_jit_warmups() -> None:
        if not jit_active:
            return
        predicted_start, _ = predicted_times()
        for stage_name in workflow.topological_order():
            if stage_name == workflow.entry:
                continue
            if stage_name in completed or stage_name in running.values() or stage_name in started_at:
                continue
            schedule_stage_warmup(stage_name, predicted_start[stage_name], "upsert")

    def wait_for_stage_warmup(stage_name: str) -> dict[str, Any]:
        sync_info: dict[str, Any] = {
            "jit_sync_enabled": jit_sync_active,
            "jit_sync_waited_ms": 0.0,
            "jit_sync_dispatch_after_warmup": False,
            "jit_sync_status": "not_applicable",
            "jit_sync_warmup_issued_monotonic": "",
            "jit_sync_warmup_completed_monotonic": "",
            "jit_warmup_completed_to_real_invoke_ms": "",
        }
        if not jit_sync_active or stage_name == workflow.entry or active_memory_by_stage is None:
            return sync_info

        start_wait = time.monotonic()
        tier = int(stage_memory(stage_name))
        max_wait_s = max(0.0, jit_latency_tables.jit_lead_ms(stage_name, tier) / 1000.0)
        pause_grace_s = max(0.0, jit_sync_pause_grace_ms / 1000.0)
        inflight_extension_s = max(0.0, jit_sync_inflight_max_ms / 1000.0)
        completion_deadline = start_wait + max_wait_s
        extension_deadline = completion_deadline + inflight_extension_s
        deadline = extension_deadline + pause_grace_s

        status = jit_warmup_tracker.get_status(request_id, stage_name, tier)
        issued = status_value(status, "issued_monotonic", "")
        completed_time = status_value(status, "completed_monotonic", "")
        if issued not in ("", None):
            sync_info["jit_sync_warmup_issued_monotonic"] = issued
            sync_info["jit_sync_status"] = "in_flight"
        else:
            sync_info["jit_sync_status"] = "not_issued"

        if completed_time in ("", None):
            remaining = max(0.0, completion_deadline - time.monotonic())
            if remaining > 0.0:
                status = jit_warmup_tracker.wait_until_completed(
                    request_id, stage_name, tier, remaining
                )
                issued = status_value(status, "issued_monotonic", issued)
                completed_time = status_value(status, "completed_monotonic", completed_time)

        completed_after_extension = False
        if completed_time in ("", None) and issued not in ("", None):
            remaining = max(0.0, extension_deadline - time.monotonic())
            if remaining > 0.0:
                status = jit_warmup_tracker.wait_until_completed(
                    request_id, stage_name, tier, remaining
                )
                issued = status_value(status, "issued_monotonic", issued)
                completed_time = status_value(status, "completed_monotonic", completed_time)
                completed_after_extension = completed_time not in ("", None)

        if issued not in ("", None):
            sync_info["jit_sync_warmup_issued_monotonic"] = issued
        if completed_time not in ("", None):
            sync_info["jit_sync_warmup_completed_monotonic"] = completed_time
            sync_info["jit_sync_dispatch_after_warmup"] = True
            sync_info["jit_sync_status"] = (
                "completed_after_extension" if completed_after_extension else "completed"
            )
            ready_at = float(completed_time) + pause_grace_s
            remaining_grace = ready_at - time.monotonic()
            remaining_budget = deadline - time.monotonic()
            if remaining_grace > 0.0 and remaining_budget > 0.0:
                time.sleep(min(remaining_grace, remaining_budget))
        elif issued not in ("", None):
            sync_info["jit_sync_status"] = "timed_out_hang"
        else:
            sync_info["jit_sync_status"] = "timed_out_not_issued"

        sync_info["jit_sync_waited_ms"] = (time.monotonic() - start_wait) * 1000.0
        return sync_info

    enqueue_initial_jit_warmups()

    def submit_dynamic_observation(
        stage_name: str,
        row: dict[str, Any] | None,
        observation_monotonic: float,
        *,
        observation_kind: str = "completion",
        watchdog_deadline_ms: float | None = None,
    ) -> Any:
        nonlocal entry_cold_observed
        if (
            not enable_dynamic_repair
            or dynamic_controller is None
            or active_memory_by_stage is None
            or jit_latency_tables is None
            or jit_scheduler is None
            or jit_warmup_tracker is None
        ):
            return None
        if stage_name == workflow.entry and row is not None:
            entry_cold_observed = str(row.get("cold_like", "")).lower() == "true"

        with runtime_lock:
            memory_snapshot = dict(active_memory_by_stage)
            completed_finish_ms = {
                name: (value - workflow_start_monotonic) * 1000.0
                for name, value in measured_completion_at.items()
            }
            stage_completed_at_submit = stage_name in measured_completion_at
            if observation_kind == "overrun" and not stage_completed_at_submit:
                completed_finish_ms[stage_name] = (
                    observation_monotonic - workflow_start_monotonic
                ) * 1000.0
            now_ms_since_start = (
                observation_monotonic - workflow_start_monotonic
            ) * 1000.0
            started_stages = set(started_at)
            issued_stages: set[str] = set()
            pending_stages: set[str] = set()
            for pending_stage in workflow.topological_order():
                if pending_stage == workflow.entry or pending_stage in completed_finish_ms:
                    continue
                tier = int(memory_snapshot[pending_stage])
                task_key = f"{request_id}:{pending_stage}:{tier}"
                status = jit_warmup_tracker.get_status(
                    request_id, pending_stage, tier
                )
                if status_value(status, "issued_monotonic", "") not in ("", None):
                    issued_stages.add(pending_stage)
                if jit_scheduler.is_pending(task_key):
                    pending_stages.add(pending_stage)
            candidates = jit_safe_upgrade_candidates(
                workflow=workflow,
                artifacts=dynamic_controller.artifacts,
                memory_by_stage=memory_snapshot,
                completed_finish_ms=completed_finish_ms,
                started_stages=started_stages,
                old_warmup_issued_stages=issued_stages,
                old_warmup_pending_stages=pending_stages,
                now_ms=now_ms_since_start,
                tiers=dynamic_controller.tiers,
            )
            if observation_kind == "overrun":
                descendants = set(workflow.descendants_of(stage_name))
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if candidate.stage_name in descendants
                    and candidate.stage_name not in dynamic_applied_stages
                )
            else:
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if candidate.stage_name not in dynamic_applied_stages
                )
            observation = DynamicObservation(
                request_id=request_id,
                completed_stage=stage_name,
                elapsed_ms=now_ms_since_start,
                stage_start_ms=(
                    started_at[stage_name] - workflow_start_monotonic
                )
                * 1000.0,
                entry_cold=entry_cold_observed,
                completed_finish_ms=completed_finish_ms,
                memory_by_stage=memory_snapshot,
                eligible_candidates=candidates,
                old_warmup_issued=bool(issued_stages),
                observation_kind=observation_kind,
                watchdog_quantile=(
                    float(dynamic_overrun_quantile)
                    if observation_kind == "overrun"
                    else None
                ),
                watchdog_deadline_ms=watchdog_deadline_ms,
                stage_elapsed_ms=(
                    observation_monotonic - started_at[stage_name]
                ) * 1000.0,
                stage_completed_at_submit=stage_completed_at_submit,
                plan_version=active_plan_version,
            )

        def apply_dynamic_repair(observation, repair) -> tuple[bool, str]:
            nonlocal active_plan_version
            final_changes = [
                (
                    candidate_stage,
                    int(observation.memory_by_stage[candidate_stage]),
                    int(repair.memory_by_stage[candidate_stage]),
                )
                for candidate_stage in workflow.topological_order()
                if int(repair.memory_by_stage[candidate_stage])
                != int(observation.memory_by_stage[candidate_stage])
            ]
            if not final_changes:
                return False, "no_tier_change"
            with runtime_lock:
                if observation.plan_version != active_plan_version:
                    return False, "stale_plan_version"
                now = time.monotonic()
                current_completed_finish_ms = {
                    name: (value - workflow_start_monotonic) * 1000.0
                    for name, value in measured_completion_at.items()
                }
                current_started = set(started_at)
                current_memory = dict(active_memory_by_stage)
                old_keys: list[str] = []
                issued_now: set[str] = set()
                pending_now: set[str] = set()
                for changed_stage, old_tier, _ in final_changes:
                    if changed_stage in dynamic_applied_stages:
                        return False, "already_upgraded_by_dynamic"
                    if changed_stage in current_started or changed_stage in completed:
                        return False, "stage_started_during_decision"
                    if int(current_memory[changed_stage]) != old_tier:
                        return False, "stale_plan"
                    status = jit_warmup_tracker.get_status(
                        request_id, changed_stage, old_tier
                    )
                    if status_value(status, "issued_monotonic", "") not in ("", None):
                        issued_now.add(changed_stage)
                    task_key = f"{request_id}:{changed_stage}:{old_tier}"
                    old_keys.append(task_key)
                    if jit_scheduler.is_pending(task_key):
                        pending_now.add(changed_stage)
                if issued_now:
                    return False, "old_warmup_issued_during_decision"

                latest_candidates = jit_safe_upgrade_candidates(
                    workflow=workflow,
                    artifacts=dynamic_controller.artifacts,
                    memory_by_stage=current_memory,
                    completed_finish_ms=current_completed_finish_ms,
                    started_stages=current_started,
                    old_warmup_issued_stages=issued_now,
                    old_warmup_pending_stages=pending_now,
                    now_ms=(now - workflow_start_monotonic) * 1000.0,
                    tiers=dynamic_controller.tiers,
                )
                allowed = {
                    (candidate.stage_name, candidate.to_tier)
                    for candidate in latest_candidates
                }
                if any(
                    (changed_stage, new_tier) not in allowed
                    for changed_stage, _, new_tier in final_changes
                ):
                    return False, "warmup_deadline_missed_during_decision"
                if not jit_scheduler.cancel_many_if_pending(old_keys):
                    return False, "old_warmup_no_longer_queued"

                for changed_stage, _, new_tier in final_changes:
                    active_memory_by_stage[changed_stage] = int(new_tier)
                    dynamic_applied_stages.add(changed_stage)
                    jit_current_fire_times.pop(changed_stage, None)
                predicted_start, _ = predicted_times()
                for changed_stage, _, _ in final_changes:
                    schedule_stage_warmup(
                        changed_stage, predicted_start[changed_stage], "dynamic"
                    )
                active_plan_version += 1
            return True, ""

        return dynamic_controller.submit(observation, apply_dynamic_repair)

    def schedule_overrun_watchdog(stage_name: str, start_monotonic: float) -> None:
        if (
            not enable_dynamic_repair
            or dynamic_trigger_mode != "overrun"
            or dynamic_controller is None
            or active_memory_by_stage is None
            or not workflow.children_of(stage_name)
        ):
            return
        tier = int(stage_memory(stage_name))
        duration_ms = dynamic_controller.artifacts.dist(
            stage_name, tier, "warm"
        ).quantile(float(dynamic_overrun_quantile))
        deadline_monotonic = start_monotonic + duration_ms / 1000.0
        deadline_ms = (deadline_monotonic - workflow_start_monotonic) * 1000.0

        def arm(delay_sec: float) -> None:
            timer = threading.Timer(max(0.0, delay_sec), fire)
            timer.daemon = True
            with runtime_lock:
                watchdog_timers[stage_name] = timer
            timer.start()

        def fire() -> None:
            with runtime_lock:
                if stage_name in measured_completion_at:
                    return
            future = submit_dynamic_observation(
                stage_name,
                None,
                time.monotonic(),
                observation_kind="overrun",
                watchdog_deadline_ms=deadline_ms,
            )
            if future is None:
                return

            def rearm_if_needed(done_future: Any) -> None:
                result = done_future.result()
                if bool(result.get("triggered")):
                    return
                with runtime_lock:
                    if stage_name in measured_completion_at:
                        return
                # P95 is the first check. If the stage remains in flight but
                # the absolute DAG risk has not crossed 5%, recheck at a small
                # cadence until completion or the first trigger.
                arm(0.100)

            future.add_done_callback(rearm_if_needed)

        arm(deadline_monotonic - time.monotonic())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while len(completed) < len(workflow.nodes):
            started_any = False
            for node in workflow.ready_nodes(completed.keys(), running.values()):
                memory_mb = stage_memory(node.name)
                cpu_cores = stage_cpu(node.name)
                action_name = stage_action(node.name)
                reservation_key = f"{request_id}:{node.name}" if jit_active else None
                with runtime_lock:
                    started_at[node.name] = time.monotonic()
                    stage_started_monotonic = started_at[node.name]
                    started_any = True
                schedule_overrun_watchdog(node.name, stage_started_monotonic)
                if dynamic_controller is not None and memory_mb is not None:
                    dynamic_controller.mark_real_invoke(node.name, int(memory_mb))

                def invoke_with_optional_sync(
                    ready_node: WorkflowNode = node,
                    ready_memory_mb: int | None = memory_mb,
                    ready_cpu_cores: float | None = cpu_cores,
                    ready_action_name: str = action_name,
                    ready_reservation_key: str | None = reservation_key,
                ) -> dict[str, Any]:
                    sync_info = wait_for_stage_warmup(ready_node.name)
                    row = invoke_node(
                        client,
                        workflow,
                        ready_node,
                        request_id,
                        entry_ts_ms,
                        completed,
                        ready_memory_mb,
                        ready_cpu_cores,
                        ready_action_name,
                        slo_class,
                        ready_reservation_key,
                        (
                            action_params_by_stage.get(ready_node.name)
                            if action_params_by_stage
                            else None
                        ),
                    )
                    completed_monotonic = sync_info.get(
                        "jit_sync_warmup_completed_monotonic", ""
                    )
                    real_invoke = row.get("real_invoke_monotonic", "")
                    if completed_monotonic not in ("", None) and real_invoke not in ("", None):
                        sync_info["jit_warmup_completed_to_real_invoke_ms"] = (
                            float(real_invoke) - float(completed_monotonic)
                        ) * 1000.0
                    row.update(sync_info)
                    row["stage_start_monotonic"] = started_at.get(ready_node.name, "")
                    return row

                future = pool.submit(
                    invoke_with_optional_sync,
                )
                running[future] = node.name

            if started_any and jit_active:
                with runtime_lock:
                    upsert_pending_jit_warmups()

            if not running:
                missing = sorted(set(workflow.nodes) - set(completed))
                raise RuntimeError(f"workflow made no progress; remaining={missing}")

            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                stage_name = running.pop(future)
                row = future.result()
                rows.append(row)
                completion_monotonic = time.monotonic()
                with runtime_lock:
                    completed[stage_name] = row.get("_result", {})
                    measured_completion_at[stage_name] = completion_monotonic
                    timer = watchdog_timers.pop(stage_name, None)
                    if timer is not None:
                        timer.cancel()
                if raise_on_error and row.get("status") != "ok":
                    raise RuntimeError(
                        f"stage {stage_name} failed: {row.get('error', '')}"
                    )
                submit_dynamic_observation(stage_name, row, completion_monotonic)
                with runtime_lock:
                    upsert_pending_jit_warmups()

    for timer in watchdog_timers.values():
        timer.cancel()
    if dynamic_controller is not None:
        dynamic_controller.wait()

    workflow_end_ms = max(int(float(row["dispatch_end_ms"])) for row in rows)
    workflow_e2e_ms = workflow_end_ms - workflow_start_ms
    topo_index = {name: index for index, name in enumerate(workflow.topological_order())}
    for row in rows:
        row["workflow_start_ms"] = workflow_start_ms
        row["workflow_end_ms"] = workflow_end_ms
        row["workflow_e2e_ms"] = workflow_e2e_ms
        row["jit_enabled"] = jit_active
        row["jit_scheduled_count"] = jit_scheduled_count
        row["jit_upsert_count"] = jit_upsert_count
        row["jit_late_count"] = jit_late_count
        row.pop("_result", None)
    return sorted(rows, key=lambda row: topo_index.get(str(row["stage_name"]), 10**9))
