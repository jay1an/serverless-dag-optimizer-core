"""Warmup invoke helpers and completion tracking for runtime JIT."""

from __future__ import annotations

import threading
import time
from typing import Any

from serverless_dag.runtime.jit_scheduler import WarmupTask
from serverless_dag.runtime.openwhisk import OpenWhiskClient, activation_annotations


class WarmupStatusTracker:
    """Track warmup issue/completion timestamps by request, stage, and tier."""

    def __init__(self):
        self.condition = threading.Condition()
        self.statuses: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _status_locked(self, request_id: str, stage_name: str, tier: object) -> dict[str, Any]:
        key = (str(request_id), str(stage_name), str(tier))
        if key not in self.statuses:
            self.statuses[key] = {
                "issued_monotonic": "",
                "completed_monotonic": "",
                "activation_id": "",
                "container_id": "",
                "error": "",
                "event": threading.Event(),
            }
        return self.statuses[key]

    def mark_issued(
        self, request_id: str, stage_name: str, tier: object, issued_monotonic: float
    ) -> None:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            status["issued_monotonic"] = issued_monotonic
            self.condition.notify_all()

    def mark_completed(
        self,
        request_id: str,
        stage_name: str,
        tier: object,
        completed_monotonic: float,
        activation_id: str = "",
        container_id: str = "",
        error: str = "",
    ) -> None:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            status["completed_monotonic"] = completed_monotonic
            status["activation_id"] = activation_id
            status["container_id"] = container_id
            status["error"] = error
            status["event"].set()
            self.condition.notify_all()

    def get_status(self, request_id: str, stage_name: str, tier: object) -> dict[str, Any]:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            return {key: value for key, value in status.items() if key != "event"}

    def wait_until_completed(
        self,
        request_id: str,
        stage_name: str,
        tier: object,
        timeout: float,
    ) -> dict[str, Any]:
        with self.condition:
            status = self._status_locked(request_id, stage_name, tier)
            event = status["event"]
        event.wait(timeout=max(0.0, timeout))
        return self.get_status(request_id, stage_name, tier)


class WarmupRecorder:
    """Asynchronously invoke warmup tasks and retain measurement records."""

    def __init__(
        self,
        client: OpenWhiskClient,
        tracker: WarmupStatusTracker | None = None,
        reserve_containers: bool = True,
    ):
        self.client = client
        self.tracker = tracker
        self.reserve_containers = reserve_containers
        self.condition = threading.Condition()
        self.records: list[dict[str, Any]] = []
        self.active_count = 0

    def callback(self, task: WarmupTask) -> None:
        callback_start = time.monotonic()
        with self.condition:
            self.active_count += 1
        worker = threading.Thread(
            target=self._invoke_warmup,
            args=(task, callback_start),
            daemon=True,
        )
        worker.start()

    def _invoke_warmup(self, task: WarmupTask, callback_start: float) -> None:
        request_id = str(task.metadata.get("request_id", ""))
        stage_name = str(task.metadata.get("stage_name", ""))
        tier = task.metadata.get("tier_mb", "")
        params: dict[str, Any] = {
            "__warmup": True,
            "request_id": request_id,
            "workflow_name": task.metadata.get("workflow_name", ""),
            "stage_name": stage_name,
            "allocated_memory_mb": tier,
            "allocated_cpu_cores": task.metadata.get("cpu_cores", ""),
        }
        if self.reserve_containers and request_id and stage_name:
            params["__ow_reservation_key"] = f"{request_id}:{stage_name}"

        sent = time.monotonic()
        if self.tracker is not None and request_id and stage_name:
            self.tracker.mark_issued(request_id, stage_name, tier, sent)

        activation: dict[str, Any] = {}
        result: dict[str, Any] = {}
        annotations: dict[str, Any] = {}
        status = "ok"
        error = ""
        try:
            activation = self.client.invoke_activation(task.action_name, params)
            response = activation.get("response", {})
            result = response.get("result", {}) if isinstance(response, dict) else {}
            if not isinstance(result, dict):
                result = {"error": result}
            annotations = activation_annotations(activation)
            activation_status = response.get("status", "") if isinstance(response, dict) else ""
            if not response:
                status = "error"
                error = "activation did not return a completed response"
            elif activation_status and activation_status != "success":
                status = "error"
                error = str(result.get("error", ""))
        except Exception as exc:
            status = "error"
            error = str(exc)

        completed = time.monotonic()
        dispatch_latency_ms = (completed - sent) * 1000.0
        action_duration_ms = result.get("action_duration_ms", "")
        try:
            platform_overhead_ms: float | str = dispatch_latency_ms - float(
                action_duration_ms
            )
        except (TypeError, ValueError):
            platform_overhead_ms = ""
        if self.tracker is not None and request_id and stage_name:
            self.tracker.mark_completed(
                request_id,
                stage_name,
                tier,
                completed,
                activation_id=str(activation.get("activationId", "")),
                container_id=str(result.get("container_id", "")),
                error=error,
            )

        record = {
            "task_key": task.task_key,
            "request_id": request_id,
            "stage_name": stage_name,
            "action_name": task.action_name,
            "tier_mb": tier,
            "cpu_cores": task.metadata.get("cpu_cores", ""),
            "scheduled_fire_time": task.fire_time,
            "raw_fire_time": task.metadata.get("raw_fire_time", ""),
            "needed_at": task.metadata.get("needed_at", ""),
            "late_jit": task.metadata.get("late_jit", ""),
            "schedule_phase": task.metadata.get("schedule_phase", ""),
            "callback_start": callback_start,
            "warmup_sent_monotonic": sent,
            "callback_end": completed,
            "dispatch_latency_ms": dispatch_latency_ms,
            "activation_id": activation.get("activationId", ""),
            "activation_duration_ms": activation.get("duration", ""),
            "action_duration_ms": action_duration_ms,
            "platform_overhead_ms": platform_overhead_ms,
            "cold_like": result.get("cold_like", ""),
            "container_id": result.get("container_id", ""),
            "ow_init_ms": annotations.get("initTime", ""),
            "ow_wait_ms": annotations.get("waitTime", ""),
            "status": status,
            "error": error,
        }
        with self.condition:
            self.records.append(record)
            self.active_count -= 1
            self.condition.notify_all()

    def snapshot(self) -> list[dict[str, Any]]:
        with self.condition:
            return list(self.records)

    def wait_for_request_count(
        self,
        request_id: str,
        expected_count: int,
        timeout: float = 90.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                count = sum(1 for record in self.records if record["request_id"] == request_id)
                if count >= expected_count:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"only saw {count}/{expected_count} warmups for request_id={request_id}"
                    )
                self.condition.wait(timeout=remaining)

    def wait_for_idle(self, timeout: float = 90.0) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.active_count > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"{self.active_count} warmups still in flight")
                self.condition.wait(timeout=remaining)
