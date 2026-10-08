"""Memory-to-CPU profiles used by deployment and measurement scripts."""

from __future__ import annotations

import math


def memory_to_cpu_cores(
    memory_mb: int | float,
    profile: str = "huawei_functiongraph",
    cpu_per_memory_mb: float | None = None,
) -> float:
    """Return the configured CPU limit for an OpenWhisk action memory tier.

    The default matches the Huawei FunctionGraph-like grid used by the current
    experiments: every 256 MiB adds 200 millicpu, capped at 3.2 cores.
    """

    memory = float(memory_mb)
    if memory <= 0.0:
        raise ValueError(f"memory_mb must be positive, got {memory_mb!r}")

    normalized = profile.strip().lower()
    if normalized in {"huawei_functiongraph", "huawei", "functiongraph"}:
        steps = math.ceil(memory / 256.0)
        return min(3200.0, 200.0 * steps) / 1000.0
    if normalized in {"legacy_256mb_250m", "openwhisk_256mb_250m"}:
        return memory / 256.0 * 0.25
    if normalized == "custom":
        if cpu_per_memory_mb is None or cpu_per_memory_mb <= 0.0:
            raise ValueError("--cpu-per-memory-mb is required for custom CPU profile")
        return memory * float(cpu_per_memory_mb)
    raise ValueError(f"unknown CPU profile: {profile!r}")
