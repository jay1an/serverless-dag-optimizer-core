"""Provider-style execution cost proxies for workflow plans."""

from __future__ import annotations

from serverless_dag.core.workflow import WorkflowSpec
from serverless_dag.planner.artifacts import PlannerArtifacts


def plan_cost_gbsec(
    workflow: WorkflowSpec,
    artifacts: PlannerArtifacts,
    memory_by_stage: dict[str, int],
) -> float:
    """Return warm execution GB-second cost for one workflow.

    This intentionally excludes platform-side idle/prewarm/pool cost. It matches
    the common provider-style billing proxy: memory GB times action execution
    seconds for each stage.
    """

    total = 0.0
    for stage_name in workflow.topological_order():
        memory_mb = int(memory_by_stage[stage_name])
        warm_ms = artifacts.warm_mean_ms(stage_name, memory_mb)
        total += (memory_mb / 1024.0) * (warm_ms / 1000.0)
    return float(total)
