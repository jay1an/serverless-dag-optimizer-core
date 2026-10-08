"""Pure timing helpers for runtime JIT warmup scheduling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from serverless_dag.core.workflow import WorkflowSpec


@dataclass(frozen=True)
class JitLatencyTables:
    """Warm duration and JIT lead lookup tables keyed by stage and tier."""

    warm_duration_ms: dict[tuple[str, int], float]
    cold_overhead_ms: dict[tuple[str, int], float]
    warmup_lead_ms: dict[tuple[str, int], float]

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "JitLatencyTables":
        base = Path(model_dir)
        warm_path = base / "warm_tier_means.csv"
        cold_path = base / "cold_overhead_by_tier.csv"
        lead_path = base / "jit_warmup_lead_by_tier.csv"
        if not warm_path.exists():
            raise FileNotFoundError(warm_path)
        if not cold_path.exists():
            raise FileNotFoundError(cold_path)

        warm: dict[tuple[str, int], float] = {}
        with warm_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                warm[
                    (str(row["stage_name"]), int(float(row["tier_mb"])))
                ] = float(row["warm_action_mean_ms"])

        cold: dict[tuple[str, int], float] = {}
        with cold_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                cold[
                    (str(row["stage_name"]), int(float(row["tier_mb"])))
                ] = float(row["cleansed_cold_overhead_ms"])

        lead = dict(cold)
        if lead_path.exists():
            with lead_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    value = None
                    for column in (
                        "lead_p95_ms",
                        "jit_warmup_lead_p95_ms",
                        "warmup_lead_p95_ms",
                    ):
                        if column in row and str(row[column]).strip() != "":
                            value = float(row[column])
                            break
                    if value is None:
                        raise ValueError(
                            f"{lead_path} must contain lead_p95_ms "
                            "or jit_warmup_lead_p95_ms"
                        )
                    lead[(str(row["stage_name"]), int(float(row["tier_mb"])))] = value

        return cls(
            warm_duration_ms=warm,
            cold_overhead_ms=cold,
            warmup_lead_ms=lead,
        )

    def warm_ms(self, stage_name: str, tier_mb: int) -> float:
        key = (stage_name, int(tier_mb))
        if key not in self.warm_duration_ms:
            raise KeyError(f"missing warm duration for {stage_name}@{tier_mb}")
        return float(self.warm_duration_ms[key])

    def cold_ms(self, stage_name: str, tier_mb: int) -> float:
        key = (stage_name, int(tier_mb))
        if key not in self.cold_overhead_ms:
            raise KeyError(f"missing cold overhead for {stage_name}@{tier_mb}")
        return float(self.cold_overhead_ms[key])

    def jit_lead_ms(self, stage_name: str, tier_mb: int) -> float:
        """Return the per-stage/tier warmup lead used by JIT scheduling."""

        key = (stage_name, int(tier_mb))
        if key not in self.warmup_lead_ms:
            raise KeyError(f"missing JIT warmup lead for {stage_name}@{tier_mb}")
        return float(self.warmup_lead_ms[key])


def compute_predicted_times(
    *,
    workflow: WorkflowSpec,
    memory_by_stage: dict[str, int],
    latency_tables: JitLatencyTables,
    workflow_start_monotonic: float,
    started_at: dict[str, float],
    measured_completion_at: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Predict start/completion times using warm execution for every stage.

    This intentionally treats the entry stage as warm. Cold overhead is used by
    the warmup fire-time calculation for the *target* stage, not by the DAG
    need-at prediction. That keeps first-hop JIT from being delayed by an entry
    cold assumption.
    """

    predicted_start: dict[str, float] = {}
    predicted_completion: dict[str, float] = {}
    for stage_name in workflow.topological_order():
        node = workflow.nodes[stage_name]
        if node.parents:
            start_at = max(predicted_completion[parent] for parent in node.parents)
        else:
            start_at = workflow_start_monotonic
        if stage_name in started_at:
            start_at = started_at[stage_name]
        predicted_start[stage_name] = start_at

        if stage_name in measured_completion_at:
            predicted_completion[stage_name] = measured_completion_at[stage_name]
        else:
            duration_s = latency_tables.warm_ms(stage_name, memory_by_stage[stage_name]) / 1000.0
            predicted_completion[stage_name] = start_at + duration_s
    return predicted_start, predicted_completion


def warmup_fire_time(
    *,
    needed_at: float,
    cold_overhead_ms: float,
    margin_ms: float = 0.0,
    settle_ms: float = 0.0,
) -> float:
    """Return the monotonic timestamp when a JIT warmup should fire."""

    return needed_at - (float(cold_overhead_ms) + float(margin_ms) + float(settle_ms)) / 1000.0
