"""Load measured latency/cost artifacts for planning."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
from statistics import NormalDist

from serverless_dag.risk.distributions import LogNormalParams, fit_lognormal


@dataclass(frozen=True)
class PlannerArtifacts:
    """Measured per-stage/tier latency and warm execution cost tables."""

    stage_params: dict[tuple[str, int, str], LogNormalParams]
    warm_action_mean_ms: dict[tuple[str, int], float]
    jit_warmup_lead_ms: dict[tuple[str, int], float]
    jit_warmup_lognormal: dict[tuple[str, int], LogNormalParams] = field(
        default_factory=dict
    )
    entry_cold_overhead_p95_ms: dict[tuple[str, int], float] = field(default_factory=dict)
    entry_cold_overhead_lognormal: dict[tuple[str, int], LogNormalParams] = field(
        default_factory=dict
    )
    stage_log_correlation: dict[tuple[str, str], float] = field(default_factory=dict)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        require_stage_correlation: bool = False,
    ) -> "PlannerArtifacts":
        base = Path(model_dir)
        params_path = base / "per_stage_tier_lognormal_params.csv"
        warm_path = base / "warm_tier_means.csv"
        lead_path = base / "jit_warmup_lead_by_tier.csv"
        samples_path = base / "stage_latency_samples.csv"
        correlation_path = base / "stage_log_correlation.csv"
        if not params_path.exists():
            raise FileNotFoundError(params_path)
        if not warm_path.exists():
            raise FileNotFoundError(warm_path)

        params: dict[tuple[str, int, str], LogNormalParams] = {}
        with params_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stage = str(row["stage_name"])
                tier = int(float(row["tier_mb"]))
                latency_class = str(row["latency_class"])
                params[(stage, tier, latency_class)] = LogNormalParams(
                    mu=float(row["mu"]),
                    sigma=float(row["sigma"]),
                )

        warm_mean: dict[tuple[str, int], float] = {}
        with warm_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                warm_mean[(str(row["stage_name"]), int(float(row["tier_mb"])))] = float(
                    row["warm_action_mean_ms"]
                )

        lead: dict[tuple[str, int], float] = {}
        warmup_lognormal: dict[tuple[str, int], LogNormalParams] = {}
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
                    key = (str(row["stage_name"]), int(float(row["tier_mb"])))
                    lead[key] = value
                    p50 = float(row["lead_p50_ms"])
                    p95 = float(row["lead_p95_ms"])
                    warmup_lognormal[key] = _lognormal_from_p50_p95(p50, p95)
        entry_cold_overhead = (
            _load_entry_cold_overhead_p95(samples_path) if samples_path.exists() else {}
        )
        entry_cold_overhead_lognormal = (
            _load_entry_cold_overhead_lognormal(samples_path)
            if samples_path.exists()
            else {}
        )
        stage_log_correlation = _load_stage_log_correlation(correlation_path)
        if require_stage_correlation and not stage_log_correlation:
            raise FileNotFoundError(correlation_path)
        return cls(
            stage_params=params,
            warm_action_mean_ms=warm_mean,
            jit_warmup_lead_ms=lead,
            jit_warmup_lognormal=warmup_lognormal,
            entry_cold_overhead_p95_ms=entry_cold_overhead,
            entry_cold_overhead_lognormal=entry_cold_overhead_lognormal,
            stage_log_correlation=stage_log_correlation,
        )

    def dist(self, stage_name: str, tier_mb: int, latency_class: str) -> LogNormalParams:
        key = (stage_name, int(tier_mb), latency_class)
        if key not in self.stage_params:
            raise KeyError(f"missing latency distribution for {stage_name}@{tier_mb}/{latency_class}")
        return self.stage_params[key]

    def warm_mean_ms(self, stage_name: str, tier_mb: int) -> float:
        key = (stage_name, int(tier_mb))
        if key not in self.warm_action_mean_ms:
            raise KeyError(f"missing warm action mean for {stage_name}@{tier_mb}")
        return float(self.warm_action_mean_ms[key])

    def jit_lead_ms(self, stage_name: str, tier_mb: int) -> float:
        key = (stage_name, int(tier_mb))
        if key not in self.jit_warmup_lead_ms:
            raise KeyError(f"missing JIT warmup lead for {stage_name}@{tier_mb}")
        return float(self.jit_warmup_lead_ms[key])

    def jit_warmup_dist(self, stage_name: str, tier_mb: int) -> LogNormalParams:
        """Return the measured cold-warmup completion-time distribution."""

        key = (stage_name, int(tier_mb))
        if key not in self.jit_warmup_lognormal:
            raise KeyError(f"missing JIT warmup distribution for {stage_name}@{tier_mb}")
        return self.jit_warmup_lognormal[key]

    def entry_cold_overhead_p95(self, stage_name: str, tier_mb: int) -> float:
        key = (stage_name, int(tier_mb))
        if key not in self.entry_cold_overhead_p95_ms:
            raise KeyError(f"missing entry cold overhead p95 for {stage_name}@{tier_mb}")
        return float(self.entry_cold_overhead_p95_ms[key])

    def entry_cold_overhead_dist(
        self, stage_name: str, tier_mb: int
    ) -> LogNormalParams:
        key = (stage_name, int(tier_mb))
        if key not in self.entry_cold_overhead_lognormal:
            raise KeyError(
                f"missing entry cold-overhead lognormal for {stage_name}@{tier_mb}"
            )
        return self.entry_cold_overhead_lognormal[key]

    def log_correlation(self, left_stage: str, right_stage: str) -> float:
        if left_stage == right_stage:
            return 1.0
        direct = self.stage_log_correlation.get((left_stage, right_stage))
        reverse = self.stage_log_correlation.get((right_stage, left_stage))
        if direct is None and reverse is None:
            raise KeyError(
                f"missing stage log correlation for {left_stage}/{right_stage}"
            )
        if direct is not None and reverse is not None and abs(direct - reverse) > 1e-8:
            raise ValueError(
                f"asymmetric stage log correlation for {left_stage}/{right_stage}"
            )
        return float(direct if direct is not None else reverse)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _lognormal_from_p50_p95(p50_ms: float, p95_ms: float) -> LogNormalParams:
    """Construct a tail-matched lognormal from measured warmup quantiles."""

    if not math.isfinite(p50_ms) or p50_ms <= 0.0:
        raise ValueError(f"warmup p50 must be finite and positive, got {p50_ms}")
    if not math.isfinite(p95_ms) or p95_ms < p50_ms:
        raise ValueError(f"warmup p95 must be finite and >= p50, got {p95_ms}")
    mu = math.log(p50_ms)
    if p95_ms == p50_ms:
        return LogNormalParams(mu=mu, sigma=0.0)
    z95 = NormalDist().inv_cdf(0.95)
    return LogNormalParams(mu=mu, sigma=(math.log(p95_ms) - mu) / z95)


def _load_entry_cold_overhead_p95(path: Path) -> dict[tuple[str, int], float]:
    """Derive entry-cold platform-overhead p95 from warm/cold samples.

    The additive entry-cold model uses ``warm DAG + H_cold`` rather than a
    single fitted cold-like lognormal. For each observation, platform overhead
    is dispatch latency minus action duration. ``H_cold`` is the empirical p95
    of cold platform overhead after subtracting the warm platform median. The
    risk model only applies this shift to the workflow entry.
    """

    warm: dict[tuple[str, int], list[float]] = {}
    cold: dict[tuple[str, int], list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stage = str(row["stage_name"])
            tier = int(float(row["tier_mb"]))
            dispatch_ms = float(row["dispatch_latency_ms"])
            action_ms = float(row["action_duration_ms"])
            value = max(0.0, dispatch_ms - action_ms)
            latency_class = str(row["latency_class"])
            if latency_class == "warm":
                warm.setdefault((stage, tier), []).append(value)
            elif latency_class == "cold_like":
                cold.setdefault((stage, tier), []).append(value)

    out: dict[tuple[str, int], float] = {}
    for key, cold_values in cold.items():
        warm_values = warm.get(key)
        if not warm_values:
            continue
        warm_median = _percentile(warm_values, 50.0)
        increments = [max(0.0, value - warm_median) for value in cold_values]
        out[key] = _percentile(increments, 95.0)
    return out


def _load_entry_cold_overhead_lognormal(
    path: Path,
) -> dict[tuple[str, int], LogNormalParams]:
    """Fit per-stage/tier lognormal models to cold platform increments."""

    warm: dict[tuple[str, int], list[float]] = {}
    cold: dict[tuple[str, int], list[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["stage_name"]), int(float(row["tier_mb"])))
            dispatch_ms = float(row["dispatch_latency_ms"])
            action_ms = float(row["action_duration_ms"])
            platform_ms = max(0.0, dispatch_ms - action_ms)
            latency_class = str(row["latency_class"])
            if latency_class == "warm":
                warm.setdefault(key, []).append(platform_ms)
            elif latency_class == "cold_like":
                cold.setdefault(key, []).append(platform_ms)

    out: dict[tuple[str, int], LogNormalParams] = {}
    for key, cold_values in cold.items():
        warm_values = warm.get(key)
        if not warm_values:
            continue
        warm_median = _percentile(warm_values, 50.0)
        increments = [
            max(0.0, value - warm_median) for value in cold_values
        ]
        positive = [value for value in increments if value > 0.0]
        if len(positive) < 2:
            continue
        out[key] = fit_lognormal(positive)
    return out


def _load_stage_log_correlation(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"stage_i", "stage_j", "correlation"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        for row in reader:
            key = (str(row["stage_i"]), str(row["stage_j"]))
            value = float(row["correlation"])
            if key in out:
                raise ValueError(f"duplicate stage correlation row {key} in {path}")
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"invalid stage correlation {key}={value}")
            out[key] = value
    return out
