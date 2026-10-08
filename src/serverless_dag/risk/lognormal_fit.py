"""Offline lognormal fitting for per-stage latency samples."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy import stats

from serverless_dag.risk.distributions import fit_lognormal


DEFAULT_STAGE_ORDER: tuple[str, ...] = ()
DEFAULT_CLASS_ORDER = ("warm", "cold_like")


def positive_latency_frame(
    samples: str | Path | pd.DataFrame,
    latency_column: str = "dispatch_latency_ms",
) -> pd.DataFrame:
    """Return positive finite latency samples with required columns."""

    df = pd.read_csv(samples) if not isinstance(samples, pd.DataFrame) else samples.copy()
    required = {"stage_name", "latency_class", latency_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"latency samples missing required columns: {missing}")
    df[latency_column] = pd.to_numeric(df[latency_column], errors="coerce")
    return df[np.isfinite(df[latency_column]) & (df[latency_column] > 0.0)].copy()


def ordered_latency_groups(
    df: pd.DataFrame,
    stage_order: Iterable[str] = DEFAULT_STAGE_ORDER,
    class_order: Iterable[str] = DEFAULT_CLASS_ORDER,
) -> Iterable[tuple[str, str, pd.DataFrame]]:
    """Yield ``(stage_name, latency_class, group)`` in a stable order."""

    seen: set[tuple[str, str]] = set()
    for stage_name in stage_order:
        for latency_class in class_order:
            group = df[(df["stage_name"] == stage_name) & (df["latency_class"] == latency_class)]
            if not group.empty:
                seen.add((stage_name, latency_class))
                yield stage_name, latency_class, group
    for (stage_name, latency_class), group in sorted(df.groupby(["stage_name", "latency_class"])):
        if (stage_name, latency_class) not in seen:
            yield str(stage_name), str(latency_class), group


def fit_per_stage_lognormal(
    samples: str | Path | pd.DataFrame,
    latency_column: str = "dispatch_latency_ms",
    stage_order: Iterable[str] = DEFAULT_STAGE_ORDER,
    class_order: Iterable[str] = DEFAULT_CLASS_ORDER,
) -> pd.DataFrame:
    """Fit lognormal parameters per ``(stage_name, latency_class)``."""

    df = positive_latency_frame(samples, latency_column=latency_column)
    rows: list[dict[str, float | int | str]] = []
    for stage_name, latency_class, group in ordered_latency_groups(
        df, stage_order=stage_order, class_order=class_order
    ):
        values = group[latency_column].to_numpy(dtype=float)
        params = fit_lognormal(values)
        dist = stats.lognorm(s=params.sigma, loc=0.0, scale=math.exp(params.mu))
        ks_statistic, ks_pvalue = stats.kstest(values, dist.cdf)

        mean_empirical = float(np.mean(values))
        std_empirical = float(np.std(values, ddof=1))
        rows.append(
            {
                "stage_name": stage_name,
                "latency_class": latency_class,
                "n_samples": int(len(values)),
                "mu": params.mu,
                "sigma": params.sigma,
                "mean_empirical": mean_empirical,
                "mean_predicted": params.mean,
                "std_empirical": std_empirical,
                "std_predicted": math.sqrt(params.variance),
                "cv_empirical": std_empirical / mean_empirical,
                "cv_predicted": params.cv,
                "p50_empirical": float(np.quantile(values, 0.50)),
                "p50_predicted": params.quantile(0.50),
                "p95_empirical": float(np.quantile(values, 0.95)),
                "p95_predicted": params.quantile(0.95),
                "ks_statistic": float(ks_statistic),
                "ks_pvalue": float(ks_pvalue),
            }
        )
    return pd.DataFrame(rows)


def fit_grouped_lognormal(
    samples: str | Path | pd.DataFrame,
    group_columns: Iterable[str],
    latency_column: str = "dispatch_latency_ms",
) -> pd.DataFrame:
    """Fit lognormal parameters for arbitrary grouping columns.

    This is useful for platform sweeps where the planner wants one latency
    distribution per ``(stage, tier, warm/cold)`` cell rather than one
    distribution per stage across all tiers.
    """

    df = pd.read_csv(samples) if not isinstance(samples, pd.DataFrame) else samples.copy()
    group_columns = tuple(group_columns)
    required = set(group_columns) | {latency_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"latency samples missing required columns: {missing}")

    df[latency_column] = pd.to_numeric(df[latency_column], errors="coerce")
    df = df[np.isfinite(df[latency_column]) & (df[latency_column] > 0.0)].copy()

    rows: list[dict[str, float | int | str]] = []
    for key, group in df.groupby(list(group_columns), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        values = group[latency_column].to_numpy(dtype=float)
        params = fit_lognormal(values)
        row: dict[str, float | int | str] = {
            column: value for column, value in zip(group_columns, key)
        }
        mean_empirical = float(np.mean(values))
        std_empirical = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row.update(
            {
                "n_samples": int(len(values)),
                "mu": params.mu,
                "sigma": params.sigma,
                "mean_empirical": mean_empirical,
                "mean_predicted": params.mean,
                "std_empirical": std_empirical,
                "std_predicted": math.sqrt(params.variance),
                "cv_empirical": (
                    std_empirical / mean_empirical if mean_empirical > 0.0 else math.nan
                ),
                "cv_predicted": params.cv,
                "p50_empirical": float(np.quantile(values, 0.50)),
                "p50_predicted": params.quantile(0.50),
                "p95_empirical": float(np.quantile(values, 0.95)),
                "p95_predicted": params.quantile(0.95),
                "p99_empirical": float(np.quantile(values, 0.99)),
                "p99_predicted": params.quantile(0.99),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_columns)).reset_index(drop=True)
