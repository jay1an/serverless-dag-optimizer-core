"""Causal count forecaster for user-provided entry-arrival traces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DAY_S = 86400.0


@dataclass(frozen=True)
class ForecastPoint:
    start_s: float
    end_s: float
    mean: float


class CountForecaster:
    """HistGradientBoosting count forecaster with lag/seasonal features."""

    def __init__(self, *, bin_s: int = 60, lag_k: int = 10, random_state: int = 20260615):
        self.bin_s = int(bin_s)
        self.lag_k = int(lag_k)
        self.random_state = int(random_state)
        self.nb_alpha: float = 0.0
        self._starts: np.ndarray | None = None
        self._pred_mean: np.ndarray | None = None
        self._actual: np.ndarray | None = None
        self._canon_start_s: float | None = None
        self._canon_end_s: float | None = None
        self.metrics: dict[str, float] = {}

    @staticmethod
    def from_csvs(
        arrivals_csv: str | Path,
        schedule_csv: str | Path,
        *,
        bin_s: int = 60,
        lag_k: int = 10,
    ) -> "CountForecaster":
        forecaster = CountForecaster(bin_s=bin_s, lag_k=lag_k)
        arrivals = pd.read_csv(arrivals_csv)
        schedule = pd.read_csv(schedule_csv)
        forecaster.fit(arrivals, schedule)
        return forecaster

    def fit(self, arrivals: pd.DataFrame, schedule: pd.DataFrame) -> None:
        for column in ["arrival_s", "split"]:
            if column not in arrivals.columns:
                raise ValueError(f"arrivals missing {column!r}")
        for column in ["source_start_s"]:
            if column not in schedule.columns:
                raise ValueError(f"schedule missing {column!r}")

        arrival_s = arrivals["arrival_s"].to_numpy(dtype=float)
        train_mask_arrivals = arrivals["split"].astype(str).eq("train").to_numpy()
        if not np.any(train_mask_arrivals):
            raise ValueError("arrivals contain no train split")
        canon_times = schedule["source_start_s"].to_numpy(dtype=float)
        canon_start = float(np.min(canon_times))
        canon_end = float(np.max(canon_times)) + self.bin_s
        train_end = float(np.max(arrival_s[train_mask_arrivals]))

        phase = canon_start % float(self.bin_s)
        first = phase
        while first > float(np.min(arrival_s)):
            first -= float(self.bin_s)
        last = math.ceil((max(float(np.max(arrival_s)), canon_end) - first) / self.bin_s)
        edges = first + np.arange(0, last + 2, dtype=float) * float(self.bin_s)
        counts, _ = np.histogram(arrival_s, bins=edges)
        starts = edges[:-1]

        X, y, indices = self._features(starts, counts.astype(float))
        train_mask = starts[indices] <= train_end
        eval_mask = (starts[indices] >= canon_start) & (starts[indices] < canon_end)
        if not np.any(train_mask) or not np.any(eval_mask):
            raise ValueError("forecaster needs non-empty training and evaluation bins")

        model = HistGradientBoostingRegressor(
            random_state=self.random_state,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
        )
        model.fit(X[train_mask], y[train_mask])
        pred = np.maximum(0.0, model.predict(X))

        self.nb_alpha = self._holdout_nb_alpha(X, y, train_mask)
        self._starts = starts[indices]
        self._actual = y
        self._pred_mean = pred
        self._canon_start_s = canon_start
        self._canon_end_s = canon_end

        eval_actual = y[eval_mask]
        eval_pred = pred[eval_mask]
        self.metrics = {
            "bin_s": float(self.bin_s),
            "eval_bins": float(np.sum(eval_mask)),
            "eval_actual_sum": float(np.sum(eval_actual)),
            "eval_pred_sum": float(np.sum(eval_pred)),
            "eval_r2": float(r2_score(eval_actual, eval_pred)),
            "eval_mae": float(mean_absolute_error(eval_actual, eval_pred)),
            "nb_alpha": float(self.nb_alpha),
            "train_n": float(np.sum(train_mask_arrivals)),
            "canon_n": float(len(canon_times)),
            "train_ends_before_canon": float(train_end < canon_start),
        }

    def forecast_window_mean(
        self,
        *,
        start_s: float,
        end_s: float,
        class_fraction: float = 1.0,
    ) -> float:
        if self._starts is None or self._pred_mean is None:
            raise RuntimeError("forecaster is not fit")
        start_s = float(start_s)
        end_s = float(end_s)
        selected = (self._starts >= start_s) & (self._starts < end_s)
        if not np.any(selected):
            return 0.0
        return float(np.sum(self._pred_mean[selected]) * float(class_fraction))

    @property
    def canon_start_s(self) -> float:
        if self._canon_start_s is None:
            raise RuntimeError("forecaster is not fit")
        return float(self._canon_start_s)

    def _features(
        self, starts: np.ndarray, counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows: list[list[float]] = []
        ys: list[float] = []
        indices: list[int] = []
        for index in range(self.lag_k, len(counts)):
            recent = counts[index - self.lag_k : index].astype(float)
            roll3 = recent[-min(3, len(recent)) :]
            roll6 = recent[-min(6, len(recent)) :]
            roll12 = recent[-min(12, len(recent)) :]
            hour = (starts[index] / 3600.0) % 24.0
            dow = (starts[index] / DAY_S) % 7.0
            rows.append(
                [
                    *recent.tolist(),
                    float(roll3.mean()),
                    float(roll3.max()),
                    float(roll6.mean()),
                    float(roll6.max()),
                    float(roll12.mean()),
                    float(roll12.max()),
                    math.sin(2.0 * math.pi * hour / 24.0),
                    math.cos(2.0 * math.pi * hour / 24.0),
                    math.sin(2.0 * math.pi * dow / 7.0),
                    math.cos(2.0 * math.pi * dow / 7.0),
                ]
            )
            ys.append(float(counts[index]))
            indices.append(index)
        return (
            np.asarray(rows, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(indices, dtype=int),
        )

    def _holdout_nb_alpha(self, X: np.ndarray, y: np.ndarray, train_mask: np.ndarray) -> float:
        train_positions = np.flatnonzero(train_mask)
        if len(train_positions) < 30:
            return 0.0
        cut = max(10, int(len(train_positions) * 0.8))
        fit_pos = train_positions[:cut]
        val_pos = train_positions[cut:]
        model = HistGradientBoostingRegressor(
            random_state=self.random_state,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
        )
        model.fit(X[fit_pos], y[fit_pos])
        mu = np.maximum(0.0, model.predict(X[val_pos]))
        numerator = float(np.sum((y[val_pos] - mu) ** 2 - mu))
        denominator = float(np.sum(mu**2))
        return max(0.0, numerator / max(denominator, 1e-9))
