"""Probability distribution definitions used by the risk model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class LogNormalParams:
    """Parameters for ``T ~ LogNormal(mu, sigma)`` in milliseconds."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError(f"mu must be finite, got {self.mu}")
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(f"sigma must be finite and non-negative, got {self.sigma}")

    @property
    def mean(self) -> float:
        return float(math.exp(self.mu + (self.sigma**2) / 2.0))

    @property
    def variance(self) -> float:
        if self.sigma == 0.0:
            return 0.0
        return float(math.expm1(self.sigma**2) * math.exp(2.0 * self.mu + self.sigma**2))

    @property
    def cv(self) -> float:
        if self.sigma == 0.0:
            return 0.0
        return float(math.sqrt(math.expm1(self.sigma**2)))

    def quantile(self, p: float) -> float:
        if not 0.0 < p < 1.0:
            raise ValueError(f"quantile probability must be in (0, 1), got {p}")
        return float(math.exp(self.mu + norm.ppf(p) * self.sigma))

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        if self.sigma == 0.0:
            return 1.0 if x >= math.exp(self.mu) else 0.0
        return float(norm.cdf((math.log(x) - self.mu) / self.sigma))

    def survival(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        if self.sigma == 0.0:
            return 0.0 if x >= math.exp(self.mu) else 1.0
        return float(norm.sf((math.log(x) - self.mu) / self.sigma))


def fit_lognormal(samples: Iterable[float]) -> LogNormalParams:
    """Fit lognormal MLE parameters to positive finite samples."""

    values = np.asarray(list(samples), dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < 2:
        raise ValueError("fit_lognormal requires at least two positive finite samples")
    logs = np.log(values)
    return LogNormalParams(mu=float(np.mean(logs)), sigma=float(np.std(logs, ddof=1)))


def lognormal_from_mean_std(mean: float, std: float) -> LogNormalParams:
    """Construct a lognormal whose arithmetic mean/std match the inputs."""

    if mean <= 0.0 or not math.isfinite(mean):
        raise ValueError(f"mean must be finite and positive, got {mean}")
    if std < 0.0 or not math.isfinite(std):
        raise ValueError(f"std must be finite and non-negative, got {std}")
    if std == 0.0:
        return LogNormalParams(mu=math.log(mean), sigma=0.0)
    sigma_sq = math.log1p((std * std) / (mean * mean))
    return LogNormalParams(mu=math.log(mean) - sigma_sq / 2.0, sigma=math.sqrt(sigma_sq))

