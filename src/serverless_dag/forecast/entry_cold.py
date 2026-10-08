"""Convert an entry demand forecast and pool capacity into cold probability."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy.stats import nbinom


def normalize_action_for_poolstate(action_name: object) -> str:
    raw = str(action_name or "")
    if "/" in raw:
        return raw.rstrip("/").split("/")[-1]
    return raw


def available_from_poolstate(
    poolstate_rows: Iterable[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    *,
    include_warming: bool = True,
) -> dict[tuple[str, int], int]:
    """Return immediately usable entry capacity keyed by ``(action, memoryMB)``.

    ``/poolState`` exposes free, busy, and warming containers.  Busy containers
    cannot serve a new entry request immediately.  For an upper-confidence
    entry-cold estimate, callers can set ``include_warming=False`` so only
    containers that are already free count as capacity.
    """

    rows: Iterable[Mapping[str, Any]]
    rows = poolstate_rows.values() if isinstance(poolstate_rows, Mapping) else poolstate_rows

    out: dict[tuple[str, int], int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action = str(row.get("action", ""))
        if not action:
            continue
        try:
            memory = int(float(row.get("memoryMB")))
        except (TypeError, ValueError):
            continue
        free = max(0, int(float(row.get("free", 0) or 0)))
        warming = max(0, int(float(row.get("warming", 0) or 0)))
        value = free + warming if include_warming else free
        for key_action in {action, normalize_action_for_poolstate(action)}:
            out[(key_action, memory)] = max(out.get((key_action, memory), 0), value)
    return out


def _expected_shortage_nb(mean: float, available: float, alpha: float) -> float:
    if mean <= 0.0:
        return 0.0
    if available <= 0.0:
        return mean
    if alpha <= 1e-12:
        # Limit case is close to Poisson, but this project intentionally uses
        # NB as the main model; keep the same interface and a stable fallback.
        variance = mean
        size = max(mean * mean / max(variance - mean, 1e-9), 1e9)
        prob = size / (size + mean)
    else:
        size = 1.0 / alpha
        prob = size / (size + mean)
        variance = mean + alpha * mean * mean

    rv = nbinom(size, prob)
    q = rv.ppf(0.999999999999)
    if not math.isfinite(float(q)):
        q = mean + 12.0 * math.sqrt(max(variance, 1e-9))
    upper = int(max(math.ceil(q), math.ceil(available) + 1, 1))
    xs = np.arange(0, upper + 1, dtype=float)
    return float(np.sum(np.maximum(xs - available, 0.0) * rv.pmf(xs)))


def expected_entry_cold_fraction(
    mean: float,
    available: float,
    *,
    alpha: float,
) -> float:
    """Expected fraction of entry arrivals that cannot find a warm slot.

    This is ``E[(D-C)+] / E[D]`` with ``D ~ NB(mean, alpha)`` and capacity ``C``.
    It is intentionally a fraction, not just ``P(D>C)``, because planning cares
    about the expected share of cold entry requests.
    """

    mean = float(mean)
    available = float(available)
    alpha = float(alpha)
    if not math.isfinite(mean) or mean < 0.0:
        raise ValueError(f"mean must be finite and >= 0, got {mean}")
    if not math.isfinite(available) or available < 0.0:
        raise ValueError(f"available must be finite and >= 0, got {available}")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    if mean <= 0.0:
        return 0.0
    return float(np.clip(_expected_shortage_nb(mean, available, alpha) / mean, 0.0, 1.0))
