"""Canonical collective-memory decay (MemoryDecay) fitting and translations.

The authoritative expected-attention baseline is snapshot-specific: the
canonical biexponential decay curve is fitted to never-film Spotify age means
within each high-coverage snapshot, in log-response space, with early-age
weighting and the MemoryDecay starting grid. Expected and excess attention and
all decay-equivalent translations derive from these fitted curves.

The bootstrap calls the same ``aggregate_age_means`` and
``fit_canonical_gridsearch`` functions as the point-estimate path. The old
bounded/log1p helper is retained only for reproducing legacy diagnostics and
is not used by the corrected bootstrap.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import expit, logit

EPS_DENOM = 1e-8

# Exact p, r, q starting grid from the public MemoryDecay implementation,
# plus three study-specific additions documented in the SI.
PRQ_STARTS = [
    (0.4, 0.15, 0.01),
    (0.3, 0.07, 0.01),
    (0.01, 0.015, 0.0001),
    (0.015, 0.02, 0.0002),
    (0.02, 0.03, 0.0003),
    (0.025, 0.04, 0.0005),
    (0.03, 0.05, 0.0008),
    (0.005, 0.00001, 0.00005),
    (0.007, 0.012, 0.00008),
    (0.009, 0.008, 0.000006),
    (0.5, 0.2, 0.03),
    (0.08, 0.04, 0.01),
    (0.12, 0.08, 0.015),
]


def canonical_curve(age, N: float, p: float, r: float, q: float) -> np.ndarray:
    """S(t) = N [ (p-q) e^{-(p+r)t} + r e^{-qt} ] / (p + r - q)."""
    age = np.asarray(age, dtype=float)
    denom = p + r - q
    if denom <= EPS_DENOM:
        return np.full_like(age, np.nan, dtype=float)
    fast = np.exp(-(p + r) * age)
    slow = np.exp(-q * age)
    return np.maximum(N * (fast + (r / denom) * (slow - fast)), EPS_DENOM)


def aggregate_age_means(df: pd.DataFrame, value_col: str = "attention") -> pd.DataFrame:
    """Integer-age aggregation: mean attention per floor(age) cell."""
    d = df.copy()
    d["age_int"] = np.floor(d["age"]).astype(int)
    agg = (
        d.groupby("age_int")
        .agg(
            attention=(value_col, "mean"),
            n=(value_col, "size"),
            sd=(value_col, "std"),
            age=("age", "mean"),
        )
        .reset_index()
    )
    return agg[agg["attention"].notna() & (agg["attention"] > 0)].copy()


def _decode(theta):
    N = float(np.exp(theta[0]))
    p = float(np.exp(theta[1]))
    r = float(np.exp(theta[2]))
    q = max((p + r) * float(expit(theta[3])) * 0.999, EPS_DENOM)
    return N, p, r, q


def _encode(N, p, r, q):
    qfrac = min(max(q / max(p + r, EPS_DENOM) / 0.999, 1e-8), 1 - 1e-8)
    return [
        math.log(max(N, EPS_DENOM)),
        math.log(max(p, EPS_DENOM)),
        math.log(max(r, EPS_DENOM)),
        float(logit(qfrac)),
    ]


def _initial_N_values(y):
    vals = [float(np.nanmax(y)), 100, 50, 20, 15, 10, 5, 4, 3, 2, 1, 0.8, 1.2, 1.5, 0.5]
    return sorted({round(v, 8) for v in vals if np.isfinite(v) and v > 0}, reverse=True)


def early_age_weights(age: np.ndarray) -> np.ndarray:
    """MemoryDecay early-age weights 1/(age + 1e-3), capped at the 99th percentile."""
    w = 1.0 / (np.maximum(np.asarray(age, dtype=float), 0.0) + 1e-3)
    w = np.minimum(w, np.nanpercentile(w[np.isfinite(w)], 99))
    return np.sqrt(w)


def fit_canonical_gridsearch(agg: pd.DataFrame, early_weight: bool = True) -> dict:
    """Canonical fit on age means: log(observed) ~ log(S(age)); multi-start
    over the MemoryDecay N x (p, r, q) grid; unconstrained least squares on a
    log/logit reparametrization that enforces positivity and q < p + r."""
    d = agg[np.isfinite(agg["age"]) & np.isfinite(agg["attention"]) & (agg["age"] >= 0)].copy()
    pos = d.loc[d["attention"] > 0, "attention"]
    epsilon = float(pos.min() / 2) if len(pos) else 1e-3
    y = d["attention"].to_numpy(float)
    y_log = np.log(np.maximum(y, epsilon))
    age = d["age"].to_numpy(float)
    w = early_age_weights(age) if early_weight else np.ones_like(age)

    def residual(theta):
        N, p, r, q = _decode(theta)
        pred = np.log(canonical_curve(age, N, p, r, q))
        bad = ~np.isfinite(pred)
        res = (pred - y_log) * w
        if bad.any():
            res[bad] = 1e6
        return res

    best = None
    n_attempts = 0
    for N0 in _initial_N_values(y):
        for p0, r0, q0 in PRQ_STARTS:
            theta0 = np.asarray(_encode(N0, p0, r0, min(q0, (p0 + r0) * 0.95)), dtype=float)
            res = least_squares(residual, theta0, loss="linear", f_scale=1.0, max_nfev=30000)
            n_attempts += 1
            sse = float(np.sum(residual(res.x) ** 2))
            if res.success and (best is None or sse < best[0]):
                best = (sse, res)
    if best is None:
        raise RuntimeError("Canonical MemoryDecay fit did not converge from any start.")
    N, p, r, q = _decode(best[1].x)
    pred = canonical_curve(age, N, p, r, q)
    err_log = y_log - np.log(pred)
    denom = p + r - q
    return {
        "model": "canonical_biexponential_log",
        "N": N,
        "p": p,
        "r": r,
        "q": q,
        "epsilon": epsilon,
        "n_age_cells": int(len(d)),
        "n_starts": n_attempts,
        "sse_log_weighted": best[0],
        "rmse_log": float(np.sqrt(np.mean(err_log**2))),
        "rmse_linear": float(np.sqrt(np.mean((y - pred) ** 2))),
        "single_exponential_limit_flag": bool(denom < 0.005),
    }


def fit_exponential_log(agg: pd.DataFrame, early_weight: bool = True) -> dict:
    """Comparison model: log(observed) ~ log(c) - q * age (closed-form WLS)."""
    d = agg[np.isfinite(agg["age"]) & np.isfinite(agg["attention"]) & (agg["age"] > 0)].copy()
    pos = d.loc[d["attention"] > 0, "attention"]
    eps = float(pos.min() / 2) if len(pos) else 1e-3
    y = d["attention"].to_numpy(float)
    ylog = np.log(np.maximum(y, eps))
    age = d["age"].to_numpy(float)
    w = early_age_weights(age) if early_weight else np.ones_like(age)
    X = np.column_stack([np.ones(len(d)), -age])
    beta = np.linalg.lstsq(X * w[:, None], ylog * w, rcond=None)[0]
    pred_log = X @ beta
    err_log = ylog - pred_log
    return {
        "model": "exponential_log",
        "c": float(np.exp(beta[0])),
        "q": float(beta[1]),
        "n_age_cells": int(len(d)),
        "rmse_log": float(np.sqrt(np.mean(err_log**2))),
        "rmse_linear": float(np.sqrt(np.mean((y - np.exp(pred_log)) ** 2))),
    }


def fit_lognormal_log(agg: pd.DataFrame, early_weight: bool = True) -> dict:
    """Comparison model: log(observed) ~ b + b1 log(age) - b2 log(age)^2."""
    d = agg[np.isfinite(agg["age"]) & np.isfinite(agg["attention"]) & (agg["age"] > 0)].copy()
    pos = d.loc[d["attention"] > 0, "attention"]
    eps = float(pos.min() / 2) if len(pos) else 1e-3
    y = d["attention"].to_numpy(float)
    ylog = np.log(np.maximum(y, eps))
    age = d["age"].to_numpy(float)
    lage = np.log(age)
    w = early_age_weights(age) if early_weight else np.ones_like(age)
    X = np.column_stack([np.ones(len(d)), lage, -(lage**2)])
    beta = np.linalg.lstsq(X * w[:, None], ylog * w, rcond=None)[0]
    pred_log = X @ beta
    err_log = ylog - pred_log
    return {
        "model": "lognormal_log",
        "b": float(beta[0]),
        "b1": float(beta[1]),
        "b2": float(beta[2]),
        "n_age_cells": int(len(d)),
        "rmse_log": float(np.sqrt(np.mean(err_log**2))),
        "rmse_linear": float(np.sqrt(np.mean((y - np.exp(pred_log)) ** 2))),
    }


def predict_comparison_model(fit: dict, ages: np.ndarray) -> np.ndarray:
    ages = np.asarray(ages, dtype=float)
    if fit["model"] == "canonical_biexponential_log":
        return canonical_curve(ages, fit["N"], fit["p"], fit["r"], fit["q"])
    if fit["model"] == "exponential_log":
        return fit["c"] * np.exp(-fit["q"] * ages)
    xp = np.maximum(ages, 1e-6)
    return np.exp(fit["b"] + fit["b1"] * np.log(xp) - fit["b2"] * np.log(xp) ** 2)


def apply_expected_excess(panel: pd.DataFrame, fits: dict[str, dict]) -> pd.DataFrame:
    """Attach expected attention (fitted snapshot curve at the song's age) and
    excess attention (observed minus expected) to a song-snapshot panel."""
    out = panel.copy()
    expected = np.full(len(out), np.nan)
    for snap, fit in fits.items():
        mask = out["snapshot"].eq(snap).to_numpy()
        expected[mask] = canonical_curve(out.loc[mask, "age"].to_numpy(float), fit["N"], fit["p"], fit["r"], fit["q"])
    out["expected_attention"] = expected
    out["excess_attention"] = out["attention"] - out["expected_attention"]
    return out


def years_equivalent_one(fit: dict, age: float, delta: float) -> tuple[float, str]:
    """Decay-equivalent translation: for a song of age t and contrast delta,
    solve S(t*) = S(t) + delta for t* < t and return t - t*.

    Status codes: valid / invalid_input / above_young_maximum / numeric_failure.
    """
    if not np.isfinite(age) or age <= 0 or not np.isfinite(delta) or delta <= 0:
        return np.nan, "invalid_input"
    params = (fit["N"], fit["p"], fit["r"], fit["q"])
    s_age = float(canonical_curve(np.asarray([age]), *params)[0])
    target = s_age + delta
    young = float(canonical_curve(np.asarray([0.0]), *params)[0])
    if target >= young:
        return np.nan, "above_young_maximum"
    try:
        root = brentq(lambda a: float(canonical_curve(np.asarray([a]), *params)[0]) - target, 0, age)
        return float(age - root), "valid"
    except Exception:
        return np.nan, "numeric_failure"


def years_equivalent_distribution(fits: dict[str, dict], ages, snaps, delta: float) -> np.ndarray:
    vals = []
    for a, s in zip(np.asarray(ages, float), np.asarray(snaps, dtype=object)):
        fit = fits.get(str(s))
        if fit is None:
            vals.append(np.nan)
            continue
        val, _ = years_equivalent_one(fit, float(a), delta)
        vals.append(val)
    return np.asarray(vals, dtype=float)


def fit_canonical_bounded(age: np.ndarray, y: np.ndarray, start: dict | None = None) -> dict:
    """Legacy bounded fit, not used by the corrected bootstrap.

    Historical specification: log1p response, soft-L1 loss, double weight on ages <= 10,
    box constraints N in [1, 150], p in [.3, .9], r in [.08, .3],
    q in [.015, .06], and a penalty keeping p + r - q > 0.005."""
    age = np.asarray(age, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.where(age <= 10, 2.0, 1.0)
    x0 = np.asarray(
        [
            float(start.get("N", 80.0)) if start else 80.0,
            float(start.get("p", 0.45)) if start else 0.45,
            float(start.get("r", 0.18)) if start else 0.18,
            float(start.get("q", 0.03)) if start else 0.03,
        ]
    )
    lo = [1, 0.3, 0.08, 0.015]
    hi = [150, 0.9, 0.3, 0.06]
    x0 = np.clip(x0, lo, hi)

    def curve(par):
        denom = max(par[1] + par[2] - par[3], 1e-10)
        fast = np.exp(-(par[1] + par[2]) * age)
        slow = np.exp(-par[3] * age)
        return np.maximum(par[0] / denom * ((par[1] - par[3]) * fast + par[2] * slow), 0.0)

    def residual(par):
        pred = curve(par)
        denom_pen = max(0, 0.005 - (par[1] + par[2] - par[3])) * 1e5
        return np.r_[(np.log1p(pred) - np.log1p(y)) * np.sqrt(weights), denom_pen]

    res = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=1500)
    return {"N": float(res.x[0]), "p": float(res.x[1]), "r": float(res.x[2]), "q": float(res.x[3]), "fit_success": bool(res.success)}


def dormant_support_fingerprint(panel: pd.DataFrame) -> str:
    """SHA-256 fingerprint of the analytical support behind the dormant-song
    inference (song-snapshot ids, attention, ages, film linkage, and the
    fitted expected/excess attention), used to tie the stored replication-level
    inference files to the raw-derived panel."""
    cols = [
        "song_artist_id",
        "snapshot",
        "snapshot_year",
        "attention",
        "age",
        "film_linked",
        "expected_attention",
        "excess_attention",
    ]
    d = panel[cols].sort_values(["song_artist_id", "snapshot"]).copy()
    payload = d.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
