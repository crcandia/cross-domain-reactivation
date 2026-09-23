"""Nearest-neighbor matching engines and replacement-aware inference."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .util import smd


def standardize_frame(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Z-scores with median imputation for missing covariate values (matching
    space only; rows with missing outcomes/treatment are dropped upstream)."""
    z = df[covariates].apply(pd.to_numeric, errors="coerce").copy()
    for col in covariates:
        med = z[col].median()
        z[col] = z[col].fillna(med)
        sd = z[col].std(ddof=0)
        if sd == 0 or pd.isna(sd):
            z[col] = 0.0
        else:
            z[col] = (z[col] - z[col].mean()) / sd
    return z


def nearest_neighbor_match(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    outcome_col: str = "attention",
    caliper_col: str | None = None,
    caliper: float | None = None,
    replace: bool = True,
    max_candidates: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-nearest-neighbor matching on standardized Euclidean distance.

    With a caliper, the nearest control within the caliper on ``caliper_col``
    (raw units) is chosen among up to ``max_candidates`` nearest candidates.
    Returns (pairs, covariate balance before/after)."""
    d = data.dropna(subset=[outcome_col, treatment_col]).copy()
    if caliper_col:
        d = d[d[caliper_col].notna()].copy()
    if d[treatment_col].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    z = standardize_frame(d, covariates)
    d = d.reset_index(drop=True)
    z = z.reset_index(drop=True)
    treat_idx = d.index[d[treatment_col].eq(1)].to_numpy()
    control_idx = d.index[d[treatment_col].eq(0)].to_numpy()
    n_neighbors = 1 if (caliper_col is None or caliper is None) and replace else min(max_candidates, len(control_idx))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(z.loc[control_idx, covariates].to_numpy())
    dist, pos = nn.kneighbors(z.loc[treat_idx, covariates].to_numpy(), n_neighbors=n_neighbors)
    used: set[int] = set()
    pairs = []
    for i, t_idx in enumerate(treat_idx):
        chosen = None
        for j, candidate_pos in enumerate(np.atleast_1d(pos[i])):
            c_idx = control_idx[candidate_pos]
            if not replace and int(c_idx) in used:
                continue
            if caliper_col is not None and caliper is not None:
                if abs(float(d.loc[c_idx, caliper_col]) - float(d.loc[t_idx, caliper_col])) > caliper:
                    continue
            chosen = (t_idx, c_idx, float(np.atleast_1d(dist[i])[j]))
            break
        if chosen is not None:
            used.add(int(chosen[1]))
            pairs.append(chosen)
    if not pairs:
        return pd.DataFrame(), pd.DataFrame()
    pair_df = pd.DataFrame(pairs, columns=["treated_index", "control_index", "match_distance"])
    treated = d.loc[pair_df["treated_index"]].reset_index(drop=True)
    controls = d.loc[pair_df["control_index"]].reset_index(drop=True)
    pair_df["treated_outcome"] = treated[outcome_col].to_numpy()
    pair_df["control_outcome"] = controls[outcome_col].to_numpy()
    pair_df["diff"] = pair_df["treated_outcome"] - pair_df["control_outcome"]
    for col in ["song", "artist", "song_key", "artist_key", "song_artist_id", "snapshot"]:
        if col in d.columns:
            pair_df[f"treated_{col}"] = treated[col].to_numpy()
            pair_df[f"control_{col}"] = controls[col].to_numpy()

    all_t = d[d[treatment_col].eq(1)]
    all_c = d[d[treatment_col].eq(0)]
    balance = pd.DataFrame(
        [
            {
                "covariate": cov,
                "treated_mean_before": all_t[cov].mean(),
                "control_mean_before": all_c[cov].mean(),
                "smd_before": smd(all_t[cov], all_c[cov]),
                "treated_mean_after": treated[cov].mean(),
                "control_mean_after": controls[cov].mean(),
                "smd_after": smd(treated[cov], controls[cov]),
            }
            for cov in covariates
        ]
    )
    return pair_df, balance


def p_value_from_diffs(diffs: pd.Series) -> float:
    from scipy import stats

    vals = pd.to_numeric(diffs, errors="coerce").dropna()
    if len(vals) <= 1:
        return np.nan
    return float(stats.ttest_1samp(vals, 0.0).pvalue)


def summarize_pairs(
    pairs: pd.DataFrame,
    balance: pd.DataFrame,
    design: str,
    snapshot: str,
    n_treated_available: int,
    n_controls_available: int,
    replacement: str = "with replacement",
) -> dict:
    if pairs.empty:
        return {
            "design": design,
            "snapshot": snapshot,
            "n_treated_available": n_treated_available,
            "n_matched_treated": 0,
            "n_unique_controls": 0,
            "n_controls_available": n_controls_available,
            "replacement": replacement,
            "estimate": np.nan,
            "std_error": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "p_value": np.nan,
            "max_abs_smd_before": np.nan,
            "max_abs_smd_after": np.nan,
        }
    diffs = pairs["diff"]
    se = diffs.std(ddof=1) / math.sqrt(len(diffs)) if len(diffs) > 1 else np.nan
    est = diffs.mean()
    return {
        "design": design,
        "snapshot": snapshot,
        "n_treated_available": n_treated_available,
        "n_matched_treated": len(pairs),
        "n_unique_controls": int(pairs["control_index"].nunique()),
        "n_controls_available": n_controls_available,
        "replacement": replacement,
        "estimate": est,
        "std_error": se,
        "conf_low": est - 1.96 * se,
        "conf_high": est + 1.96 * se,
        "p_value": p_value_from_diffs(diffs),
        "max_abs_smd_before": balance["smd_before"].abs().max(),
        "max_abs_smd_after": balance["smd_after"].abs().max(),
    }


def replacement_aware_inference(pairs: pd.DataFrame, label: str, rng: np.random.Generator, n_boot: int = 400) -> dict:
    """Naive paired SE, bootstrap over treated pairs, and a cluster bootstrap
    over reused control units (the units that create dependence when matching
    with replacement)."""
    diff = pairs["diff"].to_numpy(dtype=float)
    n = len(diff)
    est = diff.mean()
    naive_se = diff.std(ddof=1) / np.sqrt(n)
    bt = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    codes = pairs["control_key"].astype("category").cat.codes.to_numpy()
    K = codes.max() + 1
    gsum = np.bincount(codes, weights=diff, minlength=K)
    gcnt = np.bincount(codes, minlength=K).astype(float)
    draw = rng.integers(0, K, size=(n_boot, K))
    bc = gsum[draw].sum(axis=1) / gcnt[draw].sum(axis=1)
    return {
        "comparison": label,
        "n_matched_pairs": n,
        "n_unique_controls": int(pairs["control_key"].nunique()),
        "control_reuse_share": float(1 - pairs["control_key"].nunique() / n),
        "estimate": float(est),
        "naive_paired_se": float(naive_se),
        "naive_ci_low": float(est - 1.96 * naive_se),
        "naive_ci_high": float(est + 1.96 * naive_se),
        "boot_treated_se": float(bt.std(ddof=1)),
        "boot_treated_ci_low": float(np.percentile(bt, 2.5)),
        "boot_treated_ci_high": float(np.percentile(bt, 97.5)),
        "cluster_control_se": float(bc.std(ddof=1)),
        "cluster_control_ci_low": float(np.percentile(bc, 2.5)),
        "cluster_control_ci_high": float(np.percentile(bc, 97.5)),
    }


def cem_estimate(d: pd.DataFrame, treat_col: str, outcome_col: str = "attention") -> tuple[float, int]:
    """Coarsened exact matching on chart decade x baseline-popularity quintile."""
    d = d.dropna(subset=[treat_col, outcome_col, "chart_year", "previous_popularity"]).copy()
    if d.empty:
        return np.nan, 0
    d["decade"] = (np.floor(d["chart_year"] / 10) * 10).astype("Int64")
    d["pbin"] = pd.qcut(d["previous_popularity"].rank(method="first"), 5, labels=False)
    diffs, ns = [], 0
    for _, g in d.groupby(["decade", "pbin"], observed=True):
        t = g[g[treat_col].eq(1)][outcome_col]
        c = g[g[treat_col].eq(0)][outcome_col]
        if len(t) and len(c):
            diffs.append((t.mean() - c.mean(), len(t)))
            ns += len(t)
    if not diffs:
        return np.nan, 0
    w = np.array([n for _, n in diffs])
    v = np.array([x for x, _ in diffs])
    return float((v * w).sum() / w.sum()), int(ns)
