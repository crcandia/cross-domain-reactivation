"""Event-time bins and the strict exact-pre-attention matched temporal design."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

from .util import smd

# Cohort designs: songs first observed in a film in the cohort year, with the
# required Spotify snapshots on both sides of the event.
COHORT_SPECS = {
    2017: {"baseline": "pop2016", "event": "pop2017", "posts": ["pop2022", "pop2025"], "extra_pre": None},
    2022: {"baseline": "pop2017", "event": "pop2022", "posts": ["pop2025"], "extra_pre": "pop2016"},
}
HISTORY_COVARS = ["chart_year", "weeks", "peak_pos"]
CALIPERS = [0, 1, 2, 3, 5]


def event_bins(values: pd.Series) -> pd.Series:
    bins = pd.Series(np.nan, index=values.index, dtype=object)
    bins.loc[values <= -2] = "<= -2"
    bins.loc[(values > -2) & (values < 0)] = "-1"
    bins.loc[(values >= 0) & (values < 1)] = "0"
    bins.loc[(values >= 1) & (values <= 5)] = "1 to 5"
    bins.loc[values > 5] = ">5"
    return bins


def eligible_samples(master: pd.DataFrame, cohort: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec = COHORT_SPECS[cohort]
    required = [spec["baseline"], spec["event"]] + spec["posts"] + ([spec["extra_pre"]] if spec["extra_pre"] else [])
    treated = master[master["first_year"] == cohort].copy()
    controls = master[master["film_linked"] == 0].copy()
    for col in required:
        treated = treated[treated[col].notna()]
        controls = controls[controls[col].notna()]
    return treated.reset_index(drop=True), controls.reset_index(drop=True), spec


def match_attention_caliper(treated: pd.DataFrame, controls: pd.DataFrame, baseline: str, caliper: float) -> pd.DataFrame:
    """Optimal 1:1 matching without replacement (linear sum assignment) under a
    hard caliper on baseline Spotify attention; baseline closeness dominates
    the cost and standardized chart-history distance breaks ties."""
    if treated.empty or controls.empty:
        return pd.DataFrame()
    t_base = treated[baseline].to_numpy(dtype=float)
    c_base = controls[baseline].to_numpy(dtype=float)
    base_diff = np.abs(t_base[:, None] - c_base[None, :])
    valid = base_diff <= caliper
    pooled = pd.concat([treated[HISTORY_COVARS], controls[HISTORY_COVARS]], ignore_index=True).astype(float)
    scale = pooled.std(axis=0, ddof=1).replace(0, 1).to_numpy()
    hist_cost = np.zeros_like(base_diff, dtype=float)
    for idx, col in enumerate(HISTORY_COVARS):
        diff = (treated[col].to_numpy(dtype=float)[:, None] - controls[col].to_numpy(dtype=float)[None, :]) / scale[idx]
        hist_cost += diff**2
    cost = 100.0 * base_diff**2 + hist_cost
    big = 1e9
    cost = np.where(valid, cost, big)
    rows, cols = linear_sum_assignment(cost)
    keep = cost[rows, cols] < big
    rows, cols = rows[keep], cols[keep]
    records = []
    for r, c in zip(rows, cols):
        t = treated.iloc[r]
        ctrl = controls.iloc[c]
        record = {
            "treated_song": t["song"],
            "treated_artist": t["artist"],
            "control_song": ctrl["song"],
            "control_artist": ctrl["artist"],
            "baseline_abs_diff": abs(float(t[baseline]) - float(ctrl[baseline])),
            "baseline_treated": float(t[baseline]),
            "baseline_control": float(ctrl[baseline]),
            "chart_year_treated": float(t["chart_year"]),
            "chart_year_control": float(ctrl["chart_year"]),
            "weeks_treated": float(t["weeks"]),
            "weeks_control": float(ctrl["weeks"]),
            "peak_treated": float(t["peak_pos"]),
            "peak_control": float(ctrl["peak_pos"]),
        }
        for col in ["pop2016", "pop2017", "pop2022", "pop2025"]:
            record[f"t_{col}"] = t[col]
            record[f"c_{col}"] = ctrl[col]
        records.append(record)
    return pd.DataFrame(records)


def summarize_diff(values: pd.Series, rng: np.random.Generator | None = None, n_perm: int = 5000) -> dict:
    """Paired mean, SE, t-test p, and sign-flip randomization-inference p."""
    values = values.dropna()
    if len(values) == 0:
        return {"estimate": np.nan, "std_error": np.nan, "n_pairs": 0, "p_value": np.nan, "randomization_p": np.nan}
    test = stats.ttest_1samp(values, 0.0)
    randomization_p = np.nan
    if rng is not None and len(values) > 1:
        arr = values.to_numpy(dtype=float)
        sims = np.mean(rng.choice([-1, 1], size=(n_perm, len(arr))) * arr, axis=1)
        randomization_p = float((np.abs(sims) >= abs(arr.mean())).mean())
    return {
        "estimate": float(values.mean()),
        "std_error": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
        "n_pairs": int(len(values)),
        "p_value": float(test.pvalue) if len(values) > 1 else np.nan,
        "randomization_p": randomization_p,
    }


def cohort_effects(pairs: pd.DataFrame, cohort: int, spec: dict, rng: np.random.Generator) -> list[dict]:
    if pairs.empty:
        return []
    baseline = spec["baseline"]
    contrasts = [("event", spec["event"])] + [(f"post_{i}", col) for i, col in enumerate(spec["posts"], start=1)]
    rows = []
    for contrast, col in contrasts:
        diff = (pairs[f"t_{col}"] - pairs[f"t_{baseline}"]) - (pairs[f"c_{col}"] - pairs[f"c_{baseline}"])
        rows.append({"cohort": cohort, "contrast": contrast, **summarize_diff(diff, rng=rng)})
    if spec["extra_pre"]:
        extra = spec["extra_pre"]
        diff = (pairs[f"t_{baseline}"] - pairs[f"t_{extra}"]) - (pairs[f"c_{baseline}"] - pairs[f"c_{extra}"])
        rows.append({"cohort": cohort, "contrast": "pre_placebo", **summarize_diff(diff, rng=rng)})
    return rows


def pooled_event_post(all_pairs: dict, specs: dict, caliper: float, rng: np.random.Generator) -> list[dict]:
    rows = []
    for contrast_name in ["event", "post"]:
        diffs = []
        for cohort, spec in specs.items():
            pairs = all_pairs[(cohort, caliper)]
            if pairs.empty:
                continue
            baseline = spec["baseline"]
            col = spec["event"] if contrast_name == "event" else spec["posts"][0]
            diffs.append((pairs[f"t_{col}"] - pairs[f"t_{baseline}"]) - (pairs[f"c_{col}"] - pairs[f"c_{baseline}"]))
        if diffs:
            stacked = pd.concat(diffs, ignore_index=True)
            rows.append({"caliper": caliper, "contrast": contrast_name, **summarize_diff(stacked, rng=rng)})
    return rows


def balance_row(pairs: pd.DataFrame, cohort: int, caliper: float) -> dict:
    if pairs.empty:
        return {"cohort": cohort, "caliper": caliper, "matched_pairs": 0}
    return {
        "cohort": cohort,
        "caliper": caliper,
        "matched_pairs": int(len(pairs)),
        "mean_abs_baseline_diff": float(pairs["baseline_abs_diff"].mean()),
        "max_abs_baseline_diff": float(pairs["baseline_abs_diff"].max()),
        "baseline_smd": smd(pairs["baseline_treated"], pairs["baseline_control"]),
        "chart_year_smd": smd(pairs["chart_year_treated"], pairs["chart_year_control"]),
        "weeks_smd": smd(pairs["weeks_treated"], pairs["weeks_control"]),
        "peak_smd": smd(pairs["peak_treated"], pairs["peak_control"]),
    }


def run_strict_temporal_design(master: pd.DataFrame, seed: int = 20260602) -> dict[str, pd.DataFrame]:
    """Full strict temporal design over both cohorts and the caliper grid.

    A single seeded RNG is consumed across cohort/caliper/pooled summaries in
    fixed order so the randomization-inference p-values are reproducible."""
    rng = np.random.default_rng(seed)
    specs, all_pairs, effect_rows, balance_rows = {}, {}, [], []
    for cohort in [2017, 2022]:
        treated, controls, spec = eligible_samples(master, cohort)
        specs[cohort] = spec
        for caliper in CALIPERS:
            pairs = match_attention_caliper(treated, controls, spec["baseline"], caliper)
            if not pairs.empty:
                pairs.insert(0, "caliper", caliper)
                pairs.insert(0, "cohort", cohort)
            all_pairs[(cohort, caliper)] = pairs
            balance_rows.append(balance_row(pairs, cohort, caliper))
            effect_rows.extend([{"caliper": caliper, **row} for row in cohort_effects(pairs, cohort, spec, rng)])
    pooled = pd.DataFrame([row for cal in CALIPERS for row in pooled_event_post(all_pairs, specs, cal, rng)])
    return {
        "pairs": pd.concat([p for p in all_pairs.values() if not p.empty], ignore_index=True),
        "cohort_effects": pd.DataFrame(effect_rows),
        "balance": pd.DataFrame(balance_rows),
        "pooled": pooled,
    }
