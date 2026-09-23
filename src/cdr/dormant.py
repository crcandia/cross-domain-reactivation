"""Dormant-song reactivation machinery.

Primary design: dormant film-linked songs (bottom tercile of pre-film excess
attention) are compared with matched dormant never-film songs that receive
pseudo-event years drawn from the empirical distribution of treated first-film
years. Final inference integrates over 500 pseudo-event assignments and a
250-replication nested bootstrap that refits the snapshot-specific MemoryDecay
curves, recomputes expected/excess attention, reconstructs the dormant
definition, reassigns pseudo-events, and rematches controls inside every
replication.

The expensive stochastic outputs are stored as replication-level rows under
``data/inference``; the default analysis path validates those files against
the raw-derived analytical support and recomputes every reported summary from
the replication rows. ``FULL_INFERENCE=1`` reruns the complete procedure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors

from . import paths
from .memorydecay import (
    aggregate_age_means, canonical_curve, fit_canonical_gridsearch,
    years_equivalent_one,
)
from .util import SNAPSHOTS, sha256

# Covariates used to match dormant treated songs to dormant pseudo-event controls.
DORMANT_MATCH_COVS = [
    "baseline_excess",
    "baseline_popularity",
    "baseline_age",
    "chart_year",
    "weeks",
    "peak_pos",
    "artist_catalog_size_excl_song",
    "artist_total_weeks_excl_song",
    "artist_superstar_top1pct",
]

MC_SEED_START = 202607060
BOOT_SEED_START = 202607600

# Stored intervals from the bounded/log1p implementation are not interchangeable.
BOOTSTRAP_REFIT_VERSION = "canonical_gridsearch_refit_2026_09_10"


SNAP_YEARS = np.array([y for _, y, _, _ in SNAPSHOTS], dtype=float)
SNAP_NAMES = [s for s, _, _, _ in SNAPSHOTS]

WIDE_COVS = [
    "chart_year",
    "weeks",
    "peak_pos",
    "artist_catalog_size_excl_song",
    "artist_total_weeks_excl_song",
    "artist_superstar_top1pct",
]


def build_snapshot_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per song with per-snapshot attention/age columns and the
    time-invariant matching covariates (vectorized pre/post construction)."""
    base = panel.drop_duplicates("song_artist_id")[["song_artist_id", "film_linked"] + WIDE_COVS].set_index("song_artist_id")
    att = panel.pivot_table(index="song_artist_id", columns="snapshot", values="attention", aggfunc="first")
    age = panel.pivot_table(index="song_artist_id", columns="snapshot", values="age", aggfunc="first")
    exc = panel.pivot_table(index="song_artist_id", columns="snapshot", values="excess_attention", aggfunc="first")
    for j, snap in enumerate(SNAP_NAMES):
        base[f"att{j}"] = att[snap] if snap in att else np.nan
        base[f"age{j}"] = age[snap] if snap in age else np.nan
        base[f"exc{j}"] = exc[snap] if snap in exc else np.nan
    return base.reset_index()


def expected_matrix(wide: pd.DataFrame, fits: dict[str, dict]) -> np.ndarray:
    """Expected attention for every song at each snapshot's age under the fits."""
    E = np.full((len(wide), 4), np.nan)
    for j, snap in enumerate(SNAP_NAMES):
        f = fits[snap]
        ages = wide[f"age{j}"].to_numpy(float)
        ok = np.isfinite(ages)
        E[ok, j] = canonical_curve(ages[ok], f["N"], f["p"], f["r"], f["q"])
    return E


def prepost_indices(wide: pd.DataFrame, event_years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline (latest observed snapshot strictly before the event year) and
    first post (earliest observed at/after) indices; eligibility mask."""
    observed = np.column_stack([wide[f"att{j}"].notna().to_numpy() for j in range(4)])
    k = np.searchsorted(SNAP_YEARS, event_years, side="left")  # first snapshot index with year >= event
    pre = np.full(len(wide), -1, dtype=int)
    post = np.full(len(wide), -1, dtype=int)
    for j in range(4):
        sel = (j < k) & observed[:, j]
        pre[sel] = j
    for j in range(3, -1, -1):
        sel = (j >= k) & observed[:, j]
        post[sel] = j
    ok = (pre >= 0) & (post >= 0) & np.isfinite(event_years)
    return pre, post, ok


def prepost_table(wide: pd.DataFrame, E: np.ndarray | None, event_years: np.ndarray, pre: np.ndarray, post: np.ndarray, ok: np.ndarray, film_linked: int) -> pd.DataFrame:
    """Assemble the pre/post analytical rows for eligible songs. Excess comes
    from the stored panel values when ``E`` is None (point-estimate path) and
    from the refitted expected matrix inside bootstrap replications."""
    idx = np.where(ok)[0]
    pre_i, post_i = pre[idx], post[idx]
    att = np.column_stack([wide[f"att{j}"].to_numpy(float) for j in range(4)])
    age = np.column_stack([wide[f"age{j}"].to_numpy(float) for j in range(4)])
    if E is None:
        exc = np.column_stack([wide[f"exc{j}"].to_numpy(float) for j in range(4)])
        baseline_excess = exc[idx, pre_i]
        post_excess = exc[idx, post_i]
    else:
        baseline_excess = att[idx, pre_i] - E[idx, pre_i]
        post_excess = att[idx, post_i] - E[idx, post_i]
    out = pd.DataFrame({
        "song_artist_id": wide["song_artist_id"].to_numpy()[idx],
        "event_year": event_years[idx],
        "film_linked": film_linked,
        "baseline_snapshot": np.array(SNAP_NAMES, dtype=object)[pre_i],
        "first_post_snapshot": np.array(SNAP_NAMES, dtype=object)[post_i],
        "baseline_popularity": att[idx, pre_i],
        "first_post_popularity": att[idx, post_i],
        "baseline_age": age[idx, pre_i],
        "first_post_age": age[idx, post_i],
        "baseline_excess": baseline_excess,
        "first_post_excess": post_excess,
    })
    out["excess_change"] = out["first_post_excess"] - out["baseline_excess"]
    for c in WIDE_COVS:
        out[c] = wide[c].to_numpy(float)[idx]
    return out


def match_controls(treat: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    """1-nearest-neighbor matching with replacement on the pooled-standardized
    dormant matching covariates."""
    t = treat.dropna(subset=DORMANT_MATCH_COVS).copy()
    c = ctrl.dropna(subset=DORMANT_MATCH_COVS).copy()
    if t.empty or c.empty:
        return pd.DataFrame()
    pooled = pd.concat([t[DORMANT_MATCH_COVS], c[DORMANT_MATCH_COVS]], ignore_index=True)
    mu = pooled.mean()
    sd = pooled.std(ddof=0).replace(0, 1)
    xt = ((t[DORMANT_MATCH_COVS] - mu) / sd).to_numpy(float)
    xc = ((c[DORMANT_MATCH_COVS] - mu) / sd).to_numpy(float)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(xc)
    dist, idx = nn.kneighbors(xt)
    matched = pd.concat(
        [t.reset_index(drop=True).add_prefix("treated_"), c.iloc[idx[:, 0]].reset_index(drop=True).add_prefix("control_")],
        axis=1,
    )
    matched["match_distance"] = dist[:, 0]
    matched["diff_change"] = matched["treated_excess_change"] - matched["control_excess_change"]
    return matched


def prepare_dormant_base(wide: pd.DataFrame, E: np.ndarray | None, first_year_map: pd.Series) -> dict:
    """Treated pre/post support, bottom-tercile threshold, and dormant subset."""
    ev = wide["song_artist_id"].map(first_year_map).to_numpy(float)
    ev = np.where(wide["film_linked"].eq(1).to_numpy(), ev, np.nan)
    pre, post, ok = prepost_indices(wide, ev)
    treated = prepost_table(wide, E, ev, pre, post, ok, 1)
    q33 = treated["baseline_excess"].quantile(1 / 3)
    treated_d = treated[treated["baseline_excess"] <= q33].copy()
    return {
        "treated": treated,
        "treated_dormant": treated_d,
        "treated_event_years": treated["event_year"].dropna(),
        "q33": float(q33),
    }


def draw_pseudo_controls(wide_never: pd.DataFrame, E_never: np.ndarray | None, treated_years: pd.Series, q33: float, rng) -> pd.DataFrame:
    """Assign every never-film song a pseudo-event year drawn from the treated
    first-film-year distribution, apply the identical pre/post rule, and keep
    the dormant (bottom-tercile) controls."""
    pseudo = rng.choice(treated_years.to_numpy(float), size=len(wide_never), replace=True)
    pre, post, ok = prepost_indices(wide_never, pseudo)
    controls = prepost_table(wide_never, E_never, pseudo, pre, post, ok, 0)
    if controls.empty:
        return controls
    return controls[controls["baseline_excess"] <= q33].copy()


def monte_carlo_pseudoevent(wide: pd.DataFrame, first_year_map: pd.Series, n_rep: int = 500) -> pd.DataFrame:
    """500 deterministic-seed pseudo-event assignments; the treated side and the
    stored panel excess values are fixed at the point estimates."""
    base = prepare_dormant_base(wide, None, first_year_map)
    treated_d, q33 = base["treated_dormant"], base["q33"]
    is_never = wide["film_linked"].eq(0).to_numpy()
    wide_never = wide[is_never].reset_index(drop=True)
    E_never = None
    rows = []
    for i in range(1, n_rep + 1):
        seed = MC_SEED_START + i - 1
        rng = np.random.default_rng(int(seed))
        controls_d = draw_pseudo_controls(wide_never, E_never, base["treated_event_years"], q33, rng)
        matched = match_controls(treated_d, controls_d)
        if matched.empty:
            rows.append({"replication": i, "seed": int(seed), "N_treated_eligible": len(treated_d), "N_controls_eligible": len(controls_d)})
            continue
        reuse = matched["control_song_artist_id"].value_counts()
        treated_exit = float((matched["treated_first_post_excess"] > q33).mean())
        control_exit = float((matched["control_first_post_excess"] > q33).mean())
        rows.append(
            {
                "replication": i,
                "seed": int(seed),
                "N_treated_eligible": int(len(treated_d)),
                "N_controls_eligible": int(len(controls_d)),
                "N_matched_pairs": int(len(matched)),
                "N_unique_controls": int(reuse.shape[0]),
                "median_control_reuse": float(reuse.median()),
                "p95_control_reuse": float(reuse.quantile(0.95)),
                "max_control_reuse": int(reuse.max()),
                "excess_attention_estimate": float(matched["diff_change"].mean()),
                "treated_exit_probability": treated_exit,
                "control_exit_probability": control_exit,
                "exit_difference_pp": 100 * (treated_exit - control_exit),
            }
        )
    return pd.DataFrame(rows)


def _resampled_never_film_panel(panel: pd.DataFrame, sampled_ids) -> pd.DataFrame:
    """Repeat complete song trajectories, retaining the original row order.

    Using physical row multiplicities lets the point-estimate aggregation
    function compute both mean attention and mean fractional age identically.
    A draw containing every ID once therefore reproduces the original inputs.
    """
    cols = ["song_artist_id", "snapshot", "film_linked", "age", "attention"]
    missing = set(cols) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing baseline columns: {sorted(missing)}")
    never = panel.loc[panel["film_linked"].eq(0), cols].copy()
    if never.empty:
        raise ValueError("No never-film songs are available for baseline fitting.")
    if never[["song_artist_id", "snapshot"]].isna().any().any():
        raise ValueError("Baseline record and snapshot identifiers must not be missing.")
    if never.duplicated(["song_artist_id", "snapshot"]).any():
        raise ValueError("Expected one row per exact Billboard record and snapshot.")
    values = never[["age", "attention"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Baseline age and attention must be finite and nonnegative.")
    draws = pd.Series(list(sampled_ids), dtype=object)
    if draws.empty or draws.isna().any():
        raise ValueError("The baseline resample must contain nonmissing song IDs.")
    counts = draws.value_counts(sort=False)
    known = pd.Index(never["song_artist_id"].unique())
    if not counts.index.isin(known).all():
        raise ValueError("The resample contains IDs outside the never-film population.")
    multiplicity = never["song_artist_id"].map(counts).fillna(0).to_numpy(dtype=int)
    return never.iloc[np.repeat(np.arange(len(never)), multiplicity)].reset_index(drop=True)


def refit_memorydecay_bootstrap(
    panel: pd.DataFrame,
    rng,
    starts: dict[str, dict] | None = None,
    *,
    sampled_ids=None,
) -> dict[str, dict]:
    """Refit exactly the primary estimator on a song-level baseline resample.

    Shared with the point-estimate path: floor(age) cells, mean fractional age
    within each cell, positive age-cell attention means, log-response squared
    loss, capped inverse-age weights, log/logit parameterization, and the full
    data-adaptive multi-start grid. There is no bounded/log1p fallback.

    ``starts`` is retained for compatibility with notebook 03 but deliberately
    does not replace or narrow the primary starting grid. ``sampled_ids`` is an
    explicit test hook: every never-film ID once is the no-resampling check.
    The normal draw and RNG consumption are unchanged from the previous code.
    """
    if sampled_ids is None:
        never_ids = panel.loc[panel["film_linked"].eq(0), "song_artist_id"].drop_duplicates().to_numpy()
        sampled_ids = rng.choice(never_ids, size=len(never_ids), replace=True)
    sampled = _resampled_never_film_panel(panel, sampled_ids)
    fits = {}
    for snap in SNAP_NAMES:
        d = sampled.loc[sampled["snapshot"].eq(snap)]
        if d.empty:
            raise RuntimeError(f"No observations in baseline resample for {snap}.")
        agg = aggregate_age_means(d)
        if len(agg) < 4:
            raise RuntimeError(f"Fewer than four positive age cells in resample for {snap}.")
        try:
            fit = fit_canonical_gridsearch(agg)
        except Exception as exc:
            raise RuntimeError(f"Primary-estimator refit failed for snapshot {snap}.") from exc
        values = [fit[k] for k in ("N", "p", "r", "q", "sse_log_weighted")]
        pred = canonical_curve(agg["age"].to_numpy(float), *values[:4])
        if not np.isfinite(values).all() or not np.isfinite(pred).all():
            raise RuntimeError(f"Nonfinite primary-estimator refit for {snap}.")
        fits[snap] = fit
    return fits


def integrated_bootstrap(panel: pd.DataFrame, wide: pd.DataFrame, first_year_map: pd.Series, starts: dict[str, dict], n_rep: int = 250, *, start_rep: int = 1) -> pd.DataFrame:
    """Existing integrated resampling design with the primary baseline refitter.

    Baseline songs are resampled as whole trajectories. Given each refit, the
    lower-tercile group is reconstructed on the observed eligible treated pool,
    then resampled; the observed control pool receives one new pseudo-event
    assignment. This patch preserves that resampling scope and the matching
    estimator. It fixes baseline-estimator consistency, not a general theorem
    about bootstrap coverage for fixed-neighbor matching.

    ``start_rep`` permits deterministic, resumable batches without changing
    the seed associated with any replication. Failures abort rather than being
    silently omitted from percentile intervals.
    """
    if isinstance(n_rep, bool) or int(n_rep) != n_rep or n_rep < 1:
        raise ValueError("n_rep must be a positive integer.")
    if isinstance(start_rep, bool) or int(start_rep) != start_rep or start_rep < 1:
        raise ValueError("start_rep must be a positive integer.")
    is_never = wide["film_linked"].eq(0).to_numpy()
    wide_never = wide[is_never].reset_index(drop=True)
    rows = []
    t0 = time.perf_counter()
    for rep in range(int(start_rep), int(start_rep) + int(n_rep)):
        seed = BOOT_SEED_START + rep - 1
        rng = np.random.default_rng(seed)
        try:
            fits = refit_memorydecay_bootstrap(panel, rng, starts)
        except Exception as exc:
            raise RuntimeError(f"Bootstrap replication {rep}, seed {seed}: baseline refit failed.") from exc
        E = expected_matrix(wide, fits)
        base = prepare_dormant_base(wide, E, first_year_map)
        treated_d = base["treated_dormant"]
        if treated_d.empty or not np.isfinite(base["q33"]):
            raise RuntimeError(f"Bootstrap replication {rep}, seed {seed}: empty lower-tercile group.")
        treated_sample = treated_d.iloc[rng.integers(0, len(treated_d), size=len(treated_d))].reset_index(drop=True)
        controls_d = draw_pseudo_controls(wide_never, E[is_never], base["treated_event_years"], base["q33"], rng)
        matched = match_controls(treated_sample, controls_d)
        if matched.empty:
            raise RuntimeError(f"Bootstrap replication {rep}, seed {seed}: no matched pairs.")
        reuse = matched["control_song_artist_id"].value_counts()
        estimate = float(matched["diff_change"].mean())
        denominator = float((matched["treated_baseline_popularity"] - matched["treated_baseline_excess"]).mean())
        vals, invalid, boundary, numeric = [], 0, 0, 0
        for snap_name, b_age in zip(matched["treated_baseline_snapshot"], matched["treated_baseline_age"]):
            val, status = years_equivalent_one(fits[str(snap_name)], float(b_age), estimate)
            if np.isfinite(val):
                vals.append(val)
            else:
                invalid += 1
            if status == "above_young_maximum":
                boundary += 1
            if status == "numeric_failure":
                numeric += 1
        q33 = base["q33"]
        rows.append(
            {
                "replication": rep,
                "seed": seed,
                "bootstrap_refit_version": BOOTSTRAP_REFIT_VERSION,
                **{f"fit_{snap}_{key}": fits[snap][key]
                   for snap in SNAP_NAMES
                   for key in ("N", "p", "r", "q", "n_starts", "sse_log_weighted")},
                "N_treated": int(len(treated_sample)),
                "N_controls_eligible": int(len(controls_d)),
                "N_matched_pairs": int(len(matched)),
                "N_unique_controls": int(reuse.shape[0]),
                "median_control_reuse": float(reuse.median()),
                "p95_control_reuse": float(reuse.quantile(0.95)),
                "max_control_reuse": int(reuse.max()),
                "dormant_excess_attention": estimate,
                "mean_expected_attention_denominator": denominator,
                "relative_displacement_pct": float(100 * estimate / denominator) if denominator > 0 else np.nan,
                "treated_exit_probability": float((matched["treated_first_post_excess"] > q33).mean()),
                "control_exit_probability": float((matched["control_first_post_excess"] > q33).mean()),
                "exit_difference_pp": float(100 * ((matched["treated_first_post_excess"] > q33).mean() - (matched["control_first_post_excess"] > q33).mean())),
                "median_decay_equivalent_years": float(np.nanmedian(vals)) if vals else np.nan,
                "N_valid_inversions": int(len(vals)),
                "N_invalid_inversions": int(invalid),
                "N_boundary_hits": int(boundary),
                "N_numeric_failures": int(numeric),
            }
        )
        completed = rep - int(start_rep) + 1
        if completed % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"nested bootstrap {completed}/{n_rep}, last replication {rep} "
                  f"({elapsed / completed:.2f}s per replication)", flush=True)
    return pd.DataFrame(rows)


def _inference_code_hashes() -> dict[str, str]:
    """Bind stored replications to the code that produced their estimates."""
    here = Path(__file__).resolve().parent
    return {name: sha256(here / name) for name in ("dormant.py", "memorydecay.py", "util.py")}


def _inference_input_hashes() -> dict[str, str]:
    """Track timing and matching inputs not represented by the panel fingerprint."""
    names = ("spotify_song_snapshot_panel.csv", "master_song_panel.csv")
    return {name: sha256(paths.OUT_DATA / name) for name in names
            if (paths.OUT_DATA / name).is_file()}


def _check_replication_rows(mc: pd.DataFrame, boot: pd.DataFrame) -> None:
    if mc.empty or boot.empty:
        raise RuntimeError("Inference requires nonempty assignment and bootstrap rows.")
    if "bootstrap_refit_version" not in boot or not boot["bootstrap_refit_version"].eq(BOOTSTRAP_REFIT_VERSION).all():
        raise RuntimeError("Legacy or mixed bootstrap rows. Recompute with the primary baseline refitter.")
    for name, frame, value_cols, seed0 in (
        ("assignment", mc, ["excess_attention_estimate", "exit_difference_pp"], MC_SEED_START),
        ("bootstrap", boot, ["dormant_excess_attention", "exit_difference_pp"], BOOT_SEED_START),
    ):
        required = ["replication", "seed", "N_matched_pairs"] + value_cols
        if not set(required).issubset(frame.columns):
            raise RuntimeError(f"Missing required {name} columns.")
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise RuntimeError(f"Nonfinite {name} rows cannot be silently omitted.")
        reps = frame["replication"].to_numpy(float)
        if frame["replication"].duplicated().any() or not np.array_equal(reps, np.arange(1, len(frame) + 1)):
            raise RuntimeError(f"{name} replication IDs must be consecutive and start at one.")
        if not np.array_equal(frame["seed"].to_numpy(float), seed0 + reps - 1):
            raise RuntimeError(f"Unexpected {name} replication seeds.")
        if (frame["N_matched_pairs"] < 1).any():
            raise RuntimeError(f"Empty {name} matched samples.")


def write_inference_files(mc: pd.DataFrame, boot: pd.DataFrame, fingerprint: str) -> dict:
    """Write the authoritative replication-level inference files and metadata."""
    import json as _json

    _check_replication_rows(mc, boot)
    paths.INFERENCE.mkdir(parents=True, exist_ok=True)
    mc_path = paths.INFERENCE / "dormant_pseudoevent_monte_carlo_replications.csv"
    boot_path = paths.INFERENCE / "dormant_integrated_bootstrap_replications.csv"
    mc.to_csv(mc_path, index=False)
    boot.to_csv(boot_path, index=False)
    metadata = {
        "analysis_version": BOOTSTRAP_REFIT_VERSION,
        "bootstrap_refit_version": BOOTSTRAP_REFIT_VERSION,
        "inference_code_sha256": _inference_code_hashes(),
        "inference_input_sha256": _inference_input_hashes(),
        "baseline_estimator": "aggregate_age_means + fit_canonical_gridsearch (shared primary estimator)",
        "resampling_scope": "existing conditional treated-group resampling and observed control pool; one pseudo-event assignment per replication",
        "dormant_analytical_support_fingerprint": fingerprint,
        "seed_strategy": (
            "Pseudo-event assignments use deterministic consecutive seeds starting at "
            f"{MC_SEED_START}; the nested bootstrap uses deterministic consecutive seeds starting at "
            f"{BOOT_SEED_START}. Seeds are recorded per replication row."
        ),
        "files": [
            {"filename": mc_path.name, "sha256": sha256(mc_path), "rows": int(len(mc))},
            {"filename": boot_path.name, "sha256": sha256(boot_path), "rows": int(len(boot))},
        ],
    }
    (paths.INFERENCE / "inference_metadata.json").write_text(_json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_inference_files(fingerprint: str) -> dict:
    """Check stored inference files against the metadata (checksums, row counts)
    and the raw-derived dormant analytical-support fingerprint."""
    metadata = json.loads((paths.INFERENCE / "inference_metadata.json").read_text(encoding="utf-8"))
    problems = []
    if metadata.get("bootstrap_refit_version") != BOOTSTRAP_REFIT_VERSION:
        problems.append("obsolete bootstrap refitter; run PYTHONPATH=src python -m cdr.bootstrap_rebuild --publish")
    if metadata.get("inference_code_sha256") != _inference_code_hashes():
        problems.append("inference source code has changed; regenerate the stored replications")
    if metadata.get("inference_input_sha256") != _inference_input_hashes():
        problems.append("timing or matching input files have changed; regenerate inference")
    expected_names = {"dormant_pseudoevent_monte_carlo_replications.csv", "dormant_integrated_bootstrap_replications.csv"}
    if {spec.get("filename") for spec in metadata.get("files", [])} != expected_names:
        problems.append("inference metadata must list exactly the assignment and bootstrap files")
    if metadata["dormant_analytical_support_fingerprint"] != fingerprint:
        problems.append(
            "dormant analytical-support fingerprint mismatch: "
            f"metadata={metadata['dormant_analytical_support_fingerprint'][:12]}... "
            f"recomputed={fingerprint[:12]}..."
        )
    for spec in metadata["files"]:
        path = paths.INFERENCE / spec["filename"]
        if not path.exists():
            problems.append(f"missing inference file: {spec['filename']}")
            continue
        if sha256(path) != spec["sha256"]:
            problems.append(f"checksum mismatch: {spec['filename']}")
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle) - 1
        if rows != int(spec["rows"]):
            problems.append(f"row-count mismatch for {spec['filename']}: {rows} != {spec['rows']}")
    if problems:
        raise RuntimeError("Stored inference validation failed:\n" + "\n".join(problems))
    mc = pd.read_csv(paths.INFERENCE / "dormant_pseudoevent_monte_carlo_replications.csv")
    boot = pd.read_csv(paths.INFERENCE / "dormant_integrated_bootstrap_replications.csv")
    _check_replication_rows(mc, boot)
    return metadata


def percentile_ci(vals, lo=2.5, hi=97.5) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def summarize_stored_inference(mc: pd.DataFrame, boot: pd.DataFrame) -> dict:
    """Recompute every reported dormant summary from replication-level rows."""
    _check_replication_rows(mc, boot)
    est = pd.to_numeric(mc["excess_attention_estimate"], errors="coerce").dropna()
    exit_pp = pd.to_numeric(mc["exit_difference_pp"], errors="coerce").dropna()
    boot_valid = boot.dropna(subset=["dormant_excess_attention", "exit_difference_pp"])
    decay_valid = boot.dropna(subset=["median_decay_equivalent_years"])
    excess_ci = percentile_ci(boot_valid["dormant_excess_attention"])
    exit_ci = percentile_ci(boot_valid["exit_difference_pp"])
    decay_ci = percentile_ci(decay_valid["median_decay_equivalent_years"])
    total_inversions = boot["N_valid_inversions"].fillna(0).sum() + boot["N_invalid_inversions"].fillna(0).sum()
    return {
        "n_pseudo_event_assignments": int(len(mc)),
        "n_positive_assignments": int((est > 0).sum()),
        "assignment_mean": float(est.mean()),
        "assignment_median": float(est.median()),
        "assignment_sd": float(est.std(ddof=1)),
        "assignment_interval_low": float(est.quantile(0.025)),
        "assignment_interval_high": float(est.quantile(0.975)),
        "dormant_excess_point_estimate": float(est.mean()),
        "dormant_excess_integrated_ci_low": excess_ci[0],
        "dormant_excess_integrated_ci_high": excess_ci[1],
        "valid_bootstrap_replications": int(len(boot_valid)),
        "treated_exit_probability": float(mc["treated_exit_probability"].mean()),
        "control_exit_probability": float(mc["control_exit_probability"].mean()),
        "exit_difference_pp": float(exit_pp.mean()),
        "exit_integrated_ci_low": exit_ci[0],
        "exit_integrated_ci_high": exit_ci[1],
        "dormant_decay_equivalent_years": float(decay_valid["median_decay_equivalent_years"].median()),
        "dormant_decay_integrated_ci_low": decay_ci[0],
        "dormant_decay_integrated_ci_high": decay_ci[1],
        "decay_valid_bootstrap_replications": int(len(decay_valid)),
        "inversion_failure_rate": float(boot["N_invalid_inversions"].fillna(0).sum() / total_inversions) if total_inversions else np.nan,
        "boundary_hit_rate": float(boot["N_boundary_hits"].fillna(0).sum() / total_inversions) if total_inversions else np.nan,
        "median_p95_control_reuse": float(boot["p95_control_reuse"].median()),
        "max_control_reuse": int(boot["max_control_reuse"].max()),
        "n_matched_pairs_per_assignment": int(mc["N_matched_pairs"].iloc[0]),
    }


# ---------------------------------------------------------------------------
# Reduced-form dormant definitions and single-assignment placebo robustness.
# ---------------------------------------------------------------------------

REDUCED_FORM_DEFINITIONS = [
    ("Low baseline Spotify attention, bottom quartile", "dormant_low_quartile"),
    ("Low baseline Spotify attention, bottom tercile", "dormant_low_tercile"),
    ("Reduced-form below-expected residual, bottom tercile", "memory_below"),
    ("Historically successful but low baseline attention", "historically_successful_low_attention"),
    ("Older low-attention songs", "older_low_attention"),
]

REDUCED_MATCH_VARS = [
    "baseline_popularity",
    "baseline_age",
    "chart_year",
    "weeks",
    "peak_pos",
    "artist_catalog_size_excl_song",
    "artist_total_weeks_excl_song",
    "artist_superstar_top1pct",
]


def raw_prepost_for_event(row: pd.Series, event_year: float, pop_cols: dict[str, str]) -> dict | None:
    """Raw-popularity pre/post record for one song at a given (pseudo-)event year."""
    pre_label = pre_pop = pre_age = None
    for label, syear, _, _ in SNAPSHOTS:
        val = row[pop_cols[label]]
        if syear < event_year and pd.notna(val):
            pre_label, pre_pop, pre_age = label, val, syear - row["chart_year"]
    post_label = post_pop = None
    post_vals = []
    for label, syear, _, _ in SNAPSHOTS:
        val = row[pop_cols[label]]
        if syear >= event_year and pd.notna(val):
            if post_label is None:
                post_label, post_pop = label, val
            post_vals.append(val)
    if pre_label is None or post_label is None:
        return None
    return {
        "song": row["song"],
        "artist": row["artist"],
        "song_key": row["song_key_alnum"],
        "artist_key": row["artist_key_alnum"],
        "event_year": event_year,
        "baseline_snapshot": pre_label,
        "baseline_popularity": pre_pop,
        "baseline_age": pre_age,
        "first_post_snapshot": post_label,
        "first_post_change": post_pop - pre_pop,
        "max_post_change": max(post_vals) - pre_pop,
        "weeks": row["weeks"],
        "peak_pos": row["peak_pos"],
        "chart_year": row["chart_year"],
        "artist_catalog_size_excl_song": row["artist_catalog_size_excl_song"],
        "artist_total_weeks_excl_song": row["artist_total_weeks_excl_song"],
        "artist_superstar_top1pct": row["artist_superstar_top1pct"],
    }


def classify_dormant_reduced(prepost: pd.DataFrame, thresholds: dict | None = None):
    """Reduced-form dormancy definitions on the raw popularity scale. When
    thresholds is None they are estimated from the (treated) set and returned;
    otherwise the treated-derived thresholds are applied to the control pool."""
    df = prepost.dropna(subset=["baseline_popularity", "baseline_age", "weeks", "peak_pos"]).copy()
    return_thresholds = thresholds is None
    if return_thresholds:
        mem = smf.ols(
            "baseline_popularity ~ baseline_age + I(baseline_age**2) + weeks + peak_pos", data=df
        ).fit(cov_type="HC1")
        resid = df["baseline_popularity"] - mem.predict(df)
        thresholds = {
            "q25": df["baseline_popularity"].quantile(0.25),
            "q33": df["baseline_popularity"].quantile(1 / 3),
            "mem_params": mem.params,
            "rq33": resid.quantile(1 / 3),
            "rq67": resid.quantile(2 / 3),
            "weeks_q67": df["weeks"].quantile(2 / 3),
            "age_med": df["baseline_age"].median(),
        }
    p = thresholds
    df["dormant_low_quartile"] = df["baseline_popularity"] <= p["q25"]
    df["dormant_low_tercile"] = df["baseline_popularity"] <= p["q33"]
    pred = (
        p["mem_params"]["Intercept"]
        + p["mem_params"]["baseline_age"] * df["baseline_age"]
        + p["mem_params"]["I(baseline_age ** 2)"] * df["baseline_age"] ** 2
        + p["mem_params"]["weeks"] * df["weeks"]
        + p["mem_params"]["peak_pos"] * df["peak_pos"]
    )
    df["memory_residual"] = df["baseline_popularity"] - pred
    df["memory_regime"] = "intermediate-memory"
    df.loc[df["memory_residual"] <= p["rq33"], "memory_regime"] = "dormant/below-expected"
    df.loc[df["memory_residual"] >= p["rq67"], "memory_regime"] = "persistent/above-expected"
    df["historically_successful_low_attention"] = (
        (df["weeks"] >= p["weeks_q67"]) | (df["peak_pos"] <= 10)
    ) & df["dormant_low_quartile"]
    df["older_low_attention"] = (df["baseline_age"] >= p["age_med"]) & df["dormant_low_quartile"]
    if return_thresholds:
        return df, thresholds
    return df


def reduced_dormant_mask(df: pd.DataFrame, key: str) -> pd.Series:
    if key == "memory_below":
        return df["memory_regime"].eq("dormant/below-expected")
    return df[key].astype(bool)


def match_dormant_reduced(treated: pd.DataFrame, control: pd.DataFrame, rng_unused=None, replace: bool = True, caliper_sd: float = 0.5):
    """Nearest-neighbor matching with a caliper of 0.5 SD on baseline popularity."""
    cols = [c for c in REDUCED_MATCH_VARS if treated[c].notna().any() and control[c].notna().any()]
    t = treated.dropna(subset=cols).copy()
    c = control.dropna(subset=cols).copy()
    if len(t) == 0 or len(c) == 0:
        return None
    both = pd.concat([t[cols], c[cols]], axis=0)
    mu = both.mean()
    sd = both.std(ddof=0).replace(0, 1.0)
    ts = (t[cols] - mu) / sd
    cs = (c[cols] - mu) / sd
    bp_idx = cols.index("baseline_popularity")
    used = np.zeros(len(c), dtype=bool)
    matched_rows = []
    nn = NearestNeighbors(n_neighbors=min(len(c), 50)).fit(cs.to_numpy())
    _, idx = nn.kneighbors(ts.to_numpy())
    for i in range(len(t)):
        chosen = None
        for k in range(idx.shape[1]):
            j = idx[i, k]
            if (not replace) and used[j]:
                continue
            if abs(ts.iloc[i, bp_idx] - cs.iloc[j, bp_idx]) > caliper_sd:
                if chosen is None:
                    continue
                break
            chosen = j
            break
        if chosen is None:
            continue
        used[chosen] = True
        crow = c.iloc[chosen]
        trow = t.iloc[i]
        matched_rows.append(
            {
                "treated_song_key": trow["song_key"],
                "treated_change": trow["first_post_change"],
                "treated_baseline": trow["baseline_popularity"],
                "control_song_key": crow["song_key"],
                "control_change": crow["first_post_change"],
                "control_baseline": crow["baseline_popularity"],
                **{f"t_{col}": trow[col] for col in cols},
                **{f"c_{col}": crow[col] for col in cols},
            }
        )
    matched = pd.DataFrame(matched_rows)
    bal = []
    for col in cols:
        tv = matched[f"t_{col}"].astype(float)
        cv = matched[f"c_{col}"].astype(float)
        pooled_sd = np.sqrt((tv.var(ddof=1) + cv.var(ddof=1)) / 2)
        pooled_sd = pooled_sd if pooled_sd else 1.0
        bal.append({"covariate": col, "treated_mean": tv.mean(), "control_mean": cv.mean(), "smd_after": (tv.mean() - cv.mean()) / pooled_sd})
    return matched, pd.DataFrame(bal)


def estimate_reduced_definition(treated_d: pd.DataFrame, control_d: pd.DataFrame, label: str, rng, n_boot: int = 1000, n_perm: int = 2000):
    """Placebo-adjusted difference-in-change for one reduced-form definition,
    with a bootstrap over matched treated songs and a label-permutation test."""
    res = match_dormant_reduced(treated_d, control_d, replace=True)
    if res is None or res[0].empty:
        return None
    matched, bal = res
    t_change = matched["treated_change"].to_numpy()
    c_change = matched["control_change"].to_numpy()
    did = t_change.mean() - c_change.mean()
    n = len(matched)
    boot = np.array([t_change[b].mean() - c_change[b].mean() for b in (rng.integers(0, n, n) for _ in range(n_boot))])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    pooled = np.concatenate([treated_d["first_post_change"].to_numpy(), control_d["first_post_change"].to_numpy()])
    nt = len(treated_d)
    obs_raw = treated_d["first_post_change"].mean() - control_d["first_post_change"].mean()
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        perm[i] = pooled[idx[:nt]].mean() - pooled[idx[nt:]].mean()
    perm_p = float((np.abs(perm) >= abs(obs_raw)).mean())
    res_nr = match_dormant_reduced(treated_d, control_d, replace=False)
    did_nr, n_nr = np.nan, 0
    if res_nr is not None and not res_nr[0].empty:
        mnr = res_nr[0]
        did_nr = mnr["treated_change"].mean() - mnr["control_change"].mean()
        n_nr = len(mnr)
    summary = {
        "definition": label,
        "n_dormant_treated": int(len(treated_d)),
        "n_dormant_control": int(len(control_d)),
        "n_matched_pairs": int(n),
        "treated_change_mean": float(t_change.mean()),
        "matched_control_change_mean": float(c_change.mean()),
        "placebo_adjusted_did": float(did),
        "boot_ci_low": float(ci_lo),
        "boot_ci_high": float(ci_hi),
        "permutation_p_value": perm_p,
        "did_no_replacement": float(did_nr),
        "n_matched_pairs_no_replacement": int(n_nr),
        "max_abs_smd_after": float(bal["smd_after"].abs().max()),
    }
    return summary, bal.assign(definition=label), matched
