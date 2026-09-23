"""Shared utilities: normalization keys, IO, statistics, and display/save helpers."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats

# Four high-coverage Spotify snapshots used throughout the study.
SNAPSHOTS = [
    ("2016-10", 2016.80, "pop2016", "October 2016"),
    ("2017-07", 2017.55, "pop2017", "July 2017"),
    ("2022-08", 2022.65, "pop2022", "August 2022"),
    ("2025", 2025.65, "pop2025", "August 2025"),
]
SNAPSHOT_ORDER = [s[0] for s in SNAPSHOTS]
SNAPSHOT_YEAR = {s: y for s, y, _, _ in SNAPSHOTS}
SNAPSHOT_LABEL = {s: lab for s, _, _, lab in SNAPSHOTS}
# Previous high-coverage snapshot used for baseline-attention matching.
PREVIOUS_POP = {"2017-07": "pop2016", "2022-08": "pop2017", "2025": "pop2022"}
LASTFM_YEAR = 2017.55


def norm_key(value: object) -> str:
    """Normalized song/artist key used for cross-source exact linkage.

    Lower-cases, removes featuring/with connectors, strips punctuation and
    accents, and collapses whitespace. This is the linkage convention that
    reproduces the authoritative study support counts.
    """
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*(feat\.|featuring|ft\.|with|y)\s+", " ", text)
    text = re.sub(r"[&x+/]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def norm_alnum(value: object) -> str:
    """Punctuation-free alphanumeric key used for artist-catalog grouping and
    for the movie-metadata / time-respecting exposure joins."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def parse_money(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = re.sub(r"[^0-9.]", "", str(value))
    if text in {"", "."}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_flex(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        kwargs = dict(kwargs)
        kwargs.setdefault("encoding", "latin1")
        return pd.read_csv(path, **kwargs)


def sem(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return np.nan
    return vals.std(ddof=1) / math.sqrt(len(vals))


def smd(treated: pd.Series, control: pd.Series) -> float:
    t = pd.to_numeric(treated, errors="coerce").dropna()
    c = pd.to_numeric(control, errors="coerce").dropna()
    if len(t) <= 1 or len(c) <= 1:
        return np.nan
    pooled = math.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    if pooled == 0 or pd.isna(pooled):
        return 0.0
    return float((t.mean() - c.mean()) / pooled)


def ci_from_model(model, term: str) -> tuple[float, float, float, float, float]:
    coef = float(model.params.get(term, np.nan))
    se = float(model.bse.get(term, np.nan))
    p = float(model.pvalues.get(term, np.nan))
    return coef, se, coef - 1.96 * se, coef + 1.96 * se, p


def ci_mean(x) -> tuple[float, float, float, float, float]:
    """Mean, SE, 1.96-SE interval, and t-test p-value for a vector of paired differences."""
    vals = pd.Series(x).dropna().astype(float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    est = float(vals.mean())
    if n < 2:
        return est, np.nan, np.nan, np.nan, np.nan
    se = float(vals.std(ddof=1) / np.sqrt(n))
    p = float(2 * stats.t.sf(abs(est / se), df=n - 1)) if se > 0 else np.nan
    return est, se, est - 1.96 * se, est + 1.96 * se, p


def model_row(mod, term: str, comparison: str, n: int, extra: dict | None = None) -> dict:
    extra = extra or {}
    est = float(mod.params.get(term, np.nan))
    se = float(mod.bse.get(term, np.nan))
    row = {
        "comparison": comparison,
        "estimate": est,
        "std_error": se,
        "conf_low": est - 1.96 * se if np.isfinite(se) else np.nan,
        "conf_high": est + 1.96 * se if np.isfinite(se) else np.nan,
        "p_value": float(mod.pvalues.get(term, np.nan)),
        "n": int(n),
    }
    row.update(extra)
    return row


def show_and_save_table(df: pd.DataFrame, path: Path, max_rows: int | None = None, **kwargs) -> pd.DataFrame:
    """Display the live analytical object inline, then save the same object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display(df if max_rows is None else df.head(max_rows))
    df.to_csv(path, index=False, **kwargs)
    return df


def show_and_save_figure(fig, path: Path, dpi: int = 300) -> None:
    """Display the live figure object inline, then save the same object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display(fig)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    import matplotlib.pyplot as plt

    plt.close(fig)


def check(condition: bool, message: str) -> None:
    """Notebook-level validation assertion with a readable message."""
    if not condition:
        raise AssertionError(f"VALIDATION FAILED: {message}")
    print(f"PASS: {message}")


# ---------------------------------------------------------------------------
# Shared figure style and helpers for the paper artifacts generated inside the
# thematic notebooks (02-05).
# ---------------------------------------------------------------------------

BLUE, DARK_BLUE, LIGHT_BLUE = "#3B5B8A", "#1F3A5F", "#8EA6C8"
GRAY, LIGHT_GRAY, CHARCOAL, ACCENT, GREEN, GRID = "#6F6F6F", "#B8B8B8", "#2F2F2F", "#A24B4B", "#5F8F5F", "#E6E6E6"
SNAP_SHORT = {"2016-10": "2016", "2017-07": "2017", "2022-08": "2022", "2025": "2025"}

COV_LABEL = {
    "age": "Song age", "baseline_age": "Baseline song age", "chart_year": "Chart debut year",
    "weeks": "Billboard weeks", "peak_pos": "Peak chart position",
    "previous_popularity": "Prior Spotify popularity", "baseline_popularity": "Baseline Spotify popularity",
    "artist_catalog_size_excl_song": "Artist catalog size, excluding focal song",
    "artist_total_weeks_excl_song": "Artist total Billboard weeks, excluding focal song",
    "artist_mean_weeks_excl_song": "Artist mean Billboard weeks, excluding focal song",
    "artist_best_peak_excl_song": "Artist best other-song peak position",
    "artist_superstar_top1pct": "Artist superstar indicator, top 1 percent",
    "artist_billboard_songs": "Artist Billboard catalog size",
}


def set_plot_style() -> None:
    """Manuscript figure style shared by every figure-generating notebook."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def errorbarh(ax, est, lo, hi, y, color=BLUE, size=30) -> None:
    est, lo, hi = np.atleast_1d(est).astype(float), np.atleast_1d(lo).astype(float), np.atleast_1d(hi).astype(float)
    y = np.atleast_1d(y)
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="none", ecolor=color, elinewidth=1.2, capsize=2.5, zorder=1)
    ax.scatter(est, y, s=size, color=color, zorder=2)


def errorbarv(ax, x, est, lo, hi, color=BLUE, size=30) -> None:
    est, lo, hi = np.asarray(est, float), np.asarray(lo, float), np.asarray(hi, float)
    ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", color=color, ecolor=color, elinewidth=1.1, capsize=2.5, markersize=np.sqrt(size))
