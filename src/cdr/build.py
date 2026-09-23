"""Raw-data validation and analytical panel construction.

All construction starts from the preserved raw study inputs under ``data/raw``.
The Billboard song-artist universe is rebuilt from the weekly Hot 100 chart
file; the preserved one-row-per-song aggregate is used only as a cross-check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .util import (
    LASTFM_YEAR,
    PREVIOUS_POP,
    SNAPSHOTS,
    norm_alnum,
    norm_key,
    parse_money,
    read_csv_flex,
    sha256,
)


def validate_raw_checksums() -> pd.DataFrame:
    checks = pd.read_csv(paths.DATA / "raw_checksums.csv")
    rows = []
    for _, row in checks.iterrows():
        path = paths.PROJECT_ROOT / row["filename"]
        if not path.exists():
            status = "MISSING"
        else:
            status = "OK" if sha256(path) == row["sha256"] else "CHECKSUM MISMATCH"
        rows.append({"filename": row["filename"], "role": row["role"], "status": status})
    report = pd.DataFrame(rows)
    bad = report[report["status"].ne("OK")]
    if len(bad):
        raise RuntimeError("Raw-data validation failed:\n" + bad.to_string(index=False))
    return report


def rebuild_billboard_from_hot100() -> pd.DataFrame:
    """Rebuild the Billboard song-artist universe from the weekly chart file.

    One record per (song title string, performer string): chart debut date,
    total weeks on the Hot 100, and best (minimum) peak position.
    """
    weekly = read_csv_flex(paths.RAW_2025 / "Hot100_2025.csv", low_memory=False)
    weekly["chart_debut"] = pd.to_datetime(weekly["chart_debut"], errors="coerce")
    weekly["chart_date"] = pd.to_datetime(weekly["chart_date"], errors="coerce")
    # The cumulative worst_position field restarts when a song recharts; the
    # preserved aggregate records the value from the song's debut chart week.
    debut_week = weekly.sort_values("chart_date").drop_duplicates(["song", "performer"], keep="first")
    debut_worst = debut_week.set_index(["song", "performer"])["worst_position"]
    grouped = (
        weekly.groupby(["song", "performer"], sort=True)
        .agg(
            chart_debut=("chart_debut", "min"),
            number_of_weeks=("time_on_chart", "max"),
            maximum_consecutive_weeks=("consecutive_weeks", "max"),
            maximum_peak_position=("peak_position", "min"),
            n_weekly_rows=("chart_position", "size"),
        )
        .reset_index()
        .rename(columns={"performer": "artist"})
    )
    grouped["minimum_worst_position"] = (
        grouped.set_index(["song", "artist"]).index.map(debut_worst).astype(int)
    )
    return grouped


def load_preserved_billboard() -> pd.DataFrame:
    return read_csv_flex(paths.RAW_2025 / "one_row_per_song.csv", parse_dates=["chart_debut"], low_memory=False)


def billboard_key_diagnostics(bill: pd.DataFrame) -> pd.DataFrame:
    """Records that remain distinct song-artist records but would collapse
    under an over-aggressive punctuation-free normalization of the combined
    title/artist string (the source of a historical 4,680 vs 4,683 film-linked
    count discrepancy)."""
    d = bill.copy()
    d["alnum_id"] = d["song"].map(norm_alnum) + "||" + d["artist"].map(norm_alnum)
    counts = d["alnum_id"].value_counts()
    dup_ids = counts[counts > 1].index
    return d[d["alnum_id"].isin(dup_ids)].sort_values("alnum_id")[
        ["song", "artist", "chart_debut", "number_of_weeks", "maximum_peak_position", "alnum_id"]
    ]


def prep_snapshot(path, pop_col: str, out_col: str, song_col: str = "Song", artist_col: str = "Artist") -> pd.DataFrame:
    frame = read_csv_flex(path, low_memory=False)
    frame = frame.rename(columns={song_col: "song", artist_col: "artist", pop_col: out_col})
    frame = frame[["song", "artist", out_col]].copy()
    frame["song_key"] = frame["song"].map(norm_key)
    frame["artist_key"] = frame["artist"].map(norm_key)
    return frame.drop_duplicates(["song_key", "artist_key"])


def load_spotify_snapshots() -> dict[str, pd.DataFrame]:
    return {
        "pop2016": prep_snapshot(paths.RAW_2025 / "datos_spotify_octubre_2016.csv", "Max.Popularity", "pop2016"),
        "pop2017": prep_snapshot(paths.RAW_2025 / "datos_spotify_julio_2017.csv", "Max Popularity", "pop2017"),
        "pop2022": prep_snapshot(paths.RAW_2025 / "datos_spotify_agosto_2022.csv", "Max.Popularity", "pop2022"),
        "pop2025": prep_snapshot(
            paths.RAW_2025 / "data_bill_spoty_merged.csv", "popularity", "pop2025", "song", "artist"
        ),
    }


def load_imdb_links() -> pd.DataFrame:
    imdb = read_csv_flex(paths.RAW_2025 / "data_completa_IMDb.csv", low_memory=False)
    imdb = imdb[imdb["pelicula"].astype(str) != "NA"].copy()
    imdb["movie_year"] = pd.to_numeric(imdb["lanzamiento"], errors="coerce")
    imdb["movie_id"] = imdb["ID"].astype(str)
    return imdb


def load_boxoffice() -> pd.DataFrame:
    box = read_csv_flex(paths.RAW_2025 / "data_completa_boxoffice.csv", low_memory=False)
    box = box.rename(columns={c: c.replace(".x", "") for c in box.columns})
    box["movie_id"] = box["ID"].astype(str)
    for col in ["Domestic", "International", "Worldwide", "Budget", "DomesticOpening"]:
        if col in box.columns:
            box[col + "_num"] = box[col].map(parse_money)
    return box


def build_master_panel(bill: pd.DataFrame) -> pd.DataFrame:
    """Song-level master: Billboard history + four Spotify snapshots +
    film linkage (exact normalized song/artist keys) + Last.fm listeners."""
    master = bill.copy()
    master["song_key"] = master["song"].map(norm_key)
    master["artist_key"] = master["artist"].map(norm_key)
    master = master[
        ["song", "artist", "song_key", "artist_key", "chart_debut", "number_of_weeks", "maximum_peak_position"]
    ].copy()

    for out_col, snap in load_spotify_snapshots().items():
        master = master.merge(snap[["song_key", "artist_key", out_col]], on=["song_key", "artist_key"], how="left")

    imdb = load_imdb_links()
    imdb["song_key"] = imdb["title"].map(norm_key)
    imdb["artist_key"] = imdb["artist"].map(norm_key)
    movie_counts = (
        imdb.groupby(["song_key", "artist_key"])
        .agg(n_movies=("movie_id", "nunique"), first_year=("movie_year", "min"), last_movie_year=("movie_year", "max"))
        .reset_index()
    )
    master = master.merge(movie_counts, on=["song_key", "artist_key"], how="left")
    master["n_movies"] = pd.to_numeric(master["n_movies"], errors="coerce").fillna(0).astype(int)
    master["film_linked"] = master["first_year"].notna().astype(int)
    master["chart_year"] = pd.to_datetime(master["chart_debut"]).dt.year
    master["weeks"] = pd.to_numeric(master["number_of_weeks"], errors="coerce")
    master["peak_pos"] = pd.to_numeric(master["maximum_peak_position"], errors="coerce")
    master["log_weeks"] = np.log1p(master["weeks"])
    master["movie_count_cat"] = pd.cut(
        master["n_movies"], [-0.1, 0.5, 1.5, 3.5, np.inf], labels=["0", "1", "2-3", "4+"]
    ).astype(str)
    master["log_movies"] = np.log1p(master["n_movies"])
    # Alphanumeric keys used for artist-catalog grouping and metadata joins.
    master["song_key_alnum"] = master["song"].map(norm_alnum)
    master["artist_key_alnum"] = master["artist"].map(norm_alnum)
    master["song_artist_id"] = master["song_key_alnum"] + "||" + master["artist_key_alnum"]

    lastfm = build_lastfm_support()
    master = master.merge(
        lastfm[["song_key", "artist_key", "lastfm_listeners", "log_lastfm_listeners"]],
        on=["song_key", "artist_key"],
        how="left",
    )
    return master


def build_lastfm_support() -> pd.DataFrame:
    """July 2017 Last.fm cumulative listener reach, reconstructed from the raw
    Last.fm snapshot and the preserved July 2017 Billboard-Spotify-film linkage
    input. The raw field is ``Listeners``; the analytical transformation is
    log1p cumulative listeners."""
    last = read_csv_flex(paths.RAW_LASTFM / "data_lastfm_Jul31_2017.csv", low_memory=False)
    base = read_csv_flex(paths.RAW_THESIS / "BSPO_jul17_aparece.csv", low_memory=False)
    last["song_key"] = last["Song"].map(norm_key)
    last["artist_key"] = last["Artist"].map(norm_key)
    base["song_key"] = base["Song"].map(norm_key)
    base["artist_key"] = base["Artist"].map(norm_key)
    last["lastfm_listeners"] = pd.to_numeric(last["Listeners"], errors="coerce")
    last["log_lastfm_listeners"] = np.log1p(last["lastfm_listeners"])
    out = base.merge(
        last[["song_key", "artist_key", "lastfm_listeners", "log_lastfm_listeners"]],
        on=["song_key", "artist_key"],
        how="left",
    )
    return out[out["lastfm_listeners"].notna()].drop_duplicates(["song_key", "artist_key"])


def add_expanded_movie_metadata(master: pd.DataFrame) -> pd.DataFrame:
    """Movie metadata and worldwide box office linked through exact
    alphanumeric song/artist keys and exact internal movie IDs."""
    imdb = load_imdb_links()
    imdb["song_key_alnum"] = imdb["title"].map(norm_alnum)
    imdb["artist_key_alnum"] = imdb["artist"].map(norm_alnum)
    box = load_boxoffice()
    keep_cols = ["movie_id", "Worldwide_num", "Budget_num", "DomesticOpening_num", "Genres", "MPAA", "RunningTime"]
    imdb_box = imdb.merge(box[[c for c in keep_cols if c in box.columns]].drop_duplicates("movie_id"), on="movie_id", how="left")

    linked = master[["song_key_alnum", "artist_key_alnum"]].drop_duplicates().merge(
        imdb_box, on=["song_key_alnum", "artist_key_alnum"], how="inner"
    )
    agg = (
        linked.groupby(["song_key_alnum", "artist_key_alnum"])
        .agg(
            movie_rows_expanded=("movie_id", "size"),
            unique_movie_ids_expanded=("movie_id", "nunique"),
            expanded_first_movie_year=("movie_year", "min"),
            max_worldwide=("Worldwide_num", "max"),
            max_budget=("Budget_num", "max"),
            n_with_genres=("Genres", lambda s: int(s.replace("", np.nan).notna().sum())),
        )
        .reset_index()
    )
    out = master.merge(agg, on=["song_key_alnum", "artist_key_alnum"], how="left")
    out["has_movie_metadata"] = out["movie_rows_expanded"].notna().astype(int)
    out["has_boxoffice"] = out["max_worldwide"].notna().astype(int)
    out["has_genres"] = (out["n_with_genres"].fillna(0) > 0).astype(int)
    q75 = out.loc[out["film_linked"].eq(1), "max_worldwide"].quantile(0.75)
    out["high_boxoffice_q75"] = ((out["max_worldwide"] >= q75) & out["max_worldwide"].notna()).astype(int)
    out["film_visibility_cat"] = "Never film-linked"
    out.loc[out["film_linked"].eq(1) & out["max_worldwide"].isna(), "film_visibility_cat"] = "Film-linked, box office missing"
    out.loc[
        out["film_linked"].eq(1) & out["max_worldwide"].notna() & (out["max_worldwide"] < q75), "film_visibility_cat"
    ] = "Film-linked, lower box office"
    out.loc[out["film_linked"].eq(1) & (out["max_worldwide"] >= q75), "film_visibility_cat"] = "Film-linked, top-quartile box office"
    return out


def movie_metadata_coverage(master: pd.DataFrame) -> pd.DataFrame:
    film = master[master["film_linked"].eq(1)]
    n = len(film)
    rows = [
        ("movie metadata via exact song-artist / internal movie ID", int(film["has_movie_metadata"].sum())),
        ("worldwide box office", int(film["max_worldwide"].notna().sum())),
        ("production budget", int(film["max_budget"].notna().sum())),
        ("genre", int(film["has_genres"].sum())),
        ("IMDb votes / rating / country / language", 0),
    ]
    return pd.DataFrame(
        [
            {
                "variable": label,
                "available": avail,
                "missing": n - avail,
                "n_film_linked_records": n,
                "share_missing": (n - avail) / n,
            }
            for label, avail in rows
        ]
    )


def build_spotify_panel(master: pd.DataFrame) -> pd.DataFrame:
    """Long song-snapshot panel over the four high-coverage Spotify snapshots.

    ``attention`` is the platform 0-100 popularity index. ``previous_popularity``
    is the same song's popularity in the previous high-coverage snapshot (used
    by baseline-attention matching designs)."""
    rows = []
    keep = [
        "song",
        "artist",
        "song_key",
        "artist_key",
        "song_key_alnum",
        "artist_key_alnum",
        "song_artist_id",
        "film_linked",
        "chart_year",
        "weeks",
        "log_weeks",
        "peak_pos",
        "first_year",
        "n_movies",
        "movie_count_cat",
        "log_movies",
        "max_worldwide",
        "film_visibility_cat",
    ]
    artist_cols = [
        "artist_billboard_songs",
        "artist_total_billboard_weeks",
        "artist_catalog_size_excl_song",
        "artist_total_weeks_excl_song",
        "artist_mean_weeks_excl_song",
        "artist_best_peak_excl_song",
        "artist_superstar_top1pct",
        "artist_has_treated_song",
    ]
    keep += [c for c in artist_cols if c in master.columns]
    for snap, year, col, label in SNAPSHOTS:
        d = master[keep + [col]].copy()
        d["snapshot"] = snap
        d["snapshot_label"] = label
        d["snapshot_year"] = year
        d["attention"] = pd.to_numeric(d[col], errors="coerce")
        d["age"] = year - d["chart_year"]
        d["previous_popularity"] = (
            pd.to_numeric(master[PREVIOUS_POP[snap]], errors="coerce") if snap in PREVIOUS_POP else np.nan
        )
        rows.append(d.drop(columns=[col]))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["attention"].notna()].copy()
    panel["age_sq"] = panel["age"] ** 2
    panel["rel_time"] = panel["snapshot_year"] - pd.to_numeric(panel["first_year"], errors="coerce")
    return panel


def build_lastfm_panel(master: pd.DataFrame) -> pd.DataFrame:
    d = master.copy()
    d["snapshot"] = "2017-07"
    d["snapshot_year"] = LASTFM_YEAR
    d["attention"] = pd.to_numeric(d["log_lastfm_listeners"], errors="coerce")
    d["age"] = LASTFM_YEAR - d["chart_year"]
    d["age_sq"] = d["age"] ** 2
    d["previous_popularity"] = pd.to_numeric(d["pop2016"], errors="coerce")
    return d[d["attention"].notna()].copy()


def build_time_respecting_exposures(master: pd.DataFrame) -> pd.DataFrame:
    """Snapshot-specific film exposure using only film appearances (and box
    office) realized by each Spotify snapshot date. Joins use alphanumeric
    keys and exact internal movie IDs."""
    imdb = load_imdb_links()
    imdb["song_key_alnum"] = imdb["title"].map(norm_alnum)
    imdb["artist_key_alnum"] = imdb["artist"].map(norm_alnum)
    imdb = imdb[imdb["song_key_alnum"].ne("") & imdb["artist_key_alnum"].ne("")]
    imdb = imdb.drop_duplicates(["song_key_alnum", "artist_key_alnum", "movie_id", "movie_year"])
    box = load_boxoffice()
    links = imdb.merge(box[["movie_id", "Worldwide_num"]].drop_duplicates("movie_id"), on="movie_id", how="left")
    links = links.merge(
        master[["song_key_alnum", "artist_key_alnum"]].drop_duplicates(),
        on=["song_key_alnum", "artist_key_alnum"],
        how="inner",
    )
    links = links[links["movie_year"].notna()].copy()
    frames = []
    for snap, year, _, _ in SNAPSHOTS:
        d = links[links["movie_year"] <= year]
        agg = (
            d.groupby(["song_key_alnum", "artist_key_alnum"])
            .agg(
                movie_count_by_snapshot=("movie_id", "nunique"),
                max_worldwide_boxoffice_by_snapshot=("Worldwide_num", "max"),
                first_film_year_observed_by_snapshot=("movie_year", "min"),
            )
            .reset_index()
        )
        agg["snapshot"] = snap
        agg["film_linked_by_snapshot"] = 1
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def merge_time_respecting(panel: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(exposures, on=["song_key_alnum", "artist_key_alnum", "snapshot"], how="left")
    for col in ["film_linked_by_snapshot", "movie_count_by_snapshot"]:
        out[col] = out[col].fillna(0).astype(int)
    out["movie_count_by_snapshot_cat"] = pd.cut(
        out["movie_count_by_snapshot"], bins=[-0.1, 0.5, 1.5, 3.5, np.inf], labels=["0", "1", "2-3", "4+"]
    ).astype(str)
    return out


def add_artist_features(master: pd.DataFrame) -> pd.DataFrame:
    """Artist/band-level historical covariates from alphanumeric catalog keys.

    ``artist_best_peak_excl_song`` is the best (minimum) peak position among
    the artist's other Billboard songs; NaN for single-song catalogs."""
    out = master.copy()
    agg = (
        out.groupby("artist_key_alnum")
        .agg(
            artist_billboard_songs=("song_key_alnum", "nunique"),
            artist_total_billboard_weeks=("weeks", "sum"),
            artist_film_song_count=("film_linked", "sum"),
        )
        .reset_index()
    )
    out = out.merge(agg, on="artist_key_alnum", how="left")
    out["artist_catalog_size_excl_song"] = out["artist_billboard_songs"] - 1
    out["artist_total_weeks_excl_song"] = out["artist_total_billboard_weeks"] - out["weeks"]
    out["artist_mean_weeks_excl_song"] = np.where(
        out["artist_catalog_size_excl_song"] > 0,
        out["artist_total_weeks_excl_song"] / out["artist_catalog_size_excl_song"],
        np.nan,
    )
    best_excl = []
    for _, g in out.groupby("artist_key_alnum"):
        peaks = pd.to_numeric(g["peak_pos"], errors="coerce")
        for idx, _ in peaks.items():
            others = peaks.drop(index=idx).dropna()
            best_excl.append((idx, others.min() if len(others) else np.nan))
    out["artist_best_peak_excl_song"] = out.index.to_series().map(pd.Series(dict(best_excl)))
    q99 = out.drop_duplicates("artist_key_alnum")["artist_total_billboard_weeks"].quantile(0.99)
    out["artist_superstar_top1pct"] = (out["artist_total_billboard_weeks"] >= q99).astype(int)
    out["artist_has_treated_song"] = (out["artist_film_song_count"] > out["film_linked"]).astype(int)
    return out
