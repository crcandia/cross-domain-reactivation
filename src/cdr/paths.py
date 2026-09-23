"""Project-relative paths used by the notebooks and helper modules."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking up until raw_checksums.csv is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "data" / "raw_checksums.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root (data/raw_checksums.csv).")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
RAW_2025 = RAW / "2025_Data"
RAW_LASTFM = RAW / "lastfm_spotify_2016_2017"
RAW_THESIS = RAW / "student_thesis_linkage"
INFERENCE = DATA / "inference"
OUT = PROJECT_ROOT / "outputs"
OUT_DATA = OUT / "data"
RESULTS = OUT_DATA / "results"
OUT_TABLES = OUT / "tables"
OUT_TABLES_MAIN = OUT_TABLES / "main"
OUT_TABLES_SI = OUT_TABLES / "si"
OUT_FIGURES = OUT / "figures"
OUT_FIG_MAIN = OUT_FIGURES / "main"
OUT_FIG_SI = OUT_FIGURES / "si"
OUT_PDFS = OUT / "pdfs"
MANUSCRIPT = PROJECT_ROOT / "manuscript"


def ensure_output_dirs() -> None:
    for path in [
        OUT_DATA,
        RESULTS,
        OUT_TABLES_MAIN,
        OUT_TABLES_SI,
        OUT_FIG_MAIN,
        OUT_FIG_SI,
        OUT_PDFS,
    ]:
        path.mkdir(parents=True, exist_ok=True)
