# Manifest

## Execution contract

The numbered notebooks are the authoritative project workflow. They are designed to be opened, inspected, modified, and executed interactively in order from fresh kernels. Each notebook reads explicit files from the project tree, regenerates its analytical outputs, and saves the same live analytical objects that are shown inline where applicable.

There is no required project-wide runner. A user should be able to understand and reproduce the paper by following the notebooks directly.

The one intentionally reusable expensive computation is the stochastic integrated inference. Notebook 02b is the explicit checkpoint and exposes `REGENERATE_INFERENCE = False` by default. Stored replication rows may be reused only after validation against the current code, inputs, inference version, row counts, deterministic seeds, and analytical-support fingerprint. Setting the switch to `True` regenerates and overwrites those replication rows before any reported summary is computed.

## Raw Input (`data/raw/`)

- `2025_Data/Hot100_2025.csv` — weekly Billboard Hot 100 chart entries, source-level basis of the song-artist universe.
- `2025_Data/one_row_per_song.csv` — preserved song-artist aggregate, used only as a reconstruction cross-check and not as a primary runtime input.
- `2025_Data/datos_spotify_octubre_2016.csv`, `datos_spotify_julio_2017.csv`, `datos_spotify_agosto_2022.csv`, `data_bill_spoty_merged.csv` — four high-coverage Spotify popularity snapshots.
- `2025_Data/data_completa_IMDb.csv` — IMDb soundtrack song-film linkage from the student-thesis data lineage.
- `2025_Data/data_completa_boxoffice.csv` — preserved Box Office Mojo movie metadata and worldwide box office. The current pipeline parses the recorded amounts but does not perform an additional inflation/present-value adjustment.
- `2025_Data/songs_movies_processed.RData` — legacy song-movie cache retained only for lineage cross-checks, excluded from the primary runtime.
- `lastfm_spotify_2016_2017/data_lastfm_Jul31_2017.csv` — Last.fm July 2017 listener snapshot.
- `lastfm_spotify_2016_2017/all_data_billboard_30_Jul_2017.csv` — Billboard companion table preserved with the July 2017 collection.
- `lastfm_spotify_2016_2017/Spoty_Final_ENE_2016_2.RData` — legacy January 2016 processed cache, excluded from all analyses and runtime.
- `student_thesis_linkage/BSPO_jul17_aparece.csv` — July 2017 Billboard-Spotify-film linkage input used to reconstruct Last.fm analytical support.
- `data/raw_checksums.csv` — SHA-256 checksums and roles for raw inputs.

## Stored stochastic inference (`data/inference/`)

- `dormant_pseudoevent_monte_carlo_replications.csv` — 500 pseudo-event assignment rows with deterministic seeds.
- `dormant_integrated_bootstrap_replications.csv` — 250 full nested-bootstrap rows with deterministic seeds and snapshot-specific MemoryDecay refit diagnostics.
- `inference_metadata.json` — checksums, code/input hashes, inference version, row counts, seed strategy, and analytical-support fingerprint.

Notebook 02b is the canonical reuse/regeneration interface for these files. It either validates and reuses them or regenerates them from the current analytical objects. Notebook 03 then independently validates the active files and calculates all reported summaries from their replication-level rows.

## Validation contract

- `data/expected_results.csv` — support counts and headline-result references used by notebook 06 and targeted notebook checks. These values validate output; they are not used to fit models or calculate estimates.

## Analysis code

- `notebooks/00_raw_data_and_billboard.ipynb` — rebuilds the Billboard universe from the weekly chart file.
- `notebooks/00b_raw_data_checks.ipynb` — raw-data checksums and construction cross-checks.
- `notebooks/01_linkage_and_analytical_panels.ipynb` — linkage, analytical panels, exposure construction, support objects, Main Table 1 and SI coverage tables.
- `notebooks/02_attention_regimes_and_matching.ipynb` — attention models, song-level matching, replacement-aware inference, Figure 1 and associated tables.
- `notebooks/02b_integrated_inference_checkpoint.ipynb` — visible switch for validated reuse versus full regeneration of the 500 pseudo-event and 250 nested-bootstrap replication rows.
- `notebooks/03_memorydecay_and_dormant_reactivation.ipynb` — MemoryDecay, excess attention, temporal comparisons, summaries from validated active stochastic-inference rows, Main Table 2, Figure 2 and associated SI outputs.
- `notebooks/04_embedding_visibility_and_temporal.ipynb` — repeated embedding, film visibility, event-time models, Figure 3 and associated tables.
- `notebooks/05_artist_catalog_and_lastfm.ipynb` — artist-catalog, Last.fm and robustness analyses, Figure 4 and associated tables.
- `notebooks/06_validation_and_manuscript.ipynb` — artifact registry, numerical and structural checks, manuscript figure/table synchronization.
- `src/cdr/` — reusable mechanical implementations. Substantive filters, sample construction, formulas and estimands remain visible in the notebooks.
- `environment.yml` — conda environment.

## Manuscript source

- `manuscript/` contains the repository snapshot used for analysis-linked compilation and release checks.
- The active working manuscript for this project is maintained in Dropbox at `/Aplicaciones/Overleaf/Film-Linked Reactivation of Music in Collective Memory/`.
- `manuscript/Figures/{Main,SI}` and `manuscript/Tables/{Main,SI}` are refreshed from generated `outputs/` artifacts by notebook 06.

## Generated output (`outputs/`)

- `data/` — analytical panels and result objects.
- `data/artifact_trace.csv` — retained-artifact registry and claim mapping.
- `data/validation_results.csv` — final validation results.
- `tables/main/`, `tables/si/` — generated publication-facing LaTeX tables.
- `figures/main/`, `figures/si/` — generated publication figures.
- `pdfs/` — compiled repository manuscript/SI snapshots when compilation is run.
