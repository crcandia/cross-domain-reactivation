# Cross-Domain Reactivation of Collective Attention in Music and Film

Code and notebooks for the study of song reuse in films and subsequent attention to songs and artist catalogs. The analyses link Billboard Hot 100 records to Spotify popularity, Last.fm listeners, IMDb soundtrack appearances, and Box Office Mojo metadata.

The repository includes the notebooks, Python modules in `src/cdr/`, aggregate results in `outputs/data/results/`, and LaTeX tables in `outputs/tables/`.

## Data

A full rerun requires the original inputs under `data/raw/`. These files and the record-level analytical panels are excluded from GitHub. The filenames, roles, and SHA-256 checksums of the preserved inputs are listed in [`data/raw_checksums.csv`](data/raw_checksums.csv).

The data cover four Spotify snapshots (October 2016, July 2017, August 2022, and August 2025) and a July 2017 Last.fm snapshot. The film-visibility analysis uses the recorded worldwide box-office amounts with `log1p`, without an additional inflation adjustment.

Stored stochastic replications are included in `data/inference/`. Reference values in `data/expected_results.csv` are used to check the results.

## Setup

```bash
conda env create -f environment.yml
conda activate cross-domain-reactivation
jupyter notebook
```

## Running the notebooks

With the required inputs in place, run the notebooks in the order below, using a fresh kernel for each.

| Notebook | Analysis |
|---|---|
| `00_raw_data_and_billboard` | Reconstruct the Billboard song-artist universe |
| `00b_raw_data_checks` | Check raw files and the reconstructed universe |
| `01_linkage_and_analytical_panels` | Link sources and build the analytical panels |
| `02_attention_regimes_and_matching` | Compare attention levels and matched songs |
| `02b_integrated_inference_checkpoint` | Validate or regenerate stochastic replications |
| `03_memorydecay_and_dormant_reactivation` | Fit attention baselines and estimate temporal contrasts |
| `04_embedding_visibility_and_temporal` | Analyze repeated reuse, film visibility, and event time |
| `05_artist_catalog_and_lastfm` | Analyze artist catalogs and Last.fm reach |
| `06_validation_and_manuscript` | Check results, annotate linkage validation, and synchronize local manuscript files |

Notebook 02b controls the expensive stochastic calculation:

```python
REGENERATE_INFERENCE = False
```

The default, `False`, reuses the 500 pseudo-event assignments and 250 nested-bootstrap replications after checking their code and input hashes, version, counts, seeds, and analytical sample. Set it to `True` to regenerate them. Notebook 03 then checks the files and recomputes the summaries and intervals from those replications.

Notebook 06 additionally requires the manual review file at `outputs/data/validation/camila_linkage_adjudicated.csv` and the local `manuscript/` sources. These are excluded from GitHub, so the public checkout cannot run that notebook in full on its own.

## Citation

Candia, C., Utreras-Cifuentes, C., & Droguett, F. (2026). *Cross-Domain Reactivation of Collective Attention in Music and Film*. Unpublished manuscript.
