# Cross-Domain Reactivation of Collective Attention in Music and Film

Code and notebooks for the study of song reuse in films and subsequent attention to songs and artist catalogs. The analyses link Billboard Hot 100 records to Spotify popularity, Last.fm listeners, IMDb soundtrack appearances, and Box Office Mojo metadata.

## Reviewing the results

The [notebooks](notebooks/) contain the analysis code and saved figures and tables. Numerical results are available as CSV files in [`outputs/data/results/`](outputs/data/results/), with LaTeX tables in [`outputs/tables/`](outputs/tables/).

These saved results can be reviewed without running the analyses.

## Reproducing the results

A full rerun requires the original input files under `data/raw/`, which are not distributed in this repository. Their filenames and descriptions are listed in [`data/raw_checksums.csv`](data/raw_checksums.csv). The data cover four Spotify snapshots (October 2016, July 2017, August 2022, and August 2025) and a July 2017 Last.fm snapshot.

From the repository root, create the environment and open Jupyter:

```bash
conda env create -f environment.yml
conda activate cross-domain-reactivation
jupyter notebook
```

With the required inputs in place, run the notebooks in the order below, using a fresh kernel for each.

| Notebook | Analysis |
|---|---|
| `00_raw_data_and_billboard` | Reconstruct the Billboard song-artist universe |
| `00b_raw_data_checks` | Check raw files and the reconstructed universe |
| `01_linkage_and_analytical_panels` | Link sources and build the analytical panels |
| `02_attention_regimes_and_matching` | Compare attention levels and matched songs |
| `02b_integrated_inference_checkpoint` | Load or regenerate stochastic replications |
| `03_memorydecay_and_dormant_reactivation` | Fit attention baselines and estimate temporal contrasts |
| `04_embedding_visibility_and_temporal` | Analyze repeated reuse, film visibility, and event time |
| `05_artist_catalog_and_lastfm` | Analyze artist catalogs and Last.fm reach |

Notebook 02b uses the supplied 500 pseudo-event assignments and 250 nested-bootstrap replications by default (`REGENERATE_INFERENCE = False`). Set this option to `True` to rerun that calculation. Notebook 03 computes the summaries and intervals from the replication results.

Running the notebooks generates figures in `outputs/figures/`, tables in `outputs/tables/`, and numerical results in `outputs/data/results/`. The figures and tables are also displayed within the notebooks.

## Citation

Candia, C., Utreras-Cifuentes, C., & Droguett, F. (2026). *Cross-Domain Reactivation of Collective Attention in Music and Film*. Unpublished manuscript.
