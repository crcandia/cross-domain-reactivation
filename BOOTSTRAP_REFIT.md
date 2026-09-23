# Bootstrap baseline-refit correction

This note documents the correction that aligns the lower-pre-film integrated bootstrap with the primary MemoryDecay estimator. It does not change the point-estimate matching design, event dates, lower-tercile rule, or other analytical models.

## What was corrected

`src/cdr/dormant.py` reconstructs resampled never-film song trajectories and calls the same `aggregate_age_means` and `fit_canonical_gridsearch` functions used by the point-estimate analysis. The bootstrap therefore uses the same floor-age cells, mean fractional age within each cell, positive-response filtering, weighted log-response squared-error objective, capped inverse-age weights, log/logit parameterization, and full multi-start grid.

The previous bounded/log1p/soft-L1 refitter is not used for the reported inference. Replications retain deterministic consecutive seeds. Failed fits or empty comparisons raise explicit errors rather than silently shortening the percentile distribution. Snapshot-specific fitted parameters and objective values are stored with every bootstrap replication.

Stored inference is bound to an inference version, source-code hashes, input hashes, row counts, seed sequence, and analytical-support fingerprint. Legacy or incompatible replication files are rejected rather than silently reused.

## Canonical notebook workflow

Notebook 02b (`notebooks/02b_integrated_inference_checkpoint.ipynb`) is now the primary reuse/regeneration interface for the expensive stochastic inference. It exposes:

```python
REGENERATE_INFERENCE = False
```

With the default `False`, notebook 02b validates the distributed 500 pseudo-event assignment rows and 250 nested-bootstrap rows against the current code, inputs, inference version, deterministic seeds, file checksums, row counts, and raw-derived analytical-support fingerprint.

With `REGENERATE_INFERENCE = True`, notebook 02b regenerates all 500 pseudo-event assignments and 250 nested-bootstrap replications from the current analytical objects, writes the new rows and metadata to `data/inference/`, and validates the written files. Notebook 03 then independently validates the active replication files and recomputes every reported summary, interval, table, and figure from those rows.

The checkpoint and analysis notebooks are therefore directly inspectable and executable in order without requiring a project-wide wrapper script. Reuse is allowed only as a computational shortcut for an exactly compatible stochastic calculation, never as an unvalidated analytical input.

## Optional batch utility

`src/cdr/bootstrap_rebuild.py` remains available as an advanced resumable/parallel utility for long batch execution. It is not the canonical user-facing reproduction path. The paper can be followed and reproduced through the notebooks in numerical order.

## Tests

From the project root:

```bash
PYTHONPATH=src python -m cdr.bootstrap_refit_tests
```

The test suite covers full-grid identity refits, near-coincident rates, trajectory sampling, deterministic seeds, explicit failure handling, cache/version checks, checkpoint resume, and sequential/parallel equivalence. These are software consistency tests, not a proof of nominal bootstrap coverage.

## Scientific scope

The correction addresses estimator consistency inside the nested bootstrap. It does not redesign the observational estimand or resolve general limitations such as annual film-date resolution, unequal calendar windows, or all possible dependence structures.

The current manuscript reports the corrected empirical stochastic inference. The point estimate for the primary lower-pre-film contrast remains 5.04 excess-attention points. Bootstrap-based uncertainty and decay-equivalent translations are generated from the corrected replication rows.

## Rollback

The pre-correction implementation is preserved under:

```text
scratch/2026-09-10_bootstrap_refit/before/
```

The notebook-workflow version preceding the 2026-09-22 interactive execution update is preserved under:

```text
scratch/2026-09-22_notebook_reproducibility_update/before/
```
