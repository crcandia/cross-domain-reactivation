# Public GitHub view versus the full local project

The local Dropbox project is the complete author working copy. It retains all
preserved raw inputs, record-level analytical panels, internal validation
material, rollback folders, and submission staging.

Git tracks a deliberately lighter shareable view. `.gitignore` excludes files
from Git only. It does not delete them from the local project.

## Kept local but not committed

- `data/raw/`: source-level third-party inputs. Some are large, and some have
  redistribution restrictions or incomplete redistribution provenance.
  The preserved Box Office Mojo input is specifically not redistributed
  without an appropriate license.
- `Inputs_No_Upload/`: thesis/source documents retained for provenance.
- `scratch/` and `PNAS_Submission_*`: internal backups, review packages, and
  submission staging.
- record-level derived panels under `outputs/data/` and row-level
  matching/temporal files that reproduce third-party records.
- internal manual-validation material under `outputs/data/validation/`.

## Committed and shareable

- the ordered Jupyter notebooks and reusable source code;
- environment, manifest, citation, and documentation files;
- `data/inference/`, including the validated 500 pseudo-event and 250
  nested-bootstrap replication rows;
- aggregate/model result objects under `outputs/data/results/`;
- publication tables and all figure-generating code; rendered figure PDFs are regenerated locally;
- analysis-linked manuscript sources, where included.

The public repository is therefore suitable for reading and auditing the full
analysis logic, inspecting aggregate results, exploring notebooks, and
reusing/validating the distributed stochastic inference. A complete
raw-to-output rerun requires the author-held local inputs documented in
`MANIFEST.md`.

The full local Dropbox directory remains the source-complete working copy.
