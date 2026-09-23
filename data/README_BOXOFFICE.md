# Box-Office Data

The complete author/coauthor replication package includes the preserved movie box-office input used in the study:

`data/raw/2025_Data/data_completa_boxoffice.csv`

Provider-level provenance points to Box Office Mojo through the documented student thesis data lineage. The file was preserved in the study repository and linked to soundtrack records through the internal movie identifier. Exact row-level URLs, retrieval dates, and redistribution rights are not available in the repository.

The current analysis uses the monetary values as recorded in this preserved input. The executable pipeline parses the recorded currency strings to numeric values and does not apply an inflation, present-value, or 2024-dollar conversion. The film-visibility specification applies `log1p` to maximum recorded worldwide box office after restricting eligible films by release year. A September 2026 provenance check against identifiable Box Office Mojo titles found that the preserved values correspond to the nominal reported box-office amounts rather than inflation-adjusted 2024 dollars.

The student thesis documents a present-value adjustment in its original preprocessing description. That historical statement is retained as provenance but is not treated as the transformation implemented by the current preserved input and executable pipeline, because the repository contains neither a reproducible inflation-adjustment formula nor adjusted values distinct from the nominal Box Office Mojo amounts used here.

For a public OSF upload, review the file listed in `data/public_exclude.txt` before redistribution. Removing this file would limit full raw-data reproduction of the film-visibility analyses.
