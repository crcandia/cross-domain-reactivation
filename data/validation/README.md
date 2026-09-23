# Manual evidence annotations

`camila_linkage_error_annotations.csv` contains the evidence typology for the
20 unconfirmed records in the reconciled July 2026 linkage review. It is a
manual annotation input, separate from all estimation data. Sample IDs and
screenshot hashes identify the records and evidence. The supporting screenshots
remain in the original `Inputs_Camila_Julio2026` folder.

Notebook 06 joins these annotations to
`outputs/data/validation/camila_linkage_adjudicated.csv`, preserves all review
fields and labels, marks the other 180 records as not assessed for this
typology, and generates the enriched review and count summaries. Original
review fields remain in Spanish. `tipo_discrepancia` describes the evidence
rather than replacing `clasificacion_camila` or `clasificacion_adjudicada`.

Expanded performer credits and unresolved solo-artist/band names must not be
treated automatically as covers. A different credited performer does not prove
that an additional use of the original performance is absent. No master-recording
identifier was verified, and no composition-level precision is estimated.
