"""Regenerate four SI tables from existing result objects, without refitting models.

Run from the repository root with:
    PYTHONPATH=src python -m cdr.publication_table_refresh
An optional --overleaf-dir writes the same generated tables into that project's
PNAS_2026_09_08 directory. The result CSVs are read-only inputs. No .tex table is
used as an input. The same cdr.tex writers are used by the analytical notebooks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from . import tex
from .tex import fmt_num, fmt_int, fmt_ci, fmt_p


def _cols(widths, left=2):
    return ''.join(
        (r'>{\raggedright\arraybackslash}' if i < left else r'>{\centering\arraybackslash}')
        + f'p{{{w}\\linewidth}}' for i, w in enumerate(widths)
    )


def _read(path, columns):
    d = pd.read_csv(path, low_memory=False)
    missing = set(columns).difference(d.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return d


def generate(root: Path, overleaf_dir: Path | None = None) -> list[Path]:
    root = root.resolve()
    results = root / 'outputs/data/results'
    destination = root / 'outputs/tables/si'
    destination.mkdir(parents=True, exist_ok=True)
    written = []

    # Exact result rows and original layouts are retained. cdr.tex applies units
    # and presentation conventions before writing, including in notebook runs.
    for catalog in (False, True):
        kind = 'catalog' if catalog else 'song'
        data = _read(results / f'{kind}_matching_inference.csv', [
            'comparison', 'variant', 'n_matched_pairs', 'n_unique_controls',
            'estimate', 'naive_paired_se', 'naive_ci_low', 'naive_ci_high',
            'cluster_control_se', 'cluster_control_ci_low', 'cluster_control_ci_high'
        ])
        rows = []
        for _, r in data.iterrows():
            clustered = pd.notna(r['cluster_control_ci_low'])
            lo = r['cluster_control_ci_low'] if clustered else r['naive_ci_low']
            hi = r['cluster_control_ci_high'] if clustered else r['naive_ci_high']
            row = [r['comparison'], r['variant'], fmt_int(r['n_matched_pairs']),
                   fmt_int(r['n_unique_controls']), fmt_num(r['estimate']),
                   fmt_num(r['naive_paired_se'], 3)]
            if not catalog:
                row.append(fmt_num(r['boot_treated_se'], 3))
            row += [fmt_num(r['cluster_control_se'], 3), fmt_ci(lo, hi)]
            rows.append(row)
        header = ['Comparison', 'Variant', 'Pairs', r'\makecell[b]{Unique\\ controls}',
                  'Estimate', r'\makecell[b]{Naive\\ SE}']
        if not catalog:
            header.append(r'\makecell[b]{Bootstrap\\ SE\\ (treated)}')
        header += [r'\makecell[b]{Cluster\\ SE\\ (control)}', r'95\% CI']
        name = f'table_si_{kind}_matching_inference.tex'
        output = destination / name
        if catalog:
            tex.write_longtable(output,
                'Replacement-aware inference and design sensitivity for the matched catalog premium.',
                'tab:si-catalog-matching-inference', header, rows,
                col_spec=_cols([.19, .17, .052, .078, .082, .052, .082, .105]),
                escape_cells=False, stretch=1.05)
        else:
            tex.write_table(output,
                'Replacement-aware inference and design sensitivity for the song-level matched premium (August 2025).',
                'tab:si-song-matching-inference', header, rows,
                col_spec=_cols([.135, .15, .058, .078, .082, .052, .09, .082, .10]),
                size=r'\footnotesize', escape_cells=False, stretch=1.05, tabcolsep='2.5pt')
        written.append(output)

    # Support values come from the deterministic selection summary, rather than
    # numbers embedded in the manuscript or copied from a generated table.
    support = _read(root/'outputs/data/validation/dormancy_definition_diagnostic.csv',
                    ['quantity', 'value']).set_index('quantity')['value']
    summary = _read(results/'dormant_integrated_inference.csv',
                    ['quantity', 'value']).set_index('quantity')['value']
    rows = [
        ['film-linked songs with pre-film and post-film Spotify snapshots', fmt_int(support['eligible film-linked records with pre/post support'])],
        ['bottom-tercile pre-film excess-attention threshold (q33)', fmt_num(support['pre-film excess-attention cutoff'], 3)],
        ['lower-tercile pre-film excess-attention treated songs', fmt_int(support['lower-tercile records'])],
    ]
    mapping = [
        ('Pseudo-event assignments', 'n_pseudo_event_assignments'),
        ('Assignments with positive contrast', 'n_positive_assignments'),
        ('Assignment mean (integrated point estimate)', 'assignment_mean'),
        ('Assignment median', 'assignment_median'),
        ('Assignment SD', 'assignment_sd'),
        ('Assignment interval, 2.5th pct', 'assignment_interval_low'),
        ('Assignment interval, 97.5th pct', 'assignment_interval_high'),
        (r'Integrated bootstrap 95\% CI, low', 'dormant_excess_integrated_ci_low'),
        (r'Integrated bootstrap 95\% CI, high', 'dormant_excess_integrated_ci_high'),
        ('Valid nested-bootstrap replications', 'valid_bootstrap_replications'),
        ('Treated above-threshold probability', 'treated_exit_probability'),
        ('Matched-control above-threshold probability', 'control_exit_probability'),
        ('Above-threshold probability difference (pp)', 'exit_difference_pp'),
        (r'Above-threshold integrated 95\% CI, low (pp)', 'exit_integrated_ci_low'),
        (r'Above-threshold integrated 95\% CI, high (pp)', 'exit_integrated_ci_high'),
        ('Decay-equivalent years (median)', 'dormant_decay_equivalent_years'),
        (r'Decay-equivalent 95\% CI, low', 'dormant_decay_integrated_ci_low'),
        (r'Decay-equivalent 95\% CI, high', 'dormant_decay_integrated_ci_high'),
        ('Inversion failure rate', 'inversion_failure_rate'),
        ('Median 95th-pct control reuse', 'median_p95_control_reuse'),
        ('Maximum control reuse', 'max_control_reuse'),
        ('Matched lower-tercile treated songs per assignment', 'n_matched_pairs_per_assignment'),
    ]
    rows += [[label, fmt_num(summary[key], 3)] for label, key in mapping]
    note = (
        f'Inference integrates {fmt_int(summary["n_pseudo_event_assignments"])} pseudo-event assignments and '
        f'{fmt_int(summary["valid_bootstrap_replications"])} nested-bootstrap replications with snapshot-specific '
        'MemoryDecay refitting, excess-attention reconstruction, pseudo-event reassignment, and rematching. '
        'The assignment interval is a design-stability distribution across pseudo-event assignments, not a confidence interval. '
        f'The analytical support comprises {fmt_int(support["eligible film-linked records with pre/post support"])} '
        f'film-linked songs with pre-film and post-film snapshots; the {fmt_int(support["lower-tercile records"])} '
        'songs in the lower tercile of pre-film excess attention are matched in every pseudo-event assignment.'
    )
    output = destination/'table_si_dormant_integrated_inference.tex'
    tex.write_table(output,
        'Lower pre-film excess attention: support and Monte Carlo-integrated inference.',
        'tab:si-dormant-integrated-inference', ['Quantity', 'Value'], rows, note=note,
        col_spec=_cols([.52, .18], left=1), size=r'\footnotesize', escape_cells=False, stretch=1.05)
    written.append(output)

    # Retain precisely the pooled models in the existing SI comparison.
    base = _read(results/'pooled_premium_cluster_rows.csv', ['model', 'term', 'estimate'])
    extended = _read(results/'embedding_visibility_cluster_rows.csv', ['model', 'term', 'estimate'])
    robustness = _read(results/'robustness_cluster_rows.csv', ['model', 'term', 'estimate'])
    catalog = robustness[robustness['model'].eq('Pooled artist-catalog regression (final specification)')]
    if len(catalog) != 1:
        raise ValueError('Expected one pooled artist-catalog regression.')
    data = pd.concat([base, extended, catalog], ignore_index=True)
    rows = []
    for _, r in data.iterrows():
        term = str(r['term']).replace('_', ' ')
        if str(r['model']).startswith('Film visibility'):
            term = 'log1p max recorded worldwide box office; films released by snapshot year'
        rows.append([r['model'], term, fmt_int(r['n_observations']), fmt_int(r['n_clusters']),
                     fmt_num(r['estimate']), fmt_num(r['hc1_se'], 3), fmt_num(r['cluster_se'], 3),
                     fmt_ci(r['cluster_ci_low'], r['cluster_ci_high']), fmt_p(r['cluster_p'])])
    output = destination/'table_si_cluster_inference.tex'
    tex.write_longtable(output,
        'Cluster-by-song inference for pooled models with repeated song-snapshot observations.',
        'tab:si-cluster-inference',
        ['Model', 'Term', 'N obs.', 'N clusters', 'Estimate', 'SE', 'SE', r'95\% CI', '$p$'],
        rows, col_spec=_cols([.19, .15, .055, .062, .082, .045, .045, .10, .072]),
        escape_cells=False, stretch=1.05,
        group_header=[('',1),('',1),('Support',2),('',1),('HC1',1),('Cluster-by-song inference',3)])
    written.append(output)

    manuscript_tables = root/'manuscript/Tables/SI'
    manuscript_tables.mkdir(parents=True, exist_ok=True)
    for p in written:
        shutil.copy2(p, manuscript_tables/p.name)
        if overleaf_dir is not None:
            target = overleaf_dir/'PNAS_2026_09_08'
            if not target.is_dir():
                raise FileNotFoundError(target)
            shutil.copy2(p, target/p.name)
    return written


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--overleaf-dir', type=Path)
    args=parser.parse_args()
    for path in generate(args.root, args.overleaf_dir):
        print(path)


if __name__ == '__main__':
    main()
