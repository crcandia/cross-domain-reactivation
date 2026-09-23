"""LaTeX table writers for the curated main and SI artifacts.

Every writer displays the live table inline at generation time (generate ->
show inline -> save), so the notebooks remain working notebooks and no table
exists only as an exported .tex file.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _plain_text(value: object) -> str:
    """Readable inline rendering of a LaTeX header/cell for notebook display."""
    text = "" if value is None else str(value)
    text = re.sub(r"\\makecell(?:\[[a-z]\])?\{(.*)\}", r"\1", text, flags=re.S)
    text = text.replace(r"\\", " ").replace(r"\%", "%").replace(r"\ ", " ")
    text = re.sub(r"\\text[a-z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text).replace("$", "").replace("{", "").replace("}", "")
    return " ".join(text.split())


def _display_inline(caption: str, label: str, header: list[str], rows: list[list[object]]) -> None:
    """Show the exact table being written, as a live object, inside the notebook."""
    try:
        from IPython.display import Markdown, display
    except ImportError:  # imported outside a notebook: nothing to display
        return
    df = pd.DataFrame([[_plain_text(c) for c in row] for row in rows],
                      columns=[_plain_text(h) for h in header])
    display(Markdown(f"**{_plain_text(caption)}** (`{label}`)"))
    display(df)


def tex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_num(x: object, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_int(x: object) -> str:
    if pd.isna(x):
        return "--"
    return f"{int(round(float(x))):,}"


def fmt_p(x: object) -> str:
    if pd.isna(x):
        return "--"
    x = float(x)
    return "$<0.001$" if x < 0.001 else f"{x:.3f}"


def fmt_ci(lo: object, hi: object, digits: int = 2) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "--"
    return f"[{float(lo):.{digits}f}, {float(hi):.{digits}f}]"



def _current_bootstrap_table_summary() -> dict:
    """Reject stale stored inference before a table describes the corrected fit."""
    import json
    from . import dormant, paths

    metadata_path = paths.INFERENCE / "inference_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("Regenerate corrected inference before writing its table.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dormant.validate_inference_files(metadata.get("dormant_analytical_support_fingerprint", ""))
    mc = pd.read_csv(paths.INFERENCE / "dormant_pseudoevent_monte_carlo_replications.csv")
    boot = pd.read_csv(paths.INFERENCE / "dormant_integrated_bootstrap_replications.csv")
    return dormant.summarize_stored_inference(mc, boot)


def _check_bootstrap_table_values(rows, summary) -> None:
    """Do not label an old result CSV as corrected merely because a new cache exists."""
    keys = {
        "integrated bootstrap 95% ci, low": "dormant_excess_integrated_ci_low",
        "integrated bootstrap 95% ci, high": "dormant_excess_integrated_ci_high",
        "above-threshold integrated 95% ci, low (pp)": "exit_integrated_ci_low",
        "above-threshold integrated 95% ci, high (pp)": "exit_integrated_ci_high",
        "decay-equivalent years (median)": "dormant_decay_equivalent_years",
        "decay-equivalent 95% ci, low": "dormant_decay_integrated_ci_low",
        "decay-equivalent 95% ci, high": "dormant_decay_integrated_ci_high",
    }
    found = set()
    for quantity, value in rows:
        key = keys.get(_plain_text(quantity).lower())
        if key is None:
            continue
        found.add(key)
        expected = float(summary[key])
        if not np.isfinite(expected) and _plain_text(value) == "--":
            continue
        try:
            observed = float(str(value).replace(",", ""))
        except ValueError as exc:
            raise RuntimeError(f"Invalid bootstrap table value: {quantity}") from exc
        # Notebook exports may round once to four decimals and again to three.
        if not np.isfinite(observed) or not np.isclose(observed, expected, rtol=0, atol=0.000551):
            raise RuntimeError(f"Stale bootstrap table value for {quantity}. Regenerate notebook 03 outputs.")
    if found != set(keys.values()):
        raise RuntimeError("Incomplete bootstrap interval rows in the table source.")


def _publication_fields(caption, label, header, rows, note):
    """Apply table-specific units and formatting before display and serialization.

    Numeric estimates, uncertainty limits, samples, and matching designs are
    unchanged. These conventions are shared by notebook and script generation.
    """
    header = list(header)
    rows = [list(row) for row in rows]
    if label in {"tab:si-song-matching-inference", "tab:si-catalog-matching-inference"}:
        header = [r"\makecell[b]{Control\\ IDs}" if _plain_text(h) == "Unique controls" else h for h in header]
        identity_note = (
            "Control IDs are alphanumeric song-artist identifiers. No-replacement "
            "matching excludes repeated record indices, not different records sharing an ID."
        )
        if label == "tab:si-song-matching-inference":
            note = (
                "Matching with replacement reuses controls. The cluster bootstrap over "
                "reused control IDs is the primary uncertainty check, with the treated-pair "
                "bootstrap and naive paired SE shown for comparison. No-replacement, "
                "stricter-caliper, and coarsened-exact-matching (chart decade "
                r"$\times$ baseline-popularity quintile) variants are design sensitivities. "
                "The interval column reports the cluster-by-control interval where computed "
                "and the naive interval otherwise. " + identity_note
            )
        else:
            note = (
                "Cluster-by-control intervals account for control reuse in the "
                "prior-attention catalog comparison. Under this inference, the August 2022 "
                "contrast is not distinguishable from zero. Its sign varies across matching "
                "designs: the no-replacement estimate is positive, whereas the stricter-caliper "
                "and coarsened-exact-matching estimates are negative. " + identity_note
            )
    elif label == "tab:si-cluster-inference":
        note = (
            "For the pooled models listed here, the table compares HC1 and cluster-by-song "
            "standard errors using the same specification and exposure definition. "
            "Clusters are defined by alphanumeric song-title and artist identifiers, "
            "not by exact Billboard record strings. Multiple exact records can therefore "
            "share a cluster. The listed estimates retain their sign and exclude zero "
            "under cluster-by-song inference. These comparisons address repeated "
            "song-snapshot dependence and do not change the estimands."
        )
    elif label == "tab:si-dormant-integrated-inference":
        _check_bootstrap_table_values(rows, _current_bootstrap_table_summary())
        integer_quantities = {
            "film-linked songs with pre-film and post-film spotify snapshots",
            "lower-tercile pre-film excess-attention treated songs",
            "pseudo-event assignments", "assignments with positive contrast",
            "valid nested-bootstrap replications", "maximum control reuse",
            "matched lower-tercile treated songs per assignment",
        }
        for row in rows:
            quantity = _plain_text(row[0]).lower()
            if quantity in integer_quantities:
                number = float(str(row[1]).replace(",", ""))
                if not np.isfinite(number) or not number.is_integer():
                    raise ValueError(f"Noninteger count in {label}: {row[0]}={row[1]}")
                row[1] = fmt_int(number)
            elif quantity == "median 95th-pct control reuse":
                number = float(str(row[1]).replace(",", ""))
                row[1] = fmt_int(number) if number.is_integer() else fmt_num(number, 3)
        addition = (
            "Bootstrap curves use the same floor-age aggregation, mean age within each cell, "
            "log-response squared-error objective, capped inverse-age weights, parameter "
            "constraints, and full starting grid as the point-estimate curves."
        )
        if addition not in note:
            note = note.rstrip() + " " + addition
    return caption, header, rows, note


def _group_header_lines(group_header: list[tuple[str, int]]) -> list[str]:
    """Render a first-level header row from (label, span) pairs.

    Labeled groups spanning more than one column receive a \\cmidrule underneath;
    empty labels produce blank cells above identifying columns.
    """
    cells, rules, col = [], [], 1
    for label, span in group_header:
        if span == 1 and not label:
            cells.append("")
        else:
            cells.append(f"\\multicolumn{{{span}}}{{c}}{{{label}}}")
        if label and span > 1:
            rules.append(f"\\cmidrule(lr){{{col}-{col + span - 1}}}")
        col += span
    return [" & ".join(cells) + r" \\", "".join(rules)] if rules else [" & ".join(cells) + r" \\"]


def write_table(
    path: Path,
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[object]],
    note: str = "",
    col_spec: str | None = None,
    size: str = r"\small",
    escape_cells: bool = True,
    resize: bool = False,
    layout: str = "tabular",
    group_header: list[tuple[str, int]] | None = None,
    stretch: float | None = None,
    tabcolsep: str | None = None,
    placement: str = "!htbp",
) -> None:
    """Write a booktabs table with caption, semantic label, and footnote.

    layout="tabular" is the historical fixed-width behavior; layout="tabularx"
    typesets at exactly \\linewidth with wrapped X/p columns so wide tables stay
    readable without \\resizebox. group_header adds a spanned first header row.
    placement="H" pins a table in source order (needed when a floated table
    would otherwise drift past a following longtable).
    """
    caption, header, rows, note = _publication_fields(caption, label, header, rows, note)
    _display_inline(caption, label, header, rows)
    col_spec = col_spec or ("l" * len(header))
    lines = [f"\\begin{{table}}[{placement}]", r"\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if stretch:
        lines.append(f"\\renewcommand{{\\arraystretch}}{{{stretch}}}")
    if tabcolsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    if resize:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    if layout == "tabularx":
        lines.append(f"\\begin{{tabularx}}{{\\linewidth}}{{{col_spec}}}")
    else:
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    if group_header:
        lines.extend(_group_header_lines(group_header))
    lines.extend([" & ".join(header) + r" \\", r"\midrule"])
    for row in rows:
        cells = [tex_escape(c) if escape_cells else ("" if pd.isna(c) else str(c)) for c in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}" if layout == "tabularx" else r"\end{tabular}"])
    if resize:
        lines.append(r"}")
    if note:
        lines.extend([r"\begin{flushleft}\footnotesize", note, r"\end{flushleft}"])
    lines.append(r"\end{table}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_longtable(
    path: Path,
    caption: str,
    label: str,
    header: list[str],
    rows: list[list[object]],
    note: str = "",
    col_spec: str | None = None,
    size: str = r"\footnotesize",
    escape_cells: bool = True,
    group_header: list[tuple[str, int]] | None = None,
    stretch: float | None = None,
    tabcolsep: str = "2.5pt",
) -> None:
    """Write a longtable that repeats its full (optionally two-level) header on
    every continuation page and marks continuation pages with the table number."""
    caption, header, rows, note = _publication_fields(caption, label, header, rows, note)
    _display_inline(caption, label, header, rows)
    col_spec = col_spec or ("l" * len(header))
    head_lines = (_group_header_lines(group_header) if group_header else []) + [" & ".join(header) + r" \\"]
    lines = [
        f"{{{size}",
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
    ]
    if stretch:
        lines.append(f"\\renewcommand{{\\arraystretch}}{{{stretch}}}")
    lines.extend([
        f"\\begin{{longtable}}{{{col_spec}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        r"\toprule",
        *head_lines,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{" + str(len(header)) + r"}{@{}l}{\footnotesize Table~\thetable{} continued.}\\",
        r"\toprule",
        *head_lines,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ])
    for row in rows:
        cells = [tex_escape(c) if escape_cells else ("" if pd.isna(c) else str(c)) for c in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    if note:
        lines.append(r"\begin{flushleft}\footnotesize " + note + r" \end{flushleft}")
    lines.append("}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
