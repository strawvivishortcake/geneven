"""Step 7 - reporting.

Renders the two deliverables defined in the scope: the evidence-gap report
(Module 01) and the safety signal report (Module 02). Every report prints the
specification hash, the grid size and the pinned reference versions, so a
reader can see what produced it and regenerate it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import RunManifest
from .spec import AnalysisSpec


def _fmt(value: float, places: int = 2, infinite: str = "not reachable") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if np.isinf(value):
        return infinite
    return f"{value:,.{places}f}"


def _header(spec: AnalysisSpec, manifest: RunManifest) -> list[str]:
    refs = manifest.references
    lines = [
        f"**Study** {spec.study_id} | **Drug** {spec.drug}",
        "",
        f"- Specification `{spec.spec_hash()}` | code `{manifest.code_commit[:12]}`",
        f"- Data snapshot `{manifest.data_snapshot_id}` extracted {manifest.data_extracted_at}",
        f"- References: PharmVar {refs.pharmvar}, CPIC {refs.cpic}, "
        f"PharmGKB {refs.pharmgkb}, build {refs.genome_build}",
        f"- Test grid: {spec.grid_size()} pre-specified subgroup-event pairs",
        f"- Thresholds: power {spec.thresholds.power:.0%}, alpha {spec.thresholds.alpha}, "
        f"meaningful RR {spec.thresholds.meaningful_rr}, "
        f"minimum cell {spec.thresholds.min_cell_size}",
    ]
    warnings = manifest.warnings()
    if warnings:
        lines += ["", "> **Reproducibility warnings**"]
        lines += [f"> - {w}" for w in warnings]
    return lines


def evidence_gap_report(
    sufficiency: pd.DataFrame,
    coverage: pd.DataFrame,
    enrollment: pd.DataFrame,
    spec: AnalysisSpec,
    manifest: RunManifest,
) -> str:
    """Module 01 output: what this dataset can and cannot establish."""
    counts = sufficiency["verdict"].value_counts().to_dict()
    n_total = len(sufficiency)
    n_undetermined = counts.get("undetermined", 0)

    lines = [f"# Evidence gap report: {spec.drug}", ""]
    lines += _header(spec, manifest)
    lines += [
        "",
        "## Verdict summary",
        "",
        f"Of {n_total} pre-specified subgroup-event pairs, "
        f"{counts.get('supported', 0)} are **supported**, "
        f"{counts.get('limited', 0)} are **limited** and "
        f"{n_undetermined} are **undetermined**.",
        "",
    ]
    if n_undetermined:
        lines += [
            f"The {n_undetermined} undetermined pairs are not evidence of safety. "
            "The data could not have revealed a clinically meaningful risk in "
            "these subgroups, so no conclusion is drawn in either direction.",
            "",
        ]

    lines += [
        "| Subgroup | Event | Exposed | Events | Observed rate (95% CI) | "
        "Min detectable RR | Verdict |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in sufficiency.itertuples():
        ci = f"{row.observed_rate:.3f} ({row.rate_ci_low:.3f}-{row.rate_ci_high:.3f})"
        lines.append(
            f"| {row.subgroup} | {row.event} | {row.n_exposed:,} | "
            f"{row.events_exposed:,} | {ci} | {_fmt(row.min_detectable_rr)} | "
            f"{row.verdict} |"
        )

    if not coverage.empty:
        lines += [
            "",
            "## Trial versus expected treated population",
            "",
            "Expected shares come from population allele frequencies and the "
            "indication's epidemiology, not from the trial.",
            "",
            "| Subgroup | In trial | Observed share | Expected share | Shortfall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in coverage.itertuples():
            flag = " **under**" if row.underrepresented else ""
            lines.append(
                f"| {row.display} | {row.observed_n:,} | {row.observed_share:.1%} | "
                f"{row.expected_share:.1%}{flag} | {row.shortfall_n:,.0f} |"
            )

    if not enrollment.empty:
        lines += [
            "",
            "## Enrollment needed to reach `supported`",
            "",
            "| Subgroup | Currently exposed | Worst detectable RR | Additional patients |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in enrollment.itertuples():
            lines.append(
                f"| {row.display} | {row.n_exposed:,} | {_fmt(row.worst_mdrr)} | "
                f"{_fmt(row.additional_needed, 0)} |"
            )

    return "\n".join(lines) + "\n"


def signal_report(
    findings: pd.DataFrame,
    spec: AnalysisSpec,
    manifest: RunManifest,
    calibration: dict | None = None,
) -> str:
    """Module 02 output: ranked findings with their evidence tiers."""
    lines = [f"# Safety signal report: {spec.drug}", ""]
    lines += _header(spec, manifest)

    if findings.empty:
        lines += [
            "",
            "## Findings",
            "",
            "No subgroup-event pair cleared the evidence sufficiency gate, so no "
            "signal testing was performed. See the evidence gap report for the "
            "enrollment that would change this.",
        ]
        return "\n".join(lines) + "\n"

    confirmatory = findings[findings["confirmatory"]]
    significant = confirmatory[confirmatory["q_value"] <= spec.thresholds.alpha]

    lines += [
        "",
        "## Findings",
        "",
        f"{len(findings)} pairs were tested out of {spec.grid_size()} in the grid. "
        f"{len(significant)} confirmatory findings survive FDR control at "
        f"q <= {spec.thresholds.alpha}.",
        "",
        "Effect sizes are odds ratios for the genotype x exposure interaction: "
        "the factor by which the drug-event association differs in this "
        "subgroup relative to the rest of the treated population.",
        "",
        "| Rank | Subgroup | Event | Tier | Interaction OR (95% CI) | q | Model | Priority |",
        "| ---: | --- | --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in findings.itertuples():
        ci = (
            f"{_fmt(row.interaction_or)} ({_fmt(row.ci_low)}-{_fmt(row.ci_high)})"
            if np.isfinite(row.interaction_or) else "-"
        )
        rank = getattr(row, "rank", "-")
        tier = getattr(row, "tier", "-")
        priority = getattr(row, "priority_score", float("nan"))
        lines.append(
            f"| {rank} | {row.subgroup} | {row.event} | {tier} | {ci} | "
            f"{_fmt(row.q_value, 4)} | {row.model} | {_fmt(priority, 3)} |"
        )

    notes = findings[findings["note"].astype(bool)]
    if not notes.empty:
        lines += ["", "### Notes", ""]
        for row in notes.itertuples():
            lines.append(f"- **{row.subgroup} / {row.event}**: {row.note}")

    if calibration and calibration.get("n_controls"):
        lines += [
            "",
            "### Negative control calibration",
            "",
            f"{calibration['observed_significant']} of {calibration['n_controls']} "
            f"negative-control pairs were significant at 0.05 "
            f"(expected {calibration['expected_significant']:.1f}). "
            f"Median p-value {calibration['median_p']:.3f}.",
        ]
        if calibration.get("inflated"):
            lines.append(
                "\n> Residual confounding is inflating the test statistics. "
                "P-values should be empirically calibrated before these findings "
                "are reported."
            )

    return "\n".join(lines) + "\n"


def write_reports(
    output_dir: str | Path,
    sufficiency: pd.DataFrame,
    coverage: pd.DataFrame,
    enrollment: pd.DataFrame,
    findings: pd.DataFrame,
    spec: AnalysisSpec,
    manifest: RunManifest,
    calibration: dict | None = None,
) -> dict[str, Path]:
    """Write every output and record each one in the manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    gap_path = out / "evidence_gap_report.md"
    gap_path.write_text(
        evidence_gap_report(sufficiency, coverage, enrollment, spec, manifest)
    )
    written["evidence_gap_report"] = gap_path

    signal_path = out / "safety_signal_report.md"
    signal_path.write_text(signal_report(findings, spec, manifest, calibration))
    written["safety_signal_report"] = signal_path

    for name, frame in [
        ("sufficiency", sufficiency),
        ("coverage_gap", coverage),
        ("enrollment_plan", enrollment),
        ("findings", findings),
    ]:
        if frame is not None and not frame.empty:
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = path

    for name, path in written.items():
        manifest.record_output(name, path)
    manifest_path = manifest.write(out / "run_manifest.json")
    written["run_manifest"] = manifest_path
    return written
