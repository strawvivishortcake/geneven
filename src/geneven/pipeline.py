"""Pipeline orchestration and command-line entry point.

Runs Steps 4 to 7 over a prepared analysis table. Steps 1 to 3 (acquisition,
OMOP harmonization, diplotype calling) happen upstream and hand this stage a
table; see :mod:`geneven.genomics` for the profile assembly that produces the
phenotype columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .prioritize import prioritize
from .provenance import ReferenceVersions, RunManifest
from .report import write_reports
from .signals import calibrate_with_negative_controls, run_signals
from .spec import AnalysisSpec
from .sufficiency import coverage_gap, enrollment_plan, run_sufficiency


REQUIRED_COLUMNS = {"person_id", "exposed"}


def validate_analysis_table(table: pd.DataFrame, spec: AnalysisSpec) -> list[str]:
    """Check the table carries everything the specification asks for."""
    problems = []
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        problems.append(f"missing required columns: {sorted(missing)}")

    for event in spec.events:
        if event not in table.columns:
            problems.append(f"missing event column: {event}")
        elif not table[event].dropna().isin([0, 1]).all():
            problems.append(f"event column {event} must be 0/1")

    for subgroup in spec.subgroups:
        column = f"pheno_{subgroup.gene}"
        if column not in table.columns:
            problems.append(f"missing phenotype column: {column}")

    if "exposed" in table.columns and not table["exposed"].dropna().isin([0, 1]).all():
        problems.append("exposed column must be 0/1")

    for i in range(spec.ancestry_components):
        if f"pc{i + 1}" not in table.columns:
            problems.append(
                f"missing ancestry component pc{i + 1} "
                f"(spec asks for {spec.ancestry_components})"
            )
    return problems


def run_analysis(
    analysis_table: pd.DataFrame,
    spec: AnalysisSpec,
    manifest: RunManifest,
    output_dir: str | Path,
    severity_map: dict[str, str] | None = None,
    known_pairs: set[tuple[str, str]] | None = None,
) -> dict:
    """Run Steps 4 to 7 and write every output.

    Returns the frames and the paths written, so callers can inspect results
    without re-reading from disk.
    """
    problems = validate_analysis_table(analysis_table, spec)
    if problems:
        raise ValueError(
            "analysis table does not match the specification:\n  - "
            + "\n  - ".join(problems)
        )

    # Step 4 - sufficiency gates everything downstream.
    sufficiency = run_sufficiency(analysis_table, spec)
    coverage = coverage_gap(analysis_table, spec)
    enrollment = enrollment_plan(sufficiency, spec)

    # Step 5 - only eligible pairs are tested.
    findings = run_signals(analysis_table, spec, sufficiency)

    negative_ids = {
        (nc["subgroup"], nc["event"]) for nc in spec.negative_controls
    }
    calibration = (
        calibrate_with_negative_controls(findings, negative_ids)
        if not findings.empty and negative_ids
        else None
    )

    # Step 6 - tiers and ranking.
    if not findings.empty:
        findings = prioritize(
            findings, spec, severity_map=severity_map, known_pairs=known_pairs
        )

    # Step 7 - reports and manifest.
    paths = write_reports(
        output_dir, sufficiency, coverage, enrollment, findings,
        spec, manifest, calibration,
    )

    return {
        "sufficiency": sufficiency,
        "coverage": coverage,
        "enrollment": enrollment,
        "findings": findings,
        "calibration": calibration,
        "paths": paths,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="geneven",
        description="Run the Geneven sufficiency and signal pipeline.",
    )
    parser.add_argument("--spec", required=True, help="path to the analysis specification YAML")
    parser.add_argument("--data", required=True, help="path to the analysis table CSV or Parquet")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--snapshot-id", default="unset", help="data snapshot identifier")
    parser.add_argument("--extracted-at", default="unset", help="snapshot extraction date")
    parser.add_argument("--cpic-version", default="unset")
    parser.add_argument("--pharmvar-version", default="unset")
    parser.add_argument("--pharmgkb-version", default="unset")
    args = parser.parse_args(argv)

    spec = AnalysisSpec.from_yaml(args.spec)
    print(spec.describe(), file=sys.stderr)

    data_path = Path(args.data)
    table = (
        pd.read_parquet(data_path)
        if data_path.suffix in {".parquet", ".pq"}
        else pd.read_csv(data_path)
    )

    manifest = RunManifest(
        study_id=spec.study_id,
        spec_hash=spec.spec_hash(),
        data_snapshot_id=args.snapshot_id,
        data_extracted_at=args.extracted_at,
        references=ReferenceVersions(
            cpic=args.cpic_version,
            pharmvar=args.pharmvar_version,
            pharmgkb=args.pharmgkb_version,
        ),
        random_seed=spec.random_seed,
    )
    manifest.record_input("analysis_table", data_path)
    manifest.record_input("spec", args.spec)

    result = run_analysis(table, spec, manifest, args.out)

    counts = result["sufficiency"]["verdict"].value_counts().to_dict()
    print(f"\nSufficiency verdicts: {counts}", file=sys.stderr)
    print(f"Signals tested: {len(result['findings'])}", file=sys.stderr)
    for name, path in result["paths"].items():
        print(f"  {name}: {path}", file=sys.stderr)
    for warning in manifest.warnings():
        print(f"  WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
