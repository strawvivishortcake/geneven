"""End-to-end demonstration on synthetic data.

Generates a trial with a deliberately planted genotype-specific toxicity and
a deliberately underpowered subgroup, then runs the full pipeline. The point
of the demo is to show both halves working:

  - the planted signal is found in the subgroup that was adequately enrolled
  - the subgroup that was not adequately enrolled is reported as
    `undetermined`, NOT as safe, even though it shows no signal

The second is the behaviour Geneven exists for. Run:

    python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geneven.pipeline import run_analysis                      # noqa: E402
from geneven.provenance import ReferenceVersions, RunManifest  # noqa: E402
from geneven.spec import AnalysisSpec                          # noqa: E402


# Ground truth planted in the synthetic data.
TRUE_EFFECTS = {
    # CYP2D6 poor metabolizers get a real, large increase in neutropenia.
    ("CYP2D6", "Poor metabolizer", "severe_neutropenia"): 6.0,
    # CYP2C19 ultrarapid metabolizers get a real increase in hepatotoxicity,
    # but they are deliberately under-enrolled so it cannot be detected.
    ("CYP2C19", "Ultrarapid metabolizer", "hepatotoxicity"): 5.0,
}

# How many of each phenotype to enrol. CYP2C19 ultrarapid is starved on
# purpose; CYP2D6 poor is adequately represented.
ENROLMENT = {
    "CYP2C19": {
        "Normal metabolizer": 0.78,
        "Poor metabolizer": 0.11,
        "Ultrarapid metabolizer": 0.011,   # under-enrolled on purpose
        "Intermediate metabolizer": 0.099,
    },
    "CYP2D6": {
        "Normal metabolizer": 0.62,
        "Poor metabolizer": 0.18,
        "Ultrarapid metabolizer": 0.05,
        "Intermediate metabolizer": 0.15,
    },
}

BASE_RATES = {
    "severe_neutropenia": 0.030,
    "hepatotoxicity": 0.020,
    "qt_prolongation": 0.015,
}


def synthesize(n: int = 4000, seed: int = 20261107) -> pd.DataFrame:
    """Build a synthetic trial with known planted effects."""
    rng = np.random.default_rng(seed)

    rows = {
        "person_id": [f"P{i:05d}" for i in range(n)],
        "exposed": rng.binomial(1, 0.7, n),
        "age": rng.normal(58, 12, n).clip(18, 90),
        "sex": rng.choice(["F", "M"], n),
        "comorbidity_burden": rng.poisson(2.0, n),
        "concomitant_inhibitor": rng.binomial(1, 0.15, n),
        "dose_band": rng.choice(["low", "medium", "high"], n),
    }
    for gene, distribution in ENROLMENT.items():
        phenotypes = list(distribution)
        probabilities = np.array([distribution[p] for p in phenotypes], dtype=float)
        probabilities /= probabilities.sum()
        rows[f"pheno_{gene}"] = rng.choice(phenotypes, n, p=probabilities)

    frame = pd.DataFrame(rows)

    # Ancestry components: correlated with phenotype, as in real data, so the
    # adjustment in the models has something to do.
    loading = (frame["pheno_CYP2D6"] == "Poor metabolizer").astype(float)
    for i in range(5):
        noise = rng.normal(0, 1, n)
        frame[f"pc{i + 1}"] = noise + (0.4 * loading if i == 0 else 0.0)

    # Outcomes. Risk rises with exposure, and rises further inside the
    # genotype subgroups carrying a planted effect.
    for event, base in BASE_RATES.items():
        odds = base / (1 - base)
        multiplier = np.ones(n)
        multiplier *= np.where(frame["exposed"] == 1, 1.6, 1.0)
        for (gene, phenotype, target_event), effect in TRUE_EFFECTS.items():
            if target_event != event:
                continue
            hit = (frame[f"pheno_{gene}"] == phenotype) & (frame["exposed"] == 1)
            multiplier *= np.where(hit, effect, 1.0)
        # Mild confounding through comorbidity, which the models adjust for.
        multiplier *= 1.0 + 0.05 * frame["comorbidity_burden"]
        probability = (odds * multiplier) / (1 + odds * multiplier)
        frame[event] = rng.binomial(1, probability.clip(0, 0.95))

    return frame


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    spec = AnalysisSpec.from_yaml(root / "config" / "spec_example.yaml")
    table = synthesize()

    print(spec.describe())
    print(f"\nSynthetic trial: {len(table):,} patients, "
          f"{int(table['exposed'].sum()):,} exposed\n")

    treated = table[table["exposed"] == 1]
    print("Enrolment by phenotype (treated arm):")
    for gene in ("CYP2C19", "CYP2D6"):
        counts = treated[f"pheno_{gene}"].value_counts()
        for phenotype, count in counts.items():
            print(f"  {gene:8s} {phenotype:26s} {count:5,d}")

    manifest = RunManifest(
        study_id=spec.study_id,
        spec_hash=spec.spec_hash(),
        data_snapshot_id="synthetic-demo-001",
        data_extracted_at="2026-11-07",
        references=ReferenceVersions(
            pharmvar="DEMO", cpic="DEMO", pharmgkb="DEMO",
            gnomad="DEMO", reference_panel="DEMO",
        ),
        random_seed=spec.random_seed,
    )

    severity = {
        "severe_neutropenia": "hospitalization",
        "hepatotoxicity": "life_threatening",
        "qt_prolongation": "reversible",
    }

    result = run_analysis(
        table, spec, manifest,
        output_dir=root / "output",
        severity_map=severity,
    )

    print("\n--- Module 01: sufficiency verdicts ---")
    cols = ["subgroup", "event", "n_exposed", "events_exposed",
            "min_detectable_rr", "verdict"]
    print(result["sufficiency"][cols].to_string(index=False))

    print("\n--- Module 02: findings ---")
    findings = result["findings"]
    if findings.empty:
        print("  no pair cleared the sufficiency gate")
    else:
        cols = ["rank", "subgroup", "event", "tier", "interaction_or",
                "q_value", "model", "confirmatory"]
        print(findings[cols].round(4).to_string(index=False))

    print("\n--- Did the demo behave as designed? ---")
    suff = result["sufficiency"]

    detected = findings[
        (findings["subgroup"] == "CYP2D6:Poor metabolizer")
        & (findings["event"] == "severe_neutropenia")
    ] if not findings.empty else pd.DataFrame()
    found = (not detected.empty) and bool(detected.iloc[0]["q_value"] <= 0.05)
    print(f"  planted CYP2D6 PM neutropenia signal detected: {found}")

    starved = suff[
        (suff["subgroup"] == "CYP2C19:Ultrarapid metabolizer")
        & (suff["event"] == "hepatotoxicity")
    ]
    verdict = starved.iloc[0]["verdict"] if not starved.empty else "missing"
    print(f"  under-enrolled CYP2C19 UM hepatotoxicity verdict: {verdict}")
    print(f"    (the planted effect there is real but undetectable; "
          f"reporting it as 'supported' would be the failure mode)")

    print("\nOutputs written:")
    for name, path in result["paths"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
