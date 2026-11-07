"""Tests for Module 01 and the gate it enforces on Module 02.

The central property under test: a subgroup that was never adequately
studied must never be reported the same way as a subgroup that was studied
and found safe, and must never reach the signal engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geneven.spec import AnalysisSpec, Subgroup, SufficiencyThresholds  # noqa: E402
from geneven.sufficiency import (  # noqa: E402
    Verdict,
    classify,
    coverage_gap,
    eligible_for_testing,
    enrollment_plan,
    evaluate_subgroup,
    run_sufficiency,
)


THRESHOLDS = SufficiencyThresholds(
    power=0.80, alpha=0.05, meaningful_rr=2.0,
    limited_multiple=3.0, min_cell_size=20, min_expected_events=5.0,
)


def make_spec(**overrides) -> AnalysisSpec:
    base = dict(
        study_id="TEST",
        drug="testdrug",
        events=["event_a"],
        subgroups=[Subgroup(gene="CYP2D6", phenotype="Poor metabolizer")],
        covariates=["age"],
        ancestry_components=2,
        thresholds=THRESHOLDS,
    )
    base.update(overrides)
    return AnalysisSpec(**base)


def make_table(n_carriers=400, n_others=1600, carrier_events=40,
               other_events=80, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_carriers + n_others
    phenotype = ["Poor metabolizer"] * n_carriers + ["Normal metabolizer"] * n_others
    events = (
        [1] * carrier_events + [0] * (n_carriers - carrier_events)
        + [1] * other_events + [0] * (n_others - other_events)
    )
    return pd.DataFrame({
        "person_id": [f"P{i}" for i in range(n)],
        "exposed": 1,
        "pheno_CYP2D6": phenotype,
        "event_a": events,
        "age": rng.normal(60, 10, n),
        "pc1": rng.normal(size=n),
        "pc2": rng.normal(size=n),
    })


class TestClassify:
    def test_small_subgroup_is_undetermined(self):
        verdict, reason = classify(1.1, n_exposed=5, baseline_rate=0.1,
                                   thresholds=THRESHOLDS)
        assert verdict is Verdict.UNDETERMINED
        assert "minimum cell size" in reason

    def test_small_subgroup_overrides_a_flattering_mdrr(self):
        """A tiny subgroup is undetermined even when the arithmetic looks good."""
        verdict, _ = classify(1.01, n_exposed=4, baseline_rate=0.5,
                              thresholds=THRESHOLDS)
        assert verdict is Verdict.UNDETERMINED

    def test_too_few_expected_events_is_undetermined(self):
        """The normal approximation is not trusted below the event floor.

        30 patients at a 2% baseline would see ~1.2 events even at RR 2.
        The power formula still returns a finite detectable risk; the guard
        must refuse it.
        """
        verdict, reason = classify(4.0, n_exposed=30, baseline_rate=0.02,
                                   thresholds=THRESHOLDS)
        assert verdict is Verdict.UNDETERMINED
        assert "expected" in reason

    def test_detectable_meaningful_risk_is_supported(self):
        verdict, _ = classify(1.8, n_exposed=500, baseline_rate=0.1,
                              thresholds=THRESHOLDS)
        assert verdict is Verdict.SUPPORTED

    def test_middling_detectability_is_limited(self):
        verdict, _ = classify(4.0, n_exposed=500, baseline_rate=0.1,
                              thresholds=THRESHOLDS)
        assert verdict is Verdict.LIMITED

    def test_poor_detectability_is_undetermined(self):
        verdict, _ = classify(20.0, n_exposed=500, baseline_rate=0.1,
                              thresholds=THRESHOLDS)
        assert verdict is Verdict.UNDETERMINED

    def test_infinite_mdrr_is_undetermined(self):
        verdict, _ = classify(float("inf"), n_exposed=500, baseline_rate=0.1,
                              thresholds=THRESHOLDS)
        assert verdict is Verdict.UNDETERMINED


class TestEvaluateSubgroup:
    def test_zero_events_still_yields_an_upper_bound(self):
        """A subgroup with no events has a worst-case risk, not a clean bill."""
        evidence = evaluate_subgroup(
            subgroup="G:P", event="e", n_exposed=100, n_reference=900,
            events_exposed=0, events_reference=90, thresholds=THRESHOLDS,
        )
        assert evidence.observed_rate == 0.0
        assert evidence.rate_ci_high > 0.0
        assert evidence.rate_ci_low == 0.0

    def test_zero_events_is_not_reported_as_supported_safety(self):
        """The key property: no events in a thin subgroup is not safety."""
        evidence = evaluate_subgroup(
            subgroup="G:P", event="e", n_exposed=25, n_reference=2000,
            events_exposed=0, events_reference=40, thresholds=THRESHOLDS,
        )
        assert evidence.verdict == Verdict.UNDETERMINED.value

    def test_additional_enrollment_is_zero_when_already_powered(self):
        evidence = evaluate_subgroup(
            subgroup="G:P", event="e", n_exposed=5000, n_reference=5000,
            events_exposed=500, events_reference=500, thresholds=THRESHOLDS,
        )
        assert evidence.verdict == Verdict.SUPPORTED.value
        assert evidence.additional_n_for_meaningful_rr == 0.0

    def test_additional_enrollment_is_positive_when_underpowered(self):
        evidence = evaluate_subgroup(
            subgroup="G:P", event="e", n_exposed=60, n_reference=2000,
            events_exposed=6, events_reference=200, thresholds=THRESHOLDS,
        )
        assert evidence.additional_n_for_meaningful_rr > 0


class TestRunSufficiency:
    def test_evaluates_every_pair_in_the_grid(self):
        spec = make_spec(events=["event_a", "event_b"])
        table = make_table()
        table["event_b"] = table["event_a"]
        result = run_sufficiency(table, spec)
        assert len(result) == spec.grid_size() == 2

    def test_missing_phenotype_column_is_an_error(self):
        spec = make_spec()
        table = make_table().drop(columns=["pheno_CYP2D6"])
        with pytest.raises(KeyError):
            run_sufficiency(table, spec)

    def test_indeterminate_calls_are_excluded_not_pooled(self):
        """Unmapped genotypes must not inflate the reference arm.

        Pooling them in would dilute the comparison group and bias the
        contrast toward the null, hiding the effect we are looking for.
        """
        spec = make_spec()
        table = make_table(n_carriers=200, n_others=800,
                           carrier_events=20, other_events=40)
        table.loc[table.index[-300:], "pheno_CYP2D6"] = "Indeterminate"

        result = run_sufficiency(table, spec)
        row = result.iloc[0]
        assert row["n_exposed"] + row["n_reference"] < len(table)

    def test_only_treated_patients_are_counted(self):
        spec = make_spec()
        table = make_table()
        untreated = table.copy()
        untreated["exposed"] = 0
        combined = pd.concat([table, untreated], ignore_index=True)

        treated_only = run_sufficiency(table, spec).iloc[0]
        with_untreated = run_sufficiency(combined, spec).iloc[0]
        assert treated_only["n_exposed"] == with_untreated["n_exposed"]


class TestGate:
    def test_undetermined_pairs_are_not_eligible(self):
        sufficiency = pd.DataFrame({
            "subgroup": ["a", "b", "c"],
            "event": ["e", "e", "e"],
            "verdict": ["supported", "limited", "undetermined"],
        })
        eligible = eligible_for_testing(sufficiency)
        assert ("a", "e") in eligible
        assert ("b", "e") in eligible
        assert ("c", "e") not in eligible


class TestCoverageGap:
    def test_detects_underrepresentation(self):
        spec = make_spec(expected_population={"CYP2D6:Poor metabolizer": 0.30})
        # Only 10% of the trial carries the phenotype, against 30% expected.
        table = make_table(n_carriers=100, n_others=900,
                           carrier_events=10, other_events=90)
        gap = coverage_gap(table, spec)
        row = gap.iloc[0]
        assert row["underrepresented"]
        assert row["observed_share"] == pytest.approx(0.10, abs=1e-9)
        assert row["shortfall_n"] == pytest.approx(200, abs=1)

    def test_no_rows_without_an_expected_distribution(self):
        spec = make_spec(expected_population={})
        assert coverage_gap(make_table(), spec).empty


class TestEnrollmentPlan:
    def test_reports_the_most_demanding_event(self):
        spec = make_spec(events=["event_a", "event_b"])
        table = make_table()
        table["event_b"] = 0
        plan = enrollment_plan(run_sufficiency(table, spec), spec)
        assert len(plan) == 1
        assert plan.iloc[0]["events_total"] == 2
