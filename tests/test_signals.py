"""Tests for Module 02 and the end-to-end pipeline.

The headline test is :class:`TestRecoversPlantedEffect`, which plants a known
genotype-specific effect in synthetic data and checks the engine finds it,
and :class:`TestGateIsEnforced`, which checks an underpowered subgroup never
reaches testing at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geneven.prioritize import Tier, prioritize  # noqa: E402
from geneven.provenance import ReferenceVersions, RunManifest  # noqa: E402
from geneven.signals import (  # noqa: E402
    calibrate_with_negative_controls,
    choose_model,
    dose_response_trend,
    interaction_test,
    run_signals,
    shrunk_disproportionality,
)
from geneven.spec import AnalysisSpec, Subgroup, SufficiencyThresholds  # noqa: E402
from geneven.sufficiency import run_sufficiency  # noqa: E402
from geneven.pipeline import run_analysis, validate_analysis_table  # noqa: E402


def synthetic_trial(
    n=6000, interaction_or=5.0, carrier_fraction=0.2, base_rate=0.03, seed=7
) -> pd.DataFrame:
    """Trial with a planted exposure x genotype interaction."""
    rng = np.random.default_rng(seed)
    exposed = rng.binomial(1, 0.5, n)
    carrier = rng.binomial(1, carrier_fraction, n)

    odds = base_rate / (1 - base_rate)
    multiplier = np.where(exposed == 1, 1.5, 1.0)
    multiplier = multiplier * np.where((exposed == 1) & (carrier == 1), interaction_or, 1.0)
    probability = (odds * multiplier) / (1 + odds * multiplier)

    return pd.DataFrame({
        "person_id": [f"P{i}" for i in range(n)],
        "exposed": exposed,
        "pheno_CYP2D6": np.where(carrier == 1, "Poor metabolizer", "Normal metabolizer"),
        "event_a": rng.binomial(1, probability),
        "age": rng.normal(60, 12, n),
        "sex": rng.choice(["F", "M"], n),
        "comorbidity_burden": rng.poisson(2, n),
        "pc1": rng.normal(size=n),
        "pc2": rng.normal(size=n),
    })


def make_spec(**overrides) -> AnalysisSpec:
    base = dict(
        study_id="TEST",
        drug="testdrug",
        events=["event_a"],
        subgroups=[Subgroup(gene="CYP2D6", phenotype="Poor metabolizer")],
        covariates=["age", "sex", "comorbidity_burden"],
        ancestry_components=2,
        thresholds=SufficiencyThresholds(min_cell_size=20, min_expected_events=5.0),
    )
    base.update(overrides)
    return AnalysisSpec(**base)


class TestModelChoice:
    def test_rare_cells_select_firth(self):
        frame = pd.DataFrame({
            "exposed": [1] * 50 + [0] * 50,
            "in_subgroup": [1] * 25 + [0] * 25 + [1] * 25 + [0] * 25,
            "event_a": [1] + [0] * 99,
        })
        assert choose_model(frame, "event_a") == "firth"

    def test_plentiful_cells_select_logistic(self):
        rng = np.random.default_rng(0)
        n = 4000
        frame = pd.DataFrame({
            "exposed": rng.binomial(1, 0.5, n),
            "in_subgroup": rng.binomial(1, 0.5, n),
            "event_a": rng.binomial(1, 0.4, n),
        })
        assert choose_model(frame, "event_a") == "logistic"


class TestRecoversPlantedEffect:
    def test_finds_a_real_interaction(self):
        frame = synthetic_trial(interaction_or=5.0)
        frame["in_subgroup"] = (frame["pheno_CYP2D6"] == "Poor metabolizer").astype(int)

        result = interaction_test(
            frame, "event_a", ["age", "comorbidity_burden", "pc1", "pc2"], "CYP2D6:PM"
        )
        assert result.p_value < 0.01
        assert result.interaction_or > 2.0
        assert result.ci_low > 1.0   # interval excludes no-interaction

    def test_does_not_invent_an_interaction(self):
        """With no planted effect the test must not fire."""
        frame = synthetic_trial(interaction_or=1.0, seed=11)
        frame["in_subgroup"] = (frame["pheno_CYP2D6"] == "Poor metabolizer").astype(int)

        result = interaction_test(
            frame, "event_a", ["age", "comorbidity_burden", "pc1", "pc2"], "CYP2D6:PM"
        )
        assert result.p_value > 0.05

    def test_false_positive_rate_is_controlled(self):
        """Across many null datasets, roughly 5% should be significant."""
        hits = 0
        trials = 40
        for seed in range(trials):
            frame = synthetic_trial(n=2500, interaction_or=1.0, seed=100 + seed)
            frame["in_subgroup"] = (
                frame["pheno_CYP2D6"] == "Poor metabolizer"
            ).astype(int)
            result = interaction_test(frame, "event_a", ["age", "pc1"], "CYP2D6:PM")
            if np.isfinite(result.p_value) and result.p_value < 0.05:
                hits += 1
        # Generous bound: the point is that it is not wildly inflated.
        assert hits <= 0.20 * trials


class TestGateIsEnforced:
    def test_undetermined_subgroups_are_never_tested(self):
        """The architectural commitment, verified end to end."""
        spec = make_spec()
        frame = synthetic_trial(n=3000, carrier_fraction=0.004, seed=5)
        sufficiency = run_sufficiency(frame, spec)
        assert (sufficiency["verdict"] == "undetermined").all()

        findings = run_signals(frame, spec, sufficiency)
        assert findings.empty

    def test_supported_subgroups_are_marked_confirmatory(self):
        spec = make_spec()
        frame = synthetic_trial(n=6000, carrier_fraction=0.25)
        sufficiency = run_sufficiency(frame, spec)
        findings = run_signals(frame, spec, sufficiency)
        assert not findings.empty
        assert findings.iloc[0]["confirmatory"]

    def test_fdr_uses_the_full_prespecified_grid(self):
        """Correcting only over tests that ran would understate multiplicity."""
        spec = make_spec(
            events=["event_a", "event_b", "event_c"],
            subgroups=[
                Subgroup(gene="CYP2D6", phenotype="Poor metabolizer"),
                Subgroup(gene="CYP2D6", phenotype="Ultrarapid metabolizer"),
            ],
        )
        frame = synthetic_trial(n=6000, carrier_fraction=0.25)
        frame["event_b"] = 0
        frame["event_c"] = 0
        sufficiency = run_sufficiency(frame, spec)
        findings = run_signals(frame, spec, sufficiency)
        if not findings.empty:
            assert findings.iloc[0]["grid_size"] == 6


class TestSupportingAnalyses:
    def test_shrinkage_damps_a_single_report(self):
        """One event out of a handful must not look like a signal."""
        result = shrunk_disproportionality(a=1, b=2, c=50, d=5000)
        assert not result["signal"]

    def test_shrinkage_passes_a_strong_imbalance(self):
        result = shrunk_disproportionality(a=60, b=40, c=20, d=8000)
        assert result["signal"]
        assert result["shrunk_ratio"] > 1

    def test_empty_stratum_is_handled(self):
        result = shrunk_disproportionality(0, 0, 0, 0)
        assert np.isnan(result["shrunk_ratio"])

    def test_dose_response_detects_a_gradient(self):
        rng = np.random.default_rng(3)
        rows = []
        for band, rate in [("low", 0.02), ("medium", 0.10), ("high", 0.30)]:
            for _ in range(500):
                rows.append((band, rng.binomial(1, rate)))
        frame = pd.DataFrame(rows, columns=["dose_band", "event_a"])
        result = dose_response_trend(frame, "event_a")
        assert result["p_value"] < 0.001

    def test_dose_response_needs_three_bands(self):
        frame = pd.DataFrame({"dose_band": ["low", "high"] * 50,
                              "event_a": [0, 1] * 50})
        assert np.isnan(dose_response_trend(frame, "event_a")["p_value"])

    def test_calibration_flags_inflation(self):
        results = pd.DataFrame({
            "subgroup": ["a"] * 10,
            "event": [f"e{i}" for i in range(10)],
            "p_value": [0.001] * 10,
        })
        controls = {("a", f"e{i}") for i in range(10)}
        assert calibrate_with_negative_controls(results, controls)["inflated"]

    def test_calibration_accepts_a_clean_null(self):
        results = pd.DataFrame({
            "subgroup": ["a"] * 10,
            "event": [f"e{i}" for i in range(10)],
            "p_value": np.linspace(0.1, 0.9, 10),
        })
        controls = {("a", f"e{i}") for i in range(10)}
        assert not calibrate_with_negative_controls(results, controls)["inflated"]


class TestPrioritization:
    def _findings(self) -> pd.DataFrame:
        return pd.DataFrame({
            "subgroup": ["CYP2D6:Poor metabolizer", "CYP2D6:Poor metabolizer"],
            "event": ["fatal_event", "mild_event"],
            "interaction_log_or": [np.log(3.0), np.log(3.0)],
            "interaction_or": [3.0, 3.0],
            "ci_low": [1.5, 1.5],
            "ci_high": [6.0, 6.0],
            "q_value": [0.01, 0.01],
            "confirmatory": [True, True],
            "model": ["firth", "firth"],
        })

    def test_severity_breaks_the_tie(self):
        spec = make_spec()
        ranked = prioritize(
            self._findings(), spec,
            severity_map={"fatal_event": "fatal", "mild_event": "reversible"},
        )
        assert ranked.iloc[0]["event"] == "fatal_event"

    def test_known_pairs_are_tiered_established(self):
        spec = make_spec()
        ranked = prioritize(
            self._findings(), spec,
            known_pairs={("CYP2D6:Poor metabolizer", "fatal_event")},
        )
        row = ranked[ranked["event"] == "fatal_event"].iloc[0]
        assert row["tier"] == Tier.ESTABLISHED.value

    def test_exploratory_findings_are_demoted(self):
        spec = make_spec()
        findings = self._findings()
        findings.loc[0, "confirmatory"] = False
        ranked = prioritize(findings, spec,
                            severity_map={"fatal_event": "fatal",
                                          "mild_event": "reversible"})
        exploratory = ranked[ranked["event"] == "fatal_event"].iloc[0]
        assert exploratory["tier"] == Tier.EXPLORATORY.value


class TestPipeline:
    def test_validation_catches_missing_columns(self):
        spec = make_spec()
        frame = synthetic_trial(n=500).drop(columns=["pc2"])
        problems = validate_analysis_table(frame, spec)
        assert any("pc2" in p for p in problems)

    def test_validation_catches_non_binary_events(self):
        spec = make_spec()
        frame = synthetic_trial(n=500)
        frame["event_a"] = 2
        assert any("0/1" in p for p in validate_analysis_table(frame, spec))

    def test_end_to_end_writes_every_output(self, tmp_path):
        spec = make_spec()
        frame = synthetic_trial(n=6000, carrier_fraction=0.25)
        manifest = RunManifest(
            study_id=spec.study_id, spec_hash=spec.spec_hash(),
            data_snapshot_id="test-001", data_extracted_at="2026-11-07",
            references=ReferenceVersions(), random_seed=1,
        )
        result = run_analysis(frame, spec, manifest, output_dir=tmp_path)

        assert (tmp_path / "evidence_gap_report.md").exists()
        assert (tmp_path / "safety_signal_report.md").exists()
        assert (tmp_path / "run_manifest.json").exists()
        assert not result["sufficiency"].empty

    def test_unpinned_references_raise_a_warning(self, tmp_path):
        manifest = RunManifest(
            study_id="T", spec_hash="abc", data_snapshot_id="unset",
            data_extracted_at="2026-11-07", references=ReferenceVersions(),
            random_seed=1,
        )
        warnings = manifest.warnings()
        assert any("reference versions" in w for w in warnings)
        assert any("snapshot" in w for w in warnings)


class TestSpec:
    def test_hash_changes_when_a_threshold_changes(self):
        a = make_spec()
        b = make_spec(thresholds=SufficiencyThresholds(meaningful_rr=3.0))
        assert a.spec_hash() != b.spec_hash()

    def test_hash_is_stable_across_instances(self):
        assert make_spec().spec_hash() == make_spec().spec_hash()

    def test_grid_size_is_the_cross_product(self):
        spec = make_spec(
            events=["a", "b", "c"],
            subgroups=[
                Subgroup(gene="G1", phenotype="P"),
                Subgroup(gene="G2", phenotype="P"),
            ],
        )
        assert spec.grid_size() == 6
