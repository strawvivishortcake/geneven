"""Tests for the statistical primitives.

These check the maths against known reference values and known behaviour,
not against the implementation's own output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geneven.stats_core import (  # noqa: E402
    FirthLogit,
    benjamini_hochberg,
    clopper_pearson,
    min_detectable_rr,
    power_two_proportions,
    required_n_per_arm,
    rule_of_three_upper,
)


class TestPower:
    def test_power_rises_with_sample_size(self):
        powers = [power_two_proportions(n, n, 0.05, 2.0) for n in (50, 200, 1000)]
        assert powers == sorted(powers)

    def test_power_rises_with_effect_size(self):
        powers = [power_two_proportions(500, 500, 0.05, rr) for rr in (1.2, 2.0, 4.0)]
        assert powers == sorted(powers)

    def test_no_power_without_subjects(self):
        assert power_two_proportions(0, 500, 0.05, 2.0) == 0.0

    def test_round_trip_against_required_n(self):
        """required_n_per_arm inverts power_two_proportions."""
        n = required_n_per_arm(2.0, 0.05, power=0.80)
        assert power_two_proportions(n, n, 0.05, 2.0) == pytest.approx(0.80, abs=1e-3)


class TestMinDetectableRR:
    def test_falls_as_sample_grows(self):
        values = [min_detectable_rr(n, n, 0.05) for n in (50, 200, 1000, 5000)]
        assert values == sorted(values, reverse=True)

    def test_consistent_with_power_at_the_boundary(self):
        """At the returned RR, power equals the target."""
        mdrr = min_detectable_rr(400, 400, 0.05, power=0.80)
        assert power_two_proportions(400, 400, 0.05, mdrr) == pytest.approx(0.80, abs=1e-3)

    def test_infinite_when_nothing_is_detectable(self):
        assert np.isinf(min_detectable_rr(2, 2, 1e-6))

    def test_infinite_without_a_reference_arm(self):
        assert np.isinf(min_detectable_rr(100, 0, 0.05))


class TestClopperPearson:
    def test_zero_events_gives_zero_lower_bound(self):
        low, high = clopper_pearson(0, 100)
        assert low == 0.0
        assert 0 < high < 1

    def test_zero_event_bound_near_rule_of_three(self):
        """The exact bound should sit close to, and above, 3/n."""
        _, high = clopper_pearson(0, 100)
        assert high == pytest.approx(rule_of_three_upper(100), abs=0.01)
        assert high > rule_of_three_upper(100)

    def test_interval_contains_the_point_estimate(self):
        low, high = clopper_pearson(5, 100)
        assert low < 0.05 < high

    def test_all_events_gives_upper_bound_of_one(self):
        low, high = clopper_pearson(10, 10)
        assert high == 1.0
        assert low > 0

    def test_interval_narrows_with_sample_size(self):
        narrow = clopper_pearson(50, 1000)
        wide = clopper_pearson(5, 100)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            clopper_pearson(11, 10)


class TestBenjaminiHochberg:
    def test_matches_the_published_example(self):
        """Benjamini & Hochberg (1995), Table 1."""
        p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
             0.0278, 0.0298, 0.0344, 0.0459, 0.3240]
        q = benjamini_hochberg(p)
        assert q[0] == pytest.approx(0.001, abs=1e-6)
        assert np.all(np.diff(q) >= -1e-12)   # monotone
        assert np.all(q <= 1.0)
        assert np.all(q >= np.array(p) - 1e-12)  # never smaller than raw p

    def test_nan_entries_are_excluded_from_the_denominator(self):
        q = benjamini_hochberg([0.01, np.nan, 0.02])
        assert np.isnan(q[1])
        assert np.isfinite(q[0]) and np.isfinite(q[2])

    def test_all_nan_returns_all_nan(self):
        assert np.all(np.isnan(benjamini_hochberg([np.nan, np.nan])))


class TestFirthLogit:
    def test_tracks_mle_on_well_behaved_data(self):
        rng = np.random.default_rng(0)
        n = 3000
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        y = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 0.7 * x))))

        fit = FirthLogit().fit(X, y)
        # With a large sample the penalty is negligible and the estimate
        # should land near the truth.
        assert fit.beta_[1] == pytest.approx(0.7, abs=0.12)
        assert fit.converged_

    def test_finite_under_complete_separation(self):
        """Where unpenalised ML diverges, Firth must not."""
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        x = np.arange(1, 11, dtype=float)
        X = np.column_stack([np.ones(10), x])

        fit = FirthLogit().fit(X, y)
        assert np.all(np.isfinite(fit.beta_))
        assert abs(fit.beta_[1]) < 10

    def test_plr_detects_a_real_effect(self):
        rng = np.random.default_rng(1)
        n = 800
        g = rng.binomial(1, 0.3, n).astype(float)
        X = np.column_stack([np.ones(n), g])
        y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 1.5 * g)))).astype(float)

        fit = FirthLogit().fit(X, y)
        stat, p = fit.plr_test(1)
        assert stat > 0
        assert p < 0.001

    def test_plr_does_not_fire_on_noise(self):
        rng = np.random.default_rng(2)
        n = 600
        g = rng.binomial(1, 0.3, n).astype(float)
        X = np.column_stack([np.ones(n), g])
        y = rng.binomial(1, 0.1, n).astype(float)   # no relationship to g

        fit = FirthLogit().fit(X, y)
        _, p = fit.plr_test(1)
        assert p > 0.05

    def test_profile_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(3)
        n = 500
        g = rng.binomial(1, 0.4, n).astype(float)
        X = np.column_stack([np.ones(n), g])
        y = rng.binomial(1, 1 / (1 + np.exp(-(-1.5 + 1.0 * g)))).astype(float)

        fit = FirthLogit().fit(X, y)
        low, high = fit.profile_ci(1)
        assert low < fit.beta_[1] < high
        assert low > 0   # a real positive effect excludes zero
