"""Step 4 / Module 01 - the evidence sufficiency engine.

This runs before any signal testing and decides which subgroups are eligible
for it. Its output is a verdict per subgroup with the numbers behind it.

The question it answers is not "is this drug safe in this subgroup?" but
"could this dataset have told us if it weren't?". Those are different
questions, and conflating them is the failure mode Geneven exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

from .spec import AnalysisSpec, SufficiencyThresholds
from .stats_core import (
    clopper_pearson,
    min_detectable_rr,
    power_two_proportions,
    required_n_per_arm,
)


class Verdict(str, Enum):
    SUPPORTED = "supported"
    LIMITED = "limited"
    UNDETERMINED = "undetermined"


@dataclass
class SubgroupEvidence:
    """Everything the engine concluded about one subgroup-event pair."""

    subgroup: str
    event: str
    n_exposed: int
    n_reference: int
    events_exposed: int
    events_reference: int
    baseline_rate: float
    observed_rate: float
    rate_ci_low: float
    rate_ci_high: float
    min_detectable_rr: float
    power_at_meaningful_rr: float
    additional_n_for_meaningful_rr: float
    verdict: str
    verdict_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def classify(
    mdrr: float,
    n_exposed: int,
    baseline_rate: float,
    thresholds: SufficiencyThresholds,
) -> tuple[Verdict, str]:
    """Apply the verdict rules.

    supported     - the data could detect a clinically meaningful risk
    limited       - only a much larger effect would have been visible
    undetermined  - the data cannot support a conclusion in either direction

    Two guards are checked before the arithmetic, and both override it.

    The minimum cell size comes first: a handful of patients can produce a
    flattering detectable-risk number when the event is common.

    The expected-event floor comes second, and matters more. The power
    calculation behind ``mdrr`` is a normal approximation, which overstates
    power when the exposed arm would produce only a few events even under the
    alternative. Thirty patients with three expected events can return a
    detectable risk near 4, implying the subgroup was almost adequate when it
    was nowhere close. Below the floor the number is not trustworthy and the
    verdict is `undetermined` regardless of what it says.
    """
    if n_exposed < thresholds.min_cell_size:
        return (
            Verdict.UNDETERMINED,
            f"{n_exposed} exposed is below the minimum cell size of "
            f"{thresholds.min_cell_size}",
        )

    expected_events = n_exposed * baseline_rate * thresholds.meaningful_rr
    if expected_events < thresholds.min_expected_events:
        return (
            Verdict.UNDETERMINED,
            f"only {expected_events:.1f} events expected in {n_exposed} exposed "
            f"even at RR {thresholds.meaningful_rr:.2f}, below the floor of "
            f"{thresholds.min_expected_events:g} where the power approximation "
            f"is reliable",
        )

    if not np.isfinite(mdrr):
        return (
            Verdict.UNDETERMINED,
            "no finite relative risk is detectable at the specified power",
        )

    ceiling = thresholds.meaningful_rr * thresholds.limited_multiple
    if mdrr <= thresholds.meaningful_rr:
        return (
            Verdict.SUPPORTED,
            f"detectable down to RR {mdrr:.2f}, at or below the meaningful "
            f"threshold of {thresholds.meaningful_rr:.2f}",
        )
    if mdrr <= ceiling:
        return (
            Verdict.LIMITED,
            f"detectable only at RR {mdrr:.2f}, above the meaningful threshold "
            f"of {thresholds.meaningful_rr:.2f} but within {thresholds.limited_multiple:g}x",
        )
    return (
        Verdict.UNDETERMINED,
        f"detectable only at RR {mdrr:.2f}, beyond {thresholds.limited_multiple:g}x "
        f"the meaningful threshold of {thresholds.meaningful_rr:.2f}",
    )


def evaluate_subgroup(
    *,
    subgroup: str,
    event: str,
    n_exposed: int,
    n_reference: int,
    events_exposed: int,
    events_reference: int,
    thresholds: SufficiencyThresholds,
) -> SubgroupEvidence:
    """Run the full sufficiency calculation for one subgroup-event pair."""
    t = thresholds
    baseline = events_reference / n_reference if n_reference > 0 else 0.0
    observed = events_exposed / n_exposed if n_exposed > 0 else 0.0

    # Exact interval, including the zero-event case where the upper bound is
    # the whole answer.
    ci_low, ci_high = clopper_pearson(events_exposed, n_exposed, alpha=t.alpha)

    mdrr = min_detectable_rr(
        n_exposed, n_reference, baseline, alpha=t.alpha, power=t.power
    )
    power_now = power_two_proportions(
        n_exposed, n_reference, baseline, t.meaningful_rr, alpha=t.alpha
    )

    # How many more exposed patients of this genotype would close the gap.
    needed = required_n_per_arm(
        t.meaningful_rr,
        baseline,
        alpha=t.alpha,
        power=t.power,
        allocation_ratio=(n_reference / n_exposed) if n_exposed > 0 else 1.0,
    )
    additional = max(0.0, needed - n_exposed) if np.isfinite(needed) else float("inf")

    verdict, reason = classify(mdrr, n_exposed, baseline, t)

    return SubgroupEvidence(
        subgroup=subgroup,
        event=event,
        n_exposed=int(n_exposed),
        n_reference=int(n_reference),
        events_exposed=int(events_exposed),
        events_reference=int(events_reference),
        baseline_rate=float(baseline),
        observed_rate=float(observed),
        rate_ci_low=float(ci_low),
        rate_ci_high=float(ci_high),
        min_detectable_rr=float(mdrr),
        power_at_meaningful_rr=float(power_now),
        additional_n_for_meaningful_rr=float(additional),
        verdict=verdict.value,
        verdict_reason=reason,
    )


def run_sufficiency(
    analysis_table: pd.DataFrame,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    """Evaluate every subgroup-event pair in the pre-specified grid.

    ``analysis_table`` is one row per person with at least:
      - ``person_id``
      - ``exposed`` (0/1) - received the drug under study
      - ``pheno_<GENE>`` - phenotype per gene, from :mod:`geneven.genomics`
      - one 0/1 column per event in ``spec.events``

    For each subgroup, the exposed arm is "exposed and in this genotype
    group"; the reference arm is "exposed and not in this genotype group".
    The contrast is therefore genotype within treated patients, which is the
    contrast the interaction test in Step 5 will formalise.
    """
    rows = []
    for subgroup in spec.subgroups:
        column = f"pheno_{subgroup.gene}"
        if column not in analysis_table.columns:
            raise KeyError(
                f"analysis table has no column {column!r} for subgroup "
                f"{subgroup.key()!r}"
            )
        treated = analysis_table[analysis_table["exposed"] == 1]
        in_group = treated[column] == subgroup.phenotype
        # Indeterminate calls are excluded from the reference arm rather than
        # counted as unaffected; see genomics.PhenotypeTable.assign.
        determinate = treated[column] != "Indeterminate"
        carriers = treated[in_group]
        reference = treated[(~in_group) & determinate]

        for event in spec.events:
            if event not in analysis_table.columns:
                raise KeyError(f"analysis table has no event column {event!r}")
            rows.append(
                evaluate_subgroup(
                    subgroup=subgroup.key(),
                    event=event,
                    n_exposed=len(carriers),
                    n_reference=len(reference),
                    events_exposed=int(carriers[event].sum()),
                    events_reference=int(reference[event].sum()),
                    thresholds=spec.thresholds,
                ).to_dict()
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trial-to-patient comparison
# ---------------------------------------------------------------------------

def coverage_gap(
    analysis_table: pd.DataFrame,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    """Compare the trial's genotype mix with the expected treated population.

    ``spec.expected_population`` maps subgroup keys to the frequency that
    phenotype is expected to have among patients who will receive the drug.
    Those frequencies come from population allele frequencies and the
    indication's epidemiology, never from the trial itself, which is the
    thing being audited.

    Returns, per subgroup: observed and expected share, the ratio, and the
    number of additional patients of that genotype the trial is short by to
    match the expected mix at its current size.
    """
    treated = analysis_table[analysis_table["exposed"] == 1]
    total = len(treated)

    rows = []
    for subgroup in spec.subgroups:
        key = subgroup.key()
        expected_share = spec.expected_population.get(key)
        if expected_share is None:
            continue
        column = f"pheno_{subgroup.gene}"
        observed_n = int((treated[column] == subgroup.phenotype).sum())
        observed_share = observed_n / total if total else 0.0
        expected_n = expected_share * total
        rows.append(
            {
                "subgroup": key,
                "display": subgroup.display(),
                "observed_n": observed_n,
                "observed_share": observed_share,
                "expected_share": expected_share,
                "share_ratio": (observed_share / expected_share) if expected_share else np.nan,
                "shortfall_n": max(0.0, expected_n - observed_n),
                "underrepresented": observed_share < expected_share,
            }
        )
    return pd.DataFrame(rows)


def enrollment_plan(
    sufficiency: pd.DataFrame,
    spec: AnalysisSpec,
) -> pd.DataFrame:
    """Per subgroup, the enrollment that would lift it to `supported`.

    Aggregates across events by taking the most demanding event, because a
    trial is only adequate for a subgroup when it is adequate for every
    outcome the specification says matters.
    """
    if sufficiency.empty:
        return pd.DataFrame()

    finite = sufficiency.replace([np.inf], np.nan)
    grouped = finite.groupby("subgroup").agg(
        n_exposed=("n_exposed", "first"),
        worst_mdrr=("min_detectable_rr", "max"),
        additional_needed=("additional_n_for_meaningful_rr", "max"),
        events_unsupported=("verdict", lambda s: int((s != "supported").sum())),
        events_total=("verdict", "size"),
    )
    grouped = grouped.reset_index()
    grouped["unreachable"] = grouped["additional_needed"].isna()
    grouped["display"] = grouped["subgroup"].map(
        lambda k: (spec.subgroup_by_key(k).display() if spec.subgroup_by_key(k) else k)
    )
    return grouped.sort_values("additional_needed", ascending=False, na_position="first")


def eligible_for_testing(sufficiency: pd.DataFrame) -> set[tuple[str, str]]:
    """Pairs that may proceed to confirmatory signal testing.

    `supported` pairs are confirmatory. `limited` pairs are analysed but
    flagged exploratory by the signal engine. `undetermined` pairs are not
    tested at all: there is nothing to find and a null result would be
    actively misleading.
    """
    allowed = sufficiency[sufficiency["verdict"].isin(["supported", "limited"])]
    return set(zip(allowed["subgroup"], allowed["event"]))
