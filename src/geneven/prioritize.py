"""Step 6 - evidence tiers and ranking.

Turns a list of statistical results into a ranked set of findings a team can
act on, and keeps exploratory results visibly separate from established ones.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from .spec import AnalysisSpec


class Tier(str, Enum):
    ESTABLISHED = "established"   # CPIC guideline or PharmGKB level 1A-2A
    REPLICATED = "replicated"     # holds in an independent dataset
    EMERGING = "emerging"         # survives FDR here, no replication
    EXPLORATORY = "exploratory"   # limited subgroup or supporting analysis only


#: Clinical severity weights. Graded from the event itself, not from the
#: statistics, so a weak signal on a fatal event is not buried under a strong
#: signal on a reversible one.
SEVERITY_WEIGHTS = {
    "fatal": 1.00,
    "life_threatening": 0.85,
    "hospitalization": 0.65,
    "permanent_harm": 0.60,
    "reversible": 0.30,
    "unknown": 0.40,
}


def assign_tier(
    row: pd.Series,
    known_pairs: set[tuple[str, str]],
    replicated_pairs: set[tuple[str, str]],
    fdr_threshold: float = 0.05,
) -> str:
    """Tier one finding.

    ``known_pairs`` comes from the pinned CPIC and PharmGKB tables; it is not
    inferred from the data. ``replicated_pairs`` comes from an independent
    dataset analysed under the same specification.
    """
    pair = (row["subgroup"], row["event"])
    if pair in known_pairs:
        return Tier.ESTABLISHED.value
    if not row.get("confirmatory", True):
        return Tier.EXPLORATORY.value
    if pair in replicated_pairs:
        return Tier.REPLICATED.value
    q = row.get("q_value", np.nan)
    if np.isfinite(q) and q <= fdr_threshold:
        return Tier.EMERGING.value
    return Tier.EXPLORATORY.value


def evidence_strength(row: pd.Series) -> float:
    """Strength in [0, 1] from the q-value, effect size and interval width.

    A wide interval is penalised even when the point estimate is large,
    because a finding a sponsor cannot bound is a finding they cannot act on.
    """
    q = row.get("q_value", np.nan)
    q_score = 0.0 if not np.isfinite(q) else float(np.clip(1.0 - q / 0.2, 0.0, 1.0))

    log_or = row.get("interaction_log_or", np.nan)
    effect = 0.0 if not np.isfinite(log_or) else float(np.clip(abs(log_or) / np.log(5), 0, 1))

    lo, hi = row.get("ci_low", np.nan), row.get("ci_high", np.nan)
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
        width = np.log(hi) - np.log(lo)
        precision = float(np.clip(1.0 - width / np.log(100), 0.0, 1.0))
    else:
        precision = 0.0

    return float(0.5 * q_score + 0.3 * effect + 0.2 * precision)


def patient_impact(
    row: pd.Series, spec: AnalysisSpec, population_size: float = 1e6
) -> float:
    """Patients affected, from the phenotype's frequency in the *treated*
    population, not its frequency in the trial.

    A genotype rare in a trial may be common among patients, which is exactly
    the asymmetry Module 01 measures and this ranking must respect.
    """
    share = spec.expected_population.get(row["subgroup"])
    if share is None:
        return 0.0
    excess = row.get("interaction_or", np.nan)
    if not np.isfinite(excess) or excess <= 1:
        return float(share)
    return float(share * min(excess, 10.0) / 10.0 + share)


def prioritize(
    signals: pd.DataFrame,
    spec: AnalysisSpec,
    severity_map: dict[str, str] | None = None,
    known_pairs: set[tuple[str, str]] | None = None,
    replicated_pairs: set[tuple[str, str]] | None = None,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> pd.DataFrame:
    """Rank findings by severity, evidence strength and patient impact.

    The three components are always reported separately as well as combined,
    so a reader can see why a finding placed where it did. A severe reaction
    in a rare phenotype and a moderate one in a common phenotype can rank
    alike; the breakdown is what makes that legible.
    """
    if signals.empty:
        return signals

    severity_map = severity_map or {}
    known_pairs = known_pairs or set()
    replicated_pairs = replicated_pairs or set()
    w_sev, w_str, w_imp = weights

    out = signals.copy()
    out["tier"] = out.apply(
        lambda r: assign_tier(r, known_pairs, replicated_pairs), axis=1
    )
    out["severity_label"] = out["event"].map(lambda e: severity_map.get(e, "unknown"))
    out["severity_score"] = out["severity_label"].map(
        lambda s: SEVERITY_WEIGHTS.get(s, SEVERITY_WEIGHTS["unknown"])
    )
    out["strength_score"] = out.apply(evidence_strength, axis=1)
    out["impact_score"] = out.apply(lambda r: patient_impact(r, spec), axis=1)

    # Normalise impact within this analysis so it is comparable to the others.
    max_impact = out["impact_score"].max()
    if max_impact > 0:
        out["impact_score"] = out["impact_score"] / max_impact

    out["priority_score"] = (
        w_sev * out["severity_score"]
        + w_str * out["strength_score"]
        + w_imp * out["impact_score"]
    )
    # Exploratory findings never outrank confirmatory ones on the headline
    # list, whatever their arithmetic.
    out["priority_score"] = np.where(
        out["tier"] == Tier.EXPLORATORY.value,
        out["priority_score"] * 0.5,
        out["priority_score"],
    )

    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
