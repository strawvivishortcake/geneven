"""The analysis specification, frozen by content hash before data access.

No code path accepts an analysis parameter from outside this file. The
hash appears in every report, so changing a threshold changes the hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SufficiencyThresholds:
    """Settings that decide a subgroup's verdict. See :mod:`geneven.sufficiency`."""

    power: float = 0.80
    alpha: float = 0.05
    #: The relative risk the sponsor considers clinically meaningful. A
    #: subgroup counts as `supported` when the data could detect this much.
    meaningful_rr: float = 2.0
    #: How many multiples of `meaningful_rr` still counts as `limited`
    #: rather than `undetermined`.
    limited_multiple: float = 3.0
    #: Subgroups smaller than this are `undetermined` regardless of maths,
    #: and are suppressed from exports for disclosure control.
    min_cell_size: int = 20
    #: Expected events in the exposed arm, under the meaningful relative risk,
    #: below which the normal approximation to the power calculation is not
    #: trustworthy. Subgroups below this are `undetermined`: the arithmetic
    #: would otherwise return a flatteringly small detectable risk from a
    #: handful of patients. The conventional floor is 5.
    min_expected_events: float = 5.0


@dataclass
class Subgroup:
    """One pre-specified genetic subgroup.

    A subgroup is a gene plus a phenotype (or an allele carrier status). The
    adverse events tested against it come from the specification's event list,
    so the test grid is the cross product and its size is known in advance.
    """

    gene: str
    phenotype: str
    label: str | None = None

    def key(self) -> str:
        return f"{self.gene}:{self.phenotype}"

    def display(self) -> str:
        return self.label or f"{self.gene} {self.phenotype}"


@dataclass
class AnalysisSpec:
    """A complete, frozen analysis plan."""

    study_id: str
    drug: str
    events: list[str]
    subgroups: list[Subgroup]
    covariates: list[str] = field(
        default_factory=lambda: ["age", "sex", "comorbidity_burden"]
    )
    #: Number of ancestry principal components entered as covariates.
    ancestry_components: int = 5
    thresholds: SufficiencyThresholds = field(default_factory=SufficiencyThresholds)
    #: Expected phenotype distribution of the treated population, by subgroup
    #: key. Used for the trial-to-patient comparison; sourced from population
    #: allele frequencies, not from the trial.
    expected_population: dict[str, float] = field(default_factory=dict)
    #: Drug-event pairs with no plausible pharmacogenomic mechanism, used to
    #: calibrate the p-value distribution.
    negative_controls: list[dict[str, str]] = field(default_factory=list)
    random_seed: int = 20261107
    notes: str = ""


    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisSpec":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AnalysisSpec":
        thresholds = SufficiencyThresholds(**raw.get("thresholds", {}))
        subgroups = [Subgroup(**s) for s in raw.get("subgroups", [])]
        known = {
            "study_id", "drug", "events", "covariates", "ancestry_components",
            "expected_population", "negative_controls", "random_seed", "notes",
        }
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(subgroups=subgroups, thresholds=thresholds, **kwargs)


    def canonical(self) -> str:
        """Deterministic JSON form, used for hashing."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


    def test_grid(self) -> list[tuple[str, str]]:
        """Every (subgroup key, event) pair that will be tested.

        The FDR correction in :mod:`geneven.signals` is applied across this
        whole grid, and its size is printed in every report so a reader can
        see how many tests a finding survived.
        """
        return [(s.key(), e) for s in self.subgroups for e in self.events]

    def grid_size(self) -> int:
        return len(self.test_grid())

    def subgroup_by_key(self, key: str) -> Subgroup | None:
        return next((s for s in self.subgroups if s.key() == key), None)

    def describe(self) -> str:
        t = self.thresholds
        return (
            f"Study {self.study_id} | drug {self.drug} | spec {self.spec_hash()}\n"
            f"  {len(self.subgroups)} subgroups x {len(self.events)} events "
            f"= {self.grid_size()} tests\n"
            f"  power {t.power:.0%}, alpha {t.alpha}, meaningful RR {t.meaningful_rr}, "
            f"min cell {t.min_cell_size}"
        )
