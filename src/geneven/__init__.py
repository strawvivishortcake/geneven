"""Geneven - precision drug-safety analysis.

Two questions, in this order, for every genetically defined subgroup:

  1. Do we have enough evidence to know?   (Module 01, :mod:`geneven.sufficiency`)
  2. Does this drug harm this subgroup?    (Module 02, :mod:`geneven.signals`)

The order is enforced in code, not by convention: :func:`run_signals` only
tests pairs that :func:`run_sufficiency` cleared, so the pipeline cannot
produce a null result in an underpowered subgroup and let it read as safety.
"""

from .spec import AnalysisSpec, Subgroup, SufficiencyThresholds
from .provenance import ReferenceVersions, RunManifest
from .sufficiency import Verdict, run_sufficiency, coverage_gap, enrollment_plan
from .signals import run_signals, interaction_test
from .prioritize import Tier, prioritize
from .pipeline import run_analysis

__version__ = "0.1.0"

__all__ = [
    "AnalysisSpec",
    "Subgroup",
    "SufficiencyThresholds",
    "ReferenceVersions",
    "RunManifest",
    "Verdict",
    "Tier",
    "run_sufficiency",
    "coverage_gap",
    "enrollment_plan",
    "run_signals",
    "interaction_test",
    "prioritize",
    "run_analysis",
]
