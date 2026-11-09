"""Geneven - precision drug-safety analysis.

run_signals() tests only pairs that run_sufficiency() cleared, so an
underpowered subgroup cannot return a null that reads as safety.
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
