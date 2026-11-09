"""Step 3 - genomic characterization.

Phenotype assignment, HLA status, ancestry projection and confidence
accounting. Diplotype calling itself is PharmCAT and Cyrius; this reads
their output.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class Confidence(str, Enum):
    """How a genotype call was obtained. Carried through to the report.

    A subgroup built largely on imputed or indeterminate calls is labelled as
    such, because uncertainty in the genotype is uncertainty in the finding.
    """

    DIRECT = "direct"            # sequenced or directly genotyped
    IMPUTED = "imputed"          # imputed above the quality threshold
    INDETERMINATE = "indeterminate"  # no confident call, or no CPIC mapping


class PhenotypeTable:
    """CPIC diplotype-to-phenotype mapping, loaded from a pinned release.

    Production runs must load the real CPIC table and record its version in
    the run manifest. The built-in fallback below covers only the handful of
    diplotypes used by the bundled demo; it is not a substitute for CPIC and
    will refuse to pretend otherwise.
    """

    #: Minimal illustrative mapping for the demo only. Real runs load CPIC.
    _DEMO_ONLY = {
        "CYP2C19": {
            "*1/*1": ("Normal metabolizer", 2.0),
            "*1/*2": ("Intermediate metabolizer", 1.0),
            "*2/*2": ("Poor metabolizer", 0.0),
            "*1/*17": ("Rapid metabolizer", 2.5),
            "*17/*17": ("Ultrarapid metabolizer", 3.0),
        },
        "CYP2D6": {
            "*1/*1": ("Normal metabolizer", 2.0),
            "*1/*4": ("Intermediate metabolizer", 1.0),
            "*4/*4": ("Poor metabolizer", 0.0),
            "*1/*1x2": ("Ultrarapid metabolizer", 3.0),
        },
    }

    def __init__(self, mapping: dict, version: str, is_demo: bool = False):
        self.mapping = mapping
        self.version = version
        self.is_demo = is_demo

    @classmethod
    def from_cpic_file(cls, path: str | Path, version: str) -> "PhenotypeTable":
        """Load a pinned CPIC diplotype-phenotype export.

        Accepts the CPIC tabular export with columns ``gene``, ``diplotype``,
        ``phenotype`` and optionally ``activity_score``.
        """
        frame = pd.read_csv(path, sep=None, engine="python")
        frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
        required = {"gene", "diplotype", "phenotype"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CPIC file missing columns: {sorted(missing)}")

        mapping: dict[str, dict[str, tuple[str, float | None]]] = {}
        for row in frame.itertuples(index=False):
            score = getattr(row, "activity_score", None)
            score = float(score) if score is not None and pd.notna(score) else None
            mapping.setdefault(row.gene, {})[row.diplotype] = (row.phenotype, score)
        return cls(mapping, version=version, is_demo=False)

    @classmethod
    def demo(cls) -> "PhenotypeTable":
        return cls(cls._DEMO_ONLY, version="DEMO-NOT-CPIC", is_demo=True)

    def assign(self, gene: str, diplotype: str) -> tuple[str, float | None, Confidence]:
        """Translate one diplotype.

        A diplotype with no entry is returned as indeterminate. It is never
        defaulted to normal metabolizer: absence of a mapping is not evidence
        of normal function, and silently assuming it would move patients into
        the reference group and mask exactly the effect we are looking for.
        """
        gene_map = self.mapping.get(gene)
        if not gene_map:
            return ("Indeterminate", None, Confidence.INDETERMINATE)
        hit = gene_map.get(diplotype) or gene_map.get(_normalise_diplotype(diplotype))
        if hit is None:
            return ("Indeterminate", None, Confidence.INDETERMINATE)
        return (hit[0], hit[1], Confidence.DIRECT)


def _normalise_diplotype(diplotype: str) -> str:
    """Order the two alleles consistently so ``*2/*1`` matches ``*1/*2``."""
    if "/" not in diplotype:
        return diplotype
    left, right = (part.strip() for part in diplotype.split("/", 1))

    def sort_key(allele: str) -> tuple:
        digits = "".join(ch for ch in allele if ch.isdigit())
        return (int(digits) if digits else 999, allele)

    first, second = sorted([left, right], key=sort_key)
    return f"{first}/{second}"


def load_pharmcat_calls(path: str | Path) -> pd.DataFrame:
    """Read PharmCAT JSON output into one row per person per gene.

    Returns columns: ``person_id``, ``gene``, ``diplotype``, ``call_source``.
    """
    payload = json.loads(Path(path).read_text())
    rows = []
    for sample in payload.get("samples", []):
        person_id = sample.get("sampleId")
        for gene, result in sample.get("geneCalls", {}).items():
            diplotypes = result.get("diplotypes") or []
            if not diplotypes:
                rows.append((person_id, gene, None, "no-call"))
                continue
            rows.append(
                (person_id, gene, diplotypes[0].get("name"), result.get("source", "pharmcat"))
            )
    return pd.DataFrame(rows, columns=["person_id", "gene", "diplotype", "call_source"])


def build_profiles(
    calls: pd.DataFrame,
    table: PhenotypeTable,
    imputed_genes: set[str] | None = None,
) -> pd.DataFrame:
    """Assign a phenotype to every diplotype call.

    Parameters
    ----------
    calls : DataFrame
        Columns ``person_id``, ``gene``, ``diplotype``.
    table : PhenotypeTable
        A pinned CPIC table, or the demo table.
    imputed_genes : set of str, optional
        Genes whose calls came from imputation rather than direct genotyping.
        Their confidence is downgraded even when the mapping succeeds.
    """
    if table.is_demo:
        warnings.warn(
            "Using the demo phenotype table. Production runs must load a pinned "
            "CPIC release via PhenotypeTable.from_cpic_file().",
            stacklevel=2,
        )
    imputed_genes = imputed_genes or set()

    records = []
    for row in calls.itertuples(index=False):
        diplotype = row.diplotype
        if not diplotype or (isinstance(diplotype, float) and np.isnan(diplotype)):
            records.append((row.person_id, row.gene, None, "Indeterminate", None,
                            Confidence.INDETERMINATE.value))
            continue
        phenotype, score, confidence = table.assign(row.gene, diplotype)
        if confidence is Confidence.DIRECT and row.gene in imputed_genes:
            confidence = Confidence.IMPUTED
        records.append((row.person_id, row.gene, diplotype, phenotype, score, confidence.value))

    return pd.DataFrame(
        records,
        columns=["person_id", "gene", "diplotype", "phenotype", "activity_score", "confidence"],
    )


def hla_carrier_status(
    hla_calls: pd.DataFrame, allele: str, min_probability: float = 0.9
) -> pd.DataFrame:
    """Carrier status for one four-digit HLA allele, e.g. ``HLA-B*57:01``.

    ``hla_calls`` has columns ``person_id``, ``allele``, ``probability``.
    Calls below ``min_probability`` are returned as indeterminate rather than
    as non-carriers, for the same reason an unmapped diplotype is.
    """
    subset = hla_calls[hla_calls["allele"] == allele]
    confident = subset[subset["probability"] >= min_probability]
    uncertain = set(subset[subset["probability"] < min_probability]["person_id"])

    everyone = set(hla_calls["person_id"])
    carriers = set(confident["person_id"])

    rows = []
    for person in sorted(everyone):
        if person in carriers:
            rows.append((person, allele, "Carrier", Confidence.DIRECT.value))
        elif person in uncertain:
            rows.append((person, allele, "Indeterminate", Confidence.INDETERMINATE.value))
        else:
            rows.append((person, allele, "Non-carrier", Confidence.DIRECT.value))
    return pd.DataFrame(rows, columns=["person_id", "gene", "phenotype", "confidence"])


@dataclass
class AncestryProjector:
    """Projects study samples onto principal components of a reference panel.

    Ancestry enters every downstream model as continuous components. It is
    never discretised into groups, and self-reported race and ethnicity are
    never substituted for it: they measure different things, and conflating
    them both misses affected patients and mislabels unaffected ones.
    """

    components: int = 5
    _mean: np.ndarray | None = None
    _loadings: np.ndarray | None = None
    _scale: np.ndarray | None = None

    def fit(self, reference_genotypes: np.ndarray) -> "AncestryProjector":
        """Fit on a reference panel (samples x variants), dosages in 0,1,2."""
        X = np.asarray(reference_genotypes, dtype=float)
        self._mean = X.mean(axis=0)
        centred = X - self._mean
        # Scale by expected binomial SD, standard for genotype PCA.
        freq = np.clip(self._mean / 2.0, 1e-6, 1 - 1e-6)
        self._scale = np.sqrt(2 * freq * (1 - freq))
        scaled = centred / self._scale
        _, _, vt = np.linalg.svd(scaled, full_matrices=False)
        self._loadings = vt[: self.components].T
        return self

    def transform(self, genotypes: np.ndarray) -> np.ndarray:
        if self._loadings is None:
            raise RuntimeError("fit() must be called before transform()")
        X = np.asarray(genotypes, dtype=float)
        scaled = (X - self._mean) / self._scale
        return scaled @ self._loadings

    def column_names(self) -> list[str]:
        return [f"pc{i + 1}" for i in range(self.components)]


def assemble_profile_table(
    phenotypes: pd.DataFrame,
    ancestry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pivot per-gene phenotypes into one row per person, and attach ancestry.

    Produces columns ``pheno_<GENE>`` and ``conf_<GENE>`` plus ancestry
    components, keyed by ``person_id``.
    """
    wide = phenotypes.pivot_table(
        index="person_id", columns="gene", values="phenotype", aggfunc="first"
    )
    wide.columns = [f"pheno_{c}" for c in wide.columns]

    conf = phenotypes.pivot_table(
        index="person_id", columns="gene", values="confidence", aggfunc="first"
    )
    conf.columns = [f"conf_{c}" for c in conf.columns]

    out = wide.join(conf).reset_index()
    if ancestry is not None:
        out = out.merge(ancestry, on="person_id", how="left")
    return out
