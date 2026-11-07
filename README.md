# Geneven

Precision drug-safety analysis. For every genetically defined subgroup of
patients, two questions, in this order:

1. **Do we have enough evidence to know?** (Module 01, `geneven.sufficiency`)
2. **Does this drug harm this subgroup?** (Module 02, `geneven.signals`)

The order is enforced in code, not by convention. `run_signals` only tests
pairs that `run_sufficiency` cleared, so the pipeline cannot return a null
result from an underpowered subgroup and let it be read as safety.

This repository implements Steps 4 to 7 of the technical implementation
document. Steps 1 to 3 (acquisition, OMOP harmonization, diplotype calling)
run upstream; `geneven.genomics` covers the phenotype assignment and profile
assembly that hand this stage its input.

## Install

```bash
pip install -e .
```

Requires Python 3.11+. Core dependencies are numpy, scipy, pandas, PyYAML;
`statsmodels` and `lifelines` are used for the unpenalised logistic and Cox
models respectively.

## Run the demo

```bash
python examples/run_demo.py
```

The demo builds a synthetic trial with two planted effects and shows both
halves of the system working:

- **CYP2D6 poor metabolizers** are adequately enrolled and carry a real
  neutropenia effect. The engine finds it (interaction OR ~4.2, q ~0.001).
- **CYP2C19 ultrarapid metabolizers** are deliberately under-enrolled and
  carry a real hepatotoxicity effect. The engine cannot find it, and reports
  the subgroup as `undetermined` rather than clean.

The second case is the point. A conventional pipeline would report no signal
there, and a reader would take that as reassurance.

## Run on real data

```bash
python -m geneven.pipeline \
  --spec config/spec_example.yaml \
  --data analysis_table.parquet \
  --out output/ \
  --snapshot-id "trial-2026-11-07" \
  --extracted-at "2026-11-07" \
  --cpic-version "2026-09" \
  --pharmvar-version "6.2.1" \
  --pharmgkb-version "2026-10"
```

### The analysis table

One row per person, with:

| Column | Meaning |
| --- | --- |
| `person_id` | Pseudonymous identifier |
| `exposed` | 0/1, received the drug under study |
| `pheno_<GENE>` | Phenotype per gene, from `geneven.genomics` |
| `pc1` … `pcN` | Ancestry principal components, continuous |
| one per event | 0/1 outcome columns named in the spec |
| covariates | age, sex, comorbidity burden, concomitant medications |

`validate_analysis_table` checks all of this before anything runs and reports
every problem at once.

## The analysis specification

Every threshold, subgroup, covariate and model choice lives in one YAML file,
written and frozen before the data is touched. Nothing in the pipeline takes
an analysis parameter from anywhere else, so there is no path by which a
decision can be made after seeing a result.

Freezing computes a SHA-256 over the canonical form. That hash appears in
every report. Change a threshold, and the hash changes with it.

See `config/spec_example.yaml`.

## Verdict rules

| Verdict | Condition |
| --- | --- |
| `supported` | Minimum detectable relative risk at or below the meaningful threshold, cell size met, enough expected events for the approximation to hold |
| `limited` | Detectable only at a larger effect, within `limited_multiple` of the threshold |
| `undetermined` | Below minimum cell size, below the expected-event floor, or beyond that multiple |

Three guards sit in front of the arithmetic, and each overrides it:

- **Minimum cell size.** A handful of patients can produce a flattering
  detectable-risk number when the event is common.
- **Expected-event floor.** The power calculation is a normal approximation,
  which overstates power when the exposed arm would see only a few events
  even under the alternative. Thirty patients with three expected events can
  return a detectable risk near 4, implying the subgroup was nearly adequate
  when it was nowhere close.
- **Finiteness.** If no finite relative risk reaches the target power, the
  verdict is `undetermined`, not a very large number.

## Statistical choices, and why

**The primary test is an interaction test.** Not "is the event more common in
this genotype group", which confounds genotype's main effect with the drug's.
The claim Geneven makes is that the *drug-event association differs* by
genotype, which is the `exposed × in_subgroup` term.

**Firth penalization by default for rare cells.** Standard maximum likelihood
breaks down exactly where this work lives: a rare adverse event inside a small
genotype subgroup, where a zero cell sends the coefficient to infinity.
Firth's penalty removes the first-order bias and always yields finite
estimates. Inference uses the penalized likelihood ratio test and profile
intervals, which stay calibrated where Wald does not.

**Exact intervals everywhere, including zero events.** A subgroup with no
observed events still has a worst-case risk consistent with its data. The
Clopper-Pearson upper bound is that number, and it is the honest answer to
"is this safe?".

**FDR across the full pre-specified grid.** The denominator is
`subgroups × events` from the specification, not the number of tests that
happened to run. Correcting only over tests that cleared the gate would
understate multiplicity.

**Indeterminate genotype calls are dropped, never pooled.** An unmapped
diplotype is not evidence of normal function. Pooling it into the reference
arm would dilute the comparison group and bias the contrast toward the null,
hiding the effect being looked for.

**Ancestry is continuous, and is not race.** Ancestry enters every model as
principal components. Self-reported race and ethnicity are carried as
separate fields and never substituted for it.

## Layout

```
src/geneven/
  spec.py          Analysis specification: load, validate, freeze, hash
  provenance.py    Run manifests and reference-version pinning
  stats_core.py    Power, MDRR, exact intervals, BH-FDR, Firth regression
  genomics.py      Step 3: phenotype assignment, HLA, ancestry projection
  sufficiency.py   Step 4 / Module 01: the evidence sufficiency engine
  signals.py       Step 5 / Module 02: interaction tests and support analyses
  prioritize.py    Step 6: evidence tiers and ranking
  report.py        Step 7: evidence-gap and safety-signal reports
  pipeline.py      Orchestration and CLI
tests/             66 tests covering the maths and the gate
examples/          End-to-end demo on synthetic data
config/            Example analysis specification
```

## Reproducibility

Every run writes a manifest recording five things: the data snapshot, the
pinned reference versions, the code commit, the specification hash and the
random seed. `RunManifest.warnings()` reports anything that would make a run
non-reproducible, and those warnings are printed in the report itself rather
than hidden in a log.

## Tests

```bash
python -m pytest tests/ -q
```

The suite checks the maths against published reference values (the
Benjamini-Hochberg worked example, Clopper-Pearson against the rule of three,
Firth against unpenalised ML on clean data and under complete separation),
and checks the behavioural properties that matter: planted effects are
recovered, null data does not produce findings, false-positive rates are
controlled, and undetermined subgroups never reach the signal engine.

## What this repository does not do

- **Call diplotypes from sequence or array data.** That is PharmCAT's job
  (and Cyrius for CYP2D6 structural variants). `load_pharmcat_calls` reads
  their output.
- **Ship CPIC tables.** The bundled phenotype mapping covers only the
  diplotypes the demo uses and warns loudly when used. Production runs load a
  pinned CPIC release through `PhenotypeTable.from_cpic_file`.
- **Perform the OMOP ETL.** That is OHDSI tooling (WhiteRabbit,
  Rabbit-in-a-Hat, the Data Quality Dashboard), run upstream.
- **Set thresholds for you.** Power, the meaningful relative risk, the
  minimum cell size and the expected-event floor are per-engagement decisions
  recorded in the specification. The defaults are conventional starting
  points, not recommendations.
