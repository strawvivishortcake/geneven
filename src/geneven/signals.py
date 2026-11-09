"""Step 5 / Module 02 - the signal analysis engine.

The estimand is the genotype x exposure interaction, not a raw rate
difference. Only pairs cleared by Step 4 are tested; `supported` pairs
are confirmatory, `limited` pairs exploratory, `undetermined` untested.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from .spec import AnalysisSpec
from .stats_core import FirthLogit, benjamini_hochberg


@dataclass
class InteractionResult:
    """One genotype x exposure interaction test."""

    subgroup: str
    event: str
    model: str
    n_total: int
    n_events: int
    min_cell_count: int
    interaction_log_or: float
    interaction_or: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float = float("nan")
    confirmatory: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _min_cell_count(frame: pd.DataFrame, event: str) -> int:
    """Smallest count in the 2x2x2 of exposure x genotype x outcome.

    This is what decides whether standard maximum likelihood is safe. A zero
    cell means separation, and an unpenalised fit would return an infinite
    coefficient with an uninformative standard error.
    """
    counts = (
        frame.groupby(["exposed", "in_subgroup"])[event]
        .agg(["sum", "size"])
        .assign(nonevents=lambda d: d["size"] - d["sum"])
    )
    if counts.empty:
        return 0
    return int(min(counts["sum"].min(), counts["nonevents"].min()))


def choose_model(frame: pd.DataFrame, event: str, rare_threshold: int = 10) -> str:
    """Pick the model the data can support.

    firth    - any cell in the 2x2x2 below ``rare_threshold``; the usual case
               for a rare event inside a small genotype subgroup
    logistic - otherwise
    """
    return "firth" if _min_cell_count(frame, event) < rare_threshold else "logistic"


def _design_matrix(
    frame: pd.DataFrame, covariates: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Intercept, exposure, genotype, their interaction, then covariates.

    The interaction term is always at index 3, which is what the tests below
    report on.
    """
    exposed = frame["exposed"].to_numpy(dtype=float)
    in_subgroup = frame["in_subgroup"].to_numpy(dtype=float)
    columns = [
        np.ones(len(frame)),
        exposed,
        in_subgroup,
        exposed * in_subgroup,
    ]
    names = ["intercept", "exposed", "in_subgroup", "exposed:in_subgroup"]

    for covariate in covariates:
        if covariate not in frame.columns:
            continue
        series = frame[covariate]
        if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
            dummies = pd.get_dummies(series, prefix=covariate, drop_first=True)
            for name in dummies.columns:
                columns.append(dummies[name].to_numpy(dtype=float))
                names.append(name)
        else:
            values = series.to_numpy(dtype=float)
            if np.nanstd(values) == 0:
                continue
            # Centre and scale so the Newton steps behave.
            values = (values - np.nanmean(values)) / np.nanstd(values)
            columns.append(np.nan_to_num(values))
            names.append(covariate)

    return np.column_stack(columns), names


def interaction_test(
    frame: pd.DataFrame,
    event: str,
    covariates: list[str],
    subgroup: str,
    model: str | None = None,
    alpha: float = 0.05,
) -> InteractionResult:
    """Fit the exposure x genotype interaction and test it.

    ``frame`` must carry ``exposed`` (0/1), ``in_subgroup`` (0/1), the event
    column (0/1), and the covariates.

    With the Firth model, inference uses the penalized likelihood ratio test
    and a profile penalized likelihood interval, both of which stay calibrated
    where Wald intervals do not.
    """
    frame = frame.dropna(subset=["exposed", "in_subgroup", event])
    y = frame[event].to_numpy(dtype=float)
    n_events = int(y.sum())
    min_cell = _min_cell_count(frame, event)
    model = model or choose_model(frame, event)

    failure = InteractionResult(
        subgroup=subgroup, event=event, model=model, n_total=len(frame),
        n_events=n_events, min_cell_count=min_cell,
        interaction_log_or=float("nan"), interaction_or=float("nan"),
        ci_low=float("nan"), ci_high=float("nan"), p_value=float("nan"),
    )

    if frame["in_subgroup"].nunique() < 2 or frame["exposed"].nunique() < 2:
        failure.note = "no contrast: genotype or exposure is constant"
        return failure
    if n_events == 0:
        failure.note = "no events observed; nothing to model"
        return failure

    X, names = _design_matrix(frame, covariates)
    idx = names.index("exposed:in_subgroup")

    try:
        if model == "firth":
            fit = FirthLogit().fit(X, y)
            beta = float(fit.beta_[idx])
            _, p_value = fit.plr_test(idx)
            lo, hi = fit.profile_ci(idx, alpha=alpha)
        else:
            import statsmodels.api as sm

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.Logit(y, X).fit(disp=0)
            beta = float(res.params[idx])
            se = float(res.bse[idx])
            z = beta / se if se > 0 else np.nan
            p_value = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
            crit = stats.norm.ppf(1 - alpha / 2)
            lo, hi = beta - crit * se, beta + crit * se
    except Exception as exc:  # pragma: no cover - defensive
        failure.note = f"model failed to fit: {type(exc).__name__}"
        return failure

    return InteractionResult(
        subgroup=subgroup,
        event=event,
        model=model,
        n_total=len(frame),
        n_events=n_events,
        min_cell_count=min_cell,
        interaction_log_or=beta,
        interaction_or=float(np.exp(beta)),
        ci_low=float(np.exp(lo)) if np.isfinite(lo) else float("nan"),
        ci_high=float(np.exp(hi)) if np.isfinite(hi) else float("nan"),
        p_value=float(p_value),
    )


def time_to_event_interaction(
    frame: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    subgroup: str,
) -> InteractionResult:
    """Cox proportional hazards with an exposure x genotype term.

    Used where follow-up varies between patients, which is the usual shape of
    post-marketing data.
    """
    from lifelines import CoxPHFitter

    work = frame.copy()
    work["interaction"] = work["exposed"] * work["in_subgroup"]
    keep = [duration_col, event_col, "exposed", "in_subgroup", "interaction"]
    keep += [c for c in covariates if c in work.columns and work[c].dtype != object]
    work = work[keep].dropna()

    result = InteractionResult(
        subgroup=subgroup, event=event_col, model="cox",
        n_total=len(work), n_events=int(work[event_col].sum()),
        min_cell_count=_min_cell_count(frame, event_col),
        interaction_log_or=float("nan"), interaction_or=float("nan"),
        ci_low=float("nan"), ci_high=float("nan"), p_value=float("nan"),
    )
    try:
        model = CoxPHFitter(penalizer=0.1)
        model.fit(work, duration_col=duration_col, event_col=event_col)
        summary = model.summary.loc["interaction"]
        result.interaction_log_or = float(summary["coef"])
        result.interaction_or = float(summary["exp(coef)"])
        result.ci_low = float(summary["exp(coef) lower 95%"])
        result.ci_high = float(summary["exp(coef) upper 95%"])
        result.p_value = float(summary["p"])
        result.note = "hazard ratio, not odds ratio"
    except Exception as exc:  # pragma: no cover - defensive
        result.note = f"Cox model failed: {type(exc).__name__}"
    return result


def shrunk_disproportionality(
    a: int, b: int, c: int, d: int, prior_strength: float = 0.5
) -> dict:
    """Within-stratum disproportionality with Bayesian shrinkage.

    ``a`` drug+event, ``b`` drug+other event, ``c`` other drug+event,
    ``d`` other drug+other event, all within one genotype stratum.

    The shrinkage pulls estimates from small strata toward the overall rate so
    a handful of reports cannot produce a false alarm. This is
    hypothesis-generating only: spontaneous reports have no denominator, so
    nothing here is confirmatory.
    """
    n = a + b + c + d
    if n == 0:
        return {"observed": 0, "expected": 0.0, "shrunk_ratio": np.nan}

    expected = (a + b) * (a + c) / n
    # Gamma-Poisson shrinkage toward 1.
    shrunk = (a + prior_strength) / (expected + prior_strength)
    # Approximate lower 5% bound of the posterior, the usual screening metric.
    lower = stats.gamma.ppf(0.05, a + prior_strength, scale=1 / (expected + prior_strength))
    return {
        "observed": a,
        "expected": float(expected),
        "shrunk_ratio": float(shrunk),
        "lower_05": float(lower),
        "signal": bool(lower > 1.0),
    }


def dose_response_trend(
    frame: pd.DataFrame, event: str, dose_col: str = "dose_band"
) -> dict:
    """Cochran-Armitage trend test for risk rising with dose.

    A dose-response gradient inside the affected genotype group is strong
    evidence that an association is real rather than confounded.
    """
    work = frame.dropna(subset=[dose_col, event])
    if work[dose_col].nunique() < 3:
        return {"statistic": np.nan, "p_value": np.nan, "note": "fewer than 3 dose bands"}

    scores = pd.Categorical(work[dose_col]).codes.astype(float)
    y = work[event].to_numpy(dtype=float)
    n, s_bar = len(y), scores.mean()
    p_bar = y.mean()
    numerator = float(np.sum((scores - s_bar) * (y - p_bar)))
    denominator = float(np.sqrt(p_bar * (1 - p_bar) * np.sum((scores - s_bar) ** 2)))
    if denominator == 0:
        return {"statistic": np.nan, "p_value": np.nan, "note": "no variance"}
    z = numerator / denominator
    return {"statistic": z, "p_value": float(2 * stats.norm.sf(abs(z))), "note": ""}


def calibrate_with_negative_controls(
    results: pd.DataFrame, negative_control_ids: set[tuple[str, str]]
) -> dict:
    """Check the null distribution using drug-event pairs with no mechanism.

    If significant findings appear among the negative controls, residual
    confounding is inflating the whole analysis and the p-values need
    empirical calibration before anything is reported.
    """
    mask = results.apply(
        lambda r: (r["subgroup"], r["event"]) in negative_control_ids, axis=1
    )
    controls = results[mask].dropna(subset=["p_value"])
    if controls.empty:
        return {"n_controls": 0, "note": "no negative controls evaluated"}

    expected = 0.05 * len(controls)
    observed = int((controls["p_value"] < 0.05).sum())
    return {
        "n_controls": len(controls),
        "observed_significant": observed,
        "expected_significant": float(expected),
        "inflated": bool(observed > max(1, 2 * expected)),
        "median_p": float(controls["p_value"].median()),
    }


def run_signals(
    analysis_table: pd.DataFrame,
    spec: AnalysisSpec,
    sufficiency: pd.DataFrame,
) -> pd.DataFrame:
    """Test every eligible pair, then correct across the whole grid.

    The FDR correction uses the pre-specified grid size from the
    specification, not the number of tests that happened to run. Correcting
    only over the tests that survived Step 4 would understate multiplicity.
    """
    from .sufficiency import eligible_for_testing

    eligible = eligible_for_testing(sufficiency)
    verdict_lookup = {
        (row.subgroup, row.event): row.verdict for row in sufficiency.itertuples()
    }

    covariates = list(spec.covariates) + [
        f"pc{i + 1}" for i in range(spec.ancestry_components)
    ]

    rows = []
    for subgroup in spec.subgroups:
        column = f"pheno_{subgroup.gene}"
        for event in spec.events:
            pair = (subgroup.key(), event)
            if pair not in eligible:
                continue

            frame = analysis_table.copy()
            frame["in_subgroup"] = (frame[column] == subgroup.phenotype).astype(int)
            # Indeterminate genotype calls are dropped, not pooled into the
            # reference group.
            frame = frame[frame[column] != "Indeterminate"]

            result = interaction_test(
                frame, event, covariates, subgroup=subgroup.key(),
                alpha=spec.thresholds.alpha,
            )
            result.confirmatory = verdict_lookup.get(pair) == "supported"
            if not result.confirmatory and not result.note:
                result.note = "exploratory: subgroup evidence was limited"
            rows.append(result.to_dict())

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    # FDR across the full pre-specified grid. Tests that did not run are
    # carried as NaN so the denominator stays honest.
    grid = spec.test_grid()
    ran = {(r["subgroup"], r["event"]): r["p_value"] for _, r in results.iterrows()}
    full = [ran.get(pair, np.nan) for pair in grid]
    q_full = benjamini_hochberg(full)
    q_lookup = dict(zip(grid, q_full))

    results["q_value"] = [
        q_lookup.get((r["subgroup"], r["event"]), np.nan)
        for _, r in results.iterrows()
    ]
    results["grid_size"] = spec.grid_size()
    return results.sort_values("p_value", na_position="last").reset_index(drop=True)
