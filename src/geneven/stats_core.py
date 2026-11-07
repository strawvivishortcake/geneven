"""Statistical primitives shared by the sufficiency and signal engines.

Everything here is deliberately plain: closed-form or root-found quantities
built on numpy/scipy, with no hidden state. A sponsor biostatistician must be
able to read each function and reproduce it by hand.

References for the formulae used:
  - Two-proportion power: normal approximation, unpooled variance
    (Fleiss, Levin & Paik, Statistical Methods for Rates and Proportions, 3rd ed.)
  - Clopper-Pearson interval: exact binomial, Beta quantile form
  - Firth penalized likelihood: Firth (1993) Biometrika 80:27-38;
    Heinze & Schemper (2002) Stat Med 21:2409-2419
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

__all__ = [
    "power_two_proportions",
    "min_detectable_rr",
    "required_n_per_arm",
    "clopper_pearson",
    "rule_of_three_upper",
    "benjamini_hochberg",
    "FirthLogit",
]


# --------------------------------------------------------------------------
# Two-proportion power, inverted three ways
# --------------------------------------------------------------------------

def power_two_proportions(
    n_exposed: float,
    n_unexposed: float,
    p0: float,
    rr: float,
    alpha: float = 0.05,
) -> float:
    """Power to detect a relative risk ``rr`` against baseline risk ``p0``.

    Two-sided test of two independent proportions, normal approximation with
    unpooled variance in the alternative and pooled variance under the null.

    Parameters
    ----------
    n_exposed, n_unexposed : float
        Subjects in each group. Fractional values are allowed so the function
        can be inverted smoothly by a root finder.
    p0 : float
        Event probability in the unexposed (reference) group.
    rr : float
        Relative risk under the alternative. ``p1 = rr * p0``.
    alpha : float
        Two-sided significance level.

    Returns
    -------
    float
        Power in [0, 1].
    """
    if min(n_exposed, n_unexposed) <= 0:
        return 0.0
    p1 = min(rr * p0, 1.0 - 1e-12)
    if p1 <= 0 or p0 <= 0:
        return 0.0

    # Pooled proportion under the null, weighted by group size.
    p_bar = (p1 * n_exposed + p0 * n_unexposed) / (n_exposed + n_unexposed)
    se_null = np.sqrt(p_bar * (1 - p_bar) * (1 / n_exposed + 1 / n_unexposed))
    se_alt = np.sqrt(p1 * (1 - p1) / n_exposed + p0 * (1 - p0) / n_unexposed)
    if se_alt <= 0 or se_null <= 0:
        return 0.0

    z_crit = stats.norm.ppf(1 - alpha / 2)
    delta = abs(p1 - p0)
    # Power in the direction of the alternative; the opposite tail is
    # negligible for the effect sizes this engine reports and is dropped.
    z = (delta - z_crit * se_null) / se_alt
    return float(stats.norm.cdf(z))


def min_detectable_rr(
    n_exposed: float,
    n_unexposed: float,
    p0: float,
    alpha: float = 0.05,
    power: float = 0.80,
    rr_max: float = 1e4,
) -> float:
    """Smallest relative risk detectable at ``power``, given the data in hand.

    This is the headline sufficiency number. A subgroup whose minimum
    detectable relative risk is 12 has not been shown to be safe; it has been
    shown to be incapable of revealing anything short of a twelvefold increase.

    Returns ``inf`` when no finite relative risk reaches the target power
    (which happens when the subgroup is very small or the event very rare).
    """
    if min(n_exposed, n_unexposed) <= 0 or p0 <= 0:
        return float("inf")

    def shortfall(rr: float) -> float:
        return power_two_proportions(n_exposed, n_unexposed, p0, rr, alpha) - power

    # Power is monotone increasing in rr over rr >= 1 until p1 saturates at 1.
    rr_ceiling = min(rr_max, (1.0 - 1e-9) / p0)
    if rr_ceiling <= 1.0 or shortfall(rr_ceiling) < 0:
        return float("inf")
    if shortfall(1.0 + 1e-9) >= 0:
        return 1.0

    return float(optimize.brentq(shortfall, 1.0 + 1e-9, rr_ceiling, xtol=1e-6))


def required_n_per_arm(
    target_rr: float,
    p0: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    n_max: float = 1e9,
) -> float:
    """Exposed-group size needed to detect ``target_rr`` at ``power``.

    ``allocation_ratio`` is unexposed per exposed. Returns ``inf`` if the
    target is unreachable below ``n_max``.
    """
    if target_rr <= 1.0 or p0 <= 0 or p0 >= 1:
        return float("inf")

    def shortfall(n_exposed: float) -> float:
        return (
            power_two_proportions(
                n_exposed, n_exposed * allocation_ratio, p0, target_rr, alpha
            )
            - power
        )

    if shortfall(n_max) < 0:
        return float("inf")
    lo = 2.0
    if shortfall(lo) >= 0:
        return lo
    return float(optimize.brentq(shortfall, lo, n_max, xtol=0.5))


# --------------------------------------------------------------------------
# Exact interval estimation, including the zero-event case
# --------------------------------------------------------------------------

def clopper_pearson(
    events: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval for a rate.

    Used for every subgroup rate Geneven reports, including subgroups with
    zero observed events: the upper bound is the worst-case risk still
    consistent with the data, and it is the honest answer to "is this safe?".
    """
    if n <= 0:
        return (0.0, 1.0)
    if events < 0 or events > n:
        raise ValueError("events must lie in [0, n]")

    lower = 0.0 if events == 0 else float(stats.beta.ppf(alpha / 2, events, n - events + 1))
    upper = 1.0 if events == n else float(
        stats.beta.ppf(1 - alpha / 2, events + 1, n - events)
    )
    return (lower, upper)


def rule_of_three_upper(n: int) -> float:
    """Approximate 95% upper bound on a rate after zero events in ``n``.

    Kept alongside :func:`clopper_pearson` because reviewers recognise it and
    it makes the exact bound easy to sanity-check: 3/n.
    """
    return 3.0 / n if n > 0 else 1.0


# --------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------

def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg q-values over the full pre-specified test grid.

    Implemented here rather than imported so the grid definition and the
    correction stay in one auditable place.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if not finite.any():
        return q

    pf = p[finite]
    m = pf.size
    order = np.argsort(pf)
    ranked = pf[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p-value down.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    out = np.empty(m)
    out[order] = adjusted
    q[finite] = out
    return q


# --------------------------------------------------------------------------
# Firth penalized logistic regression
# --------------------------------------------------------------------------

class FirthLogit:
    """Firth-penalized logistic regression.

    Standard maximum likelihood breaks down exactly where Geneven works most
    often: a rare adverse event inside a small genotype subgroup, where a zero
    cell sends the coefficient to infinity (separation). Firth's penalty --
    adding ``0.5 * log det(X'WX)`` to the log-likelihood -- removes the
    first-order bias and always yields finite estimates.

    Inference on a single coefficient uses the penalized likelihood ratio
    test, which is better calibrated than Wald in these conditions.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol
        self.beta_: np.ndarray | None = None
        self.loglik_: float | None = None
        self.converged_: bool = False

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _probs(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        eta = np.clip(X @ beta, -35, 35)
        return 1.0 / (1.0 + np.exp(-eta))

    @classmethod
    def _penalized_loglik(
        cls, beta: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        p = cls._probs(X, beta)
        eps = 1e-12
        ll = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return -np.inf
        return ll + 0.5 * logdet

    def _fit_free(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Newton-Raphson on the Firth-modified score equations."""
        n, k = X.shape
        beta = np.zeros(k)
        for _ in range(self.max_iter):
            p = self._probs(X, beta)
            w = p * (1 - p)
            info = X.T @ (X * w[:, None])
            try:
                info_inv = np.linalg.inv(info)
            except np.linalg.LinAlgError:
                info_inv = np.linalg.pinv(info)
            # Leverages of the weighted design.
            sw = np.sqrt(w)
            Xw = X * sw[:, None]
            h = np.einsum("ij,jk,ik->i", Xw, info_inv, Xw)
            # Firth-modified score.
            score = X.T @ (y - p + h * (0.5 - p))
            step = info_inv @ score
            # Damp large steps for stability under separation.
            norm = np.linalg.norm(step)
            if norm > 5.0:
                step *= 5.0 / norm
            beta = beta + step
            if norm < self.tol:
                self.converged_ = True
                break
        return beta

    def _fit_constrained(
        self, X: np.ndarray, y: np.ndarray, index: int, value: float
    ) -> float:
        """Max penalized log-likelihood with ``beta[index]`` fixed.

        The penalty still uses the full design matrix, which is what makes
        this a profile penalized likelihood rather than a submodel fit.
        """
        k = X.shape[1]
        free = [j for j in range(k) if j != index]

        def negative(theta: np.ndarray) -> float:
            beta = np.empty(k)
            beta[index] = value
            beta[free] = theta
            val = self._penalized_loglik(beta, X, y)
            return -val if np.isfinite(val) else 1e12

        start = np.zeros(len(free))
        if self.beta_ is not None:
            start = self.beta_[free]
        result = optimize.minimize(negative, start, method="BFGS")
        return -float(result.fun)

    # -- public API --------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FirthLogit":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.beta_ = self._fit_free(X, y)
        self.loglik_ = self._penalized_loglik(self.beta_, X, y)
        self._X, self._y = X, y
        return self

    def plr_test(self, index: int) -> tuple[float, float]:
        """Penalized likelihood ratio test of ``beta[index] == 0``.

        Returns ``(statistic, p_value)`` on one degree of freedom.
        """
        if self.beta_ is None:
            raise RuntimeError("fit() must be called first")
        ll_null = self._fit_constrained(self._X, self._y, index, 0.0)
        stat = 2.0 * (self.loglik_ - ll_null)
        stat = max(stat, 0.0)
        return float(stat), float(stats.chi2.sf(stat, df=1))

    def profile_ci(
        self, index: int, alpha: float = 0.05, span: float = 10.0
    ) -> tuple[float, float]:
        """Profile penalized likelihood interval for one coefficient."""
        if self.beta_ is None:
            raise RuntimeError("fit() must be called first")
        cut = stats.chi2.ppf(1 - alpha, df=1) / 2.0
        centre = self.beta_[index]
        target = self.loglik_ - cut

        def gap(value: float) -> float:
            return self._fit_constrained(self._X, self._y, index, value) - target

        bounds = []
        for direction in (-1, 1):
            edge = centre + direction * span
            if gap(edge) > 0:
                bounds.append(float("-inf") if direction < 0 else float("inf"))
                continue
            lo, hi = (edge, centre) if direction < 0 else (centre, edge)
            try:
                bounds.append(float(optimize.brentq(gap, lo, hi, xtol=1e-4)))
            except ValueError:
                bounds.append(float("nan"))
        return (bounds[0], bounds[1])

    def wald_se(self) -> np.ndarray:
        """Standard errors from the inverse information at the estimate."""
        p = self._probs(self._X, self.beta_)
        w = p * (1 - p)
        info = self._X.T @ (self._X * w[:, None])
        return np.sqrt(np.diag(np.linalg.pinv(info)))
