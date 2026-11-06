"""Statistical tests for A/B experiments.

Frequentist
  * two_proportion_ztest   conversion rates (binary metric)
  * welch_ttest            averages such as revenue per user (continuous metric)
  * srm_check              sample ratio mismatch (is the traffic split broken?)
  * sample_size_proportions / power_proportions   planning an experiment

Bayesian
  * bayesian_proportions   Beta-Binomial model: P(B > A), expected loss, credible interval
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ProportionResult:
    rate_a: float
    rate_b: float
    abs_diff: float
    rel_lift: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def as_dict(self) -> dict:
        return asdict(self) | {"significant": self.significant}


def two_proportion_ztest(
    conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05
) -> ProportionResult:
    """Two-sided z-test for p_B - p_A.

    The test statistic uses the pooled standard error (valid under H0); the
    confidence interval uses the unpooled standard error.
    """
    if min(n_a, n_b) <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= conv_a <= n_a and 0 <= conv_b <= n_b):
        raise ValueError("conversions must be between 0 and the sample size")
    p_a, p_b = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    diff = p_b - p_a
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return ProportionResult(
        rate_a=p_a,
        rate_b=p_b,
        abs_diff=diff,
        rel_lift=diff / p_a if p_a > 0 else float("nan"),
        z=float(z),
        p_value=float(p_value),
        ci_low=float(diff - z_crit * se),
        ci_high=float(diff + z_crit * se),
        alpha=alpha,
    )


@dataclass(frozen=True)
class MeanResult:
    mean_a: float
    mean_b: float
    diff: float
    t: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def as_dict(self) -> dict:
        return asdict(self) | {"significant": self.significant}


def welch_ttest(a, b, alpha: float = 0.05) -> MeanResult:
    """Welch's t-test (unequal variances) for mean(B) - mean(A)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    diff = b.mean() - a.mean()
    t = diff / se
    p_value = 2 * stats.t.sf(abs(t), df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return MeanResult(
        mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(diff), t=float(t),
        df=float(df), p_value=float(p_value),
        ci_low=float(diff - t_crit * se), ci_high=float(diff + t_crit * se), alpha=alpha,
    )


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Chi-square p-value for sample ratio mismatch. p < 0.001 means: investigate!"""
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    return float(stats.chisquare([n_a, n_b], expected).pvalue)


def sample_size_proportions(
    baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80, relative: bool = False
) -> int:
    """Users needed *per group* to detect an effect of size ``mde``.

    ``mde`` is an absolute difference (0.01 = one percentage point) unless
    ``relative=True`` (0.10 = +10 % of the baseline).
    """
    p1 = baseline
    p2 = baseline * (1 + mde) if relative else baseline + mde
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("baseline and baseline+mde must be different rates in (0, 1)")
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p2 - p1) ** 2))


def power_proportions(baseline: float, treatment: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Probability of detecting a true difference baseline -> treatment."""
    p_bar = (baseline + treatment) / 2
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se1 = np.sqrt((baseline * (1 - baseline) + treatment * (1 - treatment)) / n_per_group)
    z_a = stats.norm.ppf(1 - alpha / 2)
    delta = abs(treatment - baseline)
    return float(stats.norm.cdf((delta - z_a * se0) / se1) + stats.norm.cdf((-delta - z_a * se0) / se1))


@dataclass(frozen=True)
class BayesResult:
    prob_b_better: float
    expected_loss_choose_b: float  # conversion rate we lose (on average) if we pick B and it is worse
    lift_ci_low: float
    lift_ci_high: float
    samples_a: np.ndarray
    samples_b: np.ndarray


def bayesian_proportions(
    conv_a: int, n_a: int, conv_b: int, n_b: int,
    prior: tuple[float, float] = (1.0, 1.0), draws: int = 200_000, seed: int = 0,
    credible: float = 0.95,
) -> BayesResult:
    """Beta-Binomial A/B test using Monte Carlo draws from both posteriors."""
    rng = np.random.default_rng(seed)
    a0, b0 = prior
    pa = rng.beta(a0 + conv_a, b0 + n_a - conv_a, draws)
    pb = rng.beta(a0 + conv_b, b0 + n_b - conv_b, draws)
    lift = (pb - pa) / pa
    tail = (1 - credible) / 2
    return BayesResult(
        prob_b_better=float((pb > pa).mean()),
        expected_loss_choose_b=float(np.maximum(pa - pb, 0).mean()),
        lift_ci_low=float(np.quantile(lift, tail)),
        lift_ci_high=float(np.quantile(lift, 1 - tail)),
        samples_a=pa,
        samples_b=pb,
    )
