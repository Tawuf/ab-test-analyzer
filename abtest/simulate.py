"""Simulations that show common A/B testing mistakes.

``peeking_false_positive_rate`` runs many A/A tests (no real difference) and
checks the p-value every day, stopping as soon as p < alpha. Stopping early
like this inflates the false positive rate far above the nominal 5 %.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def peeking_false_positive_rate(
    n_experiments: int = 2000,
    days: int = 20,
    users_per_day: int = 500,
    rate: float = 0.05,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Return false positive rates for 'look once at the end' vs. 'peek daily'."""
    rng = np.random.default_rng(seed)
    # daily conversions for both groups, shape (experiments, days)
    conv_a = rng.binomial(users_per_day, rate, (n_experiments, days)).cumsum(axis=1)
    conv_b = rng.binomial(users_per_day, rate, (n_experiments, days)).cumsum(axis=1)
    n = users_per_day * np.arange(1, days + 1)

    pooled = (conv_a + conv_b) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (conv_b / n - conv_a / n) / se, 0.0)
    p = 2 * stats.norm.sf(np.abs(z))

    significant = p < alpha
    fixed_horizon = significant[:, -1].mean()
    ever_significant_by_day = np.maximum.accumulate(significant, axis=1).mean(axis=0)
    return {
        "fixed_horizon_fpr": float(fixed_horizon),
        "peeking_fpr": float(ever_significant_by_day[-1]),
        "peeking_fpr_by_day": ever_significant_by_day.tolist(),
        "days": days,
        "alpha": alpha,
    }
