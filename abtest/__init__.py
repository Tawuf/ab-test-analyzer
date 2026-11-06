"""A/B test analysis toolkit."""
from .stats import (
    bayesian_proportions,
    power_proportions,
    sample_size_proportions,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
)

__all__ = [
    "bayesian_proportions",
    "power_proportions",
    "sample_size_proportions",
    "srm_check",
    "two_proportion_ztest",
    "welch_ttest",
]
__version__ = "1.0.0"
