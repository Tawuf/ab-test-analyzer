import numpy as np
import pytest
from scipy import stats as sps

from abtest import stats as st
from abtest.simulate import peeking_false_positive_rate


def test_ztest_matches_hand_calculation():
    r = st.two_proportion_ztest(200, 4000, 260, 4000)
    p = 460 / 8000
    se = np.sqrt(p * (1 - p) * (2 / 4000))
    z = (260 / 4000 - 200 / 4000) / se
    assert r.z == pytest.approx(z)
    assert r.p_value == pytest.approx(2 * sps.norm.sf(abs(z)))
    assert r.rel_lift == pytest.approx(0.30)
    assert r.significant


def test_ztest_ci_contains_estimate():
    r = st.two_proportion_ztest(50, 1000, 55, 1000)
    assert r.ci_low < r.abs_diff < r.ci_high
    assert not r.significant


def test_ztest_rejects_bad_input():
    with pytest.raises(ValueError):
        st.two_proportion_ztest(10, 0, 5, 100)
    with pytest.raises(ValueError):
        st.two_proportion_ztest(120, 100, 5, 100)


def test_welch_matches_scipy():
    rng = np.random.default_rng(3)
    a, b = rng.normal(10, 2, 300), rng.normal(10.5, 4, 200)
    r = st.welch_ttest(a, b)
    ref = sps.ttest_ind(b, a, equal_var=False)
    assert r.t == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_sample_size_classic_example():
    # 10 % -> 12 %, alpha 0.05, power 0.8: 3,841 per group
    assert st.sample_size_proportions(0.10, 0.02) == 3841
    assert st.sample_size_proportions(0.10, 0.20, relative=True) == 3841


def test_power_at_planned_sample_size_is_about_80_percent():
    n = st.sample_size_proportions(0.05, 0.01)
    assert st.power_proportions(0.05, 0.06, n) == pytest.approx(0.80, abs=0.01)


def test_srm_detects_broken_split():
    assert st.srm_check(5000, 5050) > 0.05
    assert st.srm_check(5000, 5600) < 0.001


def test_bayesian_symmetric_when_data_identical():
    r = st.bayesian_proportions(100, 2000, 100, 2000, draws=100_000)
    assert r.prob_b_better == pytest.approx(0.5, abs=0.01)


def test_bayesian_detects_clear_winner():
    r = st.bayesian_proportions(100, 2000, 160, 2000)
    assert r.prob_b_better > 0.99
    assert r.lift_ci_low > 0


def test_peeking_inflates_false_positives():
    res = peeking_false_positive_rate(n_experiments=1500, days=15, seed=1)
    assert res["fixed_horizon_fpr"] == pytest.approx(0.05, abs=0.02)
    assert res["peeking_fpr"] > 2 * res["fixed_horizon_fpr"]
