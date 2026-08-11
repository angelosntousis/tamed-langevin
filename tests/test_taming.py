import numpy as np

from tamed_langevin.taming import (
    adaptive_tamed_drift,
    check_g_c1,
    g_switch,
    tamed_drift,
)


def test_g_switch_values():
    values = g_switch(np.array([0.0, 0.5, 1.0, 2.0, 3.0]))
    assert np.allclose(values[[0, 1, 2]], [0.0, 0.0, 0.0])
    assert np.isclose(values[3], 2.0)
    assert np.isclose(values[4], 3.0)


def test_g_switch_c1():
    check_g_c1()


def test_adaptive_taming_inactive_for_small_drift():
    drift = np.array([0.1, -0.2])
    state = np.array([1.0, -1.0])
    tamed, active = adaptive_tamed_drift(drift, state, step_size=0.01, a_tame=0.05)
    assert active is False
    assert np.allclose(tamed, drift)


def test_adaptive_taming_uses_global_norm_and_single_denominator():
    drift = np.array([1.0, 2.0])
    state = np.array([1.0, 2.0])
    tamed, active = adaptive_tamed_drift(drift, state, step_size=1.0, a_tame=0.0)

    # lambda * ||state||^2 = 5 for ell=0, so g(5) = 5.
    expected = drift / np.sqrt(6.0)

    assert tamed.shape == drift.shape
    assert active is True
    assert np.allclose(tamed, expected)


def test_adaptive_taming_couples_coordinates_through_global_norm():
    drift = np.array([0.1, 0.1])
    state = np.array([0.8, 0.8])
    tamed, active = adaptive_tamed_drift(drift, state, step_size=1.0, a_tame=0.0)

    # Neither coordinate crosses the threshold alone, but the vector norm does.
    assert active is True
    assert not np.allclose(tamed, drift)


def test_adaptive_taming_respects_ell_exponent():
    drift = np.array([2.0])
    state = np.array([0.8])

    tamed_ell0, active_ell0 = adaptive_tamed_drift(
        drift, state, step_size=2.0, a_tame=0.0, ell=0.0
    )
    untamed_ell2, active_ell2 = adaptive_tamed_drift(
        drift, state, step_size=2.0, a_tame=0.0, ell=2.0
    )

    # 2 * 0.8^2 >= 1, whereas 2 * 0.8^6 < 1.
    assert active_ell0 is True
    assert active_ell2 is False
    assert not np.allclose(tamed_ell0, drift)
    assert np.allclose(untamed_ell2, drift)


def test_unswitched_taming_uses_global_denominator_without_g():
    drift = np.array([1.0, 2.0])
    state = np.array([1.0, 2.0])
    result = tamed_drift(drift, state, step_size=1.0, a_tame=0.0, ell=0.0)
    assert np.allclose(result, drift / np.sqrt(6.0))
