import numpy as np

from tamed_langevin.samplers import KTULASampler, StandardKTULASampler, TRLMCSampler


def drift(x):
    return x**3 - x


def test_ktula_sampler_shape():
    sampler = KTULASampler(drift=drift, step_size=0.01, beta=1.0)
    x0 = np.zeros(5)

    samples = sampler.sample(x0, n_steps=20, burn_in=5, seed=1, return_active=False)
    samples = np.asarray(samples)

    assert samples.shape == (15, 5)
    assert np.isfinite(samples).all()


def test_standard_ktula_sampler_shape():
    sampler = StandardKTULASampler(drift=drift, step_size=0.01, beta=1.0)
    samples = sampler.sample(np.zeros(5), n_steps=20, burn_in=5, seed=1)
    assert samples.shape == (15, 5)
    assert np.isfinite(samples).all()


def test_trlmc_sampler_shape():
    sampler = TRLMCSampler(drift=drift, step_size=0.01, beta=1.0)
    x0 = np.zeros(5)

    samples = sampler.sample(x0, n_steps=20, burn_in=5, seed=1, return_active=False)
    samples = np.asarray(samples)

    assert samples.shape == (15, 5)
    assert np.isfinite(samples).all()


def test_active_fraction_output():
    sampler = KTULASampler(
        drift=drift,
        step_size=0.1,
        beta=1.0,
        ell_tame=2.0,
    )
    x0 = np.array([200.0, 0.0, 0.0])

    samples, active = sampler.sample(
        x0,
        n_steps=10,
        burn_in=0,
        seed=1,
        return_active=True,
    )

    assert samples.shape == (10, 3)
    assert active.shape == (10,)
    assert np.all((active >= 0.0) & (active <= 1.0))
