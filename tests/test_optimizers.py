import numpy as np

from tamed_langevin.optimizers import KTULAOptimizer, TRLMCOptimizer


def objective_grad(theta):
    return theta**5 - theta


def test_ktula_optimizer_shape():
    rng = np.random.default_rng(1)
    theta = np.ones(5)
    grad = objective_grad(theta)

    optimizer = KTULAOptimizer(step_size=0.01, beta=1.0e6, a_tame=0.05)
    theta_next, active = optimizer.step(theta, grad, rng)

    assert theta_next.shape == theta.shape
    assert isinstance(active, bool)
    assert np.isfinite(theta_next).all()


def test_trlmc_optimizer_shape():
    rng = np.random.default_rng(1)
    theta = np.ones(5)
    grad = objective_grad(theta)

    optimizer = TRLMCOptimizer(step_size=0.01, beta=1.0e6, a_tame=0.05)
    theta_next, active = optimizer.step(
        theta=theta,
        grad=grad,
        grad_at=objective_grad,
        rng=rng,
    )

    assert theta_next.shape == theta.shape
    assert isinstance(active, bool)
    assert np.isfinite(theta_next).all()


def test_ktula_optimizer_active_fraction():
    rng = np.random.default_rng(1)
    theta = np.array([200.0, 0.0, 0.0])
    grad = objective_grad(theta)

    optimizer = KTULAOptimizer(step_size=0.1, beta=1.0e6, a_tame=0.05)
    theta_next, active = optimizer.step(theta, grad, rng)

    assert theta_next.shape == theta.shape
    assert isinstance(active, bool)
