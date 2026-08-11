import numpy as np

from tamed_langevin import KTULAOptimizer, TRLMCOptimizer


def objective(theta):
    return float(np.sum(theta**6 / 6.0 - theta**2 / 2.0))


def grad(theta):
    return theta**5 - theta


def run_optimizer(name, optimizer, theta0, n_steps, seed):
    rng = np.random.default_rng(seed)
    theta = theta0.copy()

    values = []
    active_fractions = []

    for _ in range(n_steps):
        g = grad(theta)

        if name == "kTULA":
            theta, active = optimizer.step(theta, g, rng)
        elif name == "tRLMC":
            theta, active = optimizer.step(theta, g, grad, rng)
        else:
            raise ValueError(name)

        values.append(objective(theta))
        active_fractions.append(float(active))

    return theta, np.asarray(values), np.asarray(active_fractions)


def main():
    theta0 = np.zeros(20)
    theta0[0] = 10.0

    ktula = KTULAOptimizer(
        step_size=0.01, beta=1.0e6, a_tame=0.05, ell_tame=4.0
    )
    trlmc = TRLMCOptimizer(
        step_size=0.01, beta=1.0e6, a_tame=0.05, ell_tame=4.0
    )

    theta_k, values_k, active_k = run_optimizer(
        name="kTULA",
        optimizer=ktula,
        theta0=theta0,
        n_steps=1000,
        seed=1,
    )

    theta_r, values_r, active_r = run_optimizer(
        name="tRLMC",
        optimizer=trlmc,
        theta0=theta0,
        n_steps=1000,
        seed=1,
    )

    print("kTULA final objective:", values_k[-1])
    print("tRLMC final objective:", values_r[-1])
    print("kTULA final norm:", np.linalg.norm(theta_k))
    print("tRLMC final norm:", np.linalg.norm(theta_r))
    print("kTULA mean active fraction:", np.mean(active_k))
    print("tRLMC mean active fraction:", np.mean(active_r))


if __name__ == "__main__":
    main()
