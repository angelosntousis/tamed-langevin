import numpy as np

from tamed_langevin import KTULASampler, TRLMCSampler


def drift(x):
    out = np.empty_like(x)
    out[0] = x[0] ** 3 - x[0]
    out[1:] = 100.0 * x[1:]
    return out


def main():
    x0 = np.zeros(100)
    x0[0] = 200.0

    ktula = KTULASampler(
        drift=drift,
        step_size=0.01,
        beta=1.0,
        a_tame=0.05,
        ell_tame=2.0,
    )

    trlmc = TRLMCSampler(
        drift=drift,
        step_size=0.01,
        beta=1.0,
        a_tame=0.05,
        ell_tame=2.0,
    )

    ktula_samples, ktula_active = ktula.sample(
        x0,
        n_steps=10000,
        burn_in=1000,
        seed=1,
        return_active=True,
    )

    trlmc_samples, trlmc_active = trlmc.sample(
        x0,
        n_steps=10000,
        burn_in=1000,
        seed=1,
        return_active=True,
    )

    print("kTULA samples:", ktula_samples.shape)
    print("tRLMC samples:", trlmc_samples.shape)
    print("kTULA active fraction:", np.mean(ktula_active))
    print("tRLMC active fraction:", np.mean(trlmc_active))


if __name__ == "__main__":
    main()
