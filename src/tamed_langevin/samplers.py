from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tamed_langevin.taming import adaptive_tamed_drift, tamed_drift


Array = np.ndarray
DriftFn = Callable[[Array], Array]


@dataclass
class StandardKTULASampler:
    """Global kTULA sampler using the unswitched taming denominator."""

    drift: DriftFn
    step_size: float
    beta: float = 1.0
    a_tame: float = 0.05
    ell_tame: float = 0.0

    def step(self, x: Array, rng: np.random.Generator) -> tuple[Array, bool]:
        h = self.drift(x)
        h_tamed = tamed_drift(
            drift=h,
            state=x,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )
        noise = rng.normal(size=x.shape)
        x_next = (
            x
            - self.step_size * h_tamed
            + np.sqrt(2.0 * self.step_size / self.beta) * noise
        )
        return x_next, True

    def sample(
        self,
        x0: Array,
        n_steps: int,
        burn_in: int = 0,
        seed: int | None = None,
    ) -> Array:
        rng = np.random.default_rng(seed)
        x = np.asarray(x0, dtype=float).copy()
        samples = []

        for step in range(n_steps):
            x, _ = self.step(x, rng)
            if not np.isfinite(x).all():
                raise FloatingPointError(f"kTULA diverged at step {step}")
            if step >= burn_in:
                samples.append(x.copy())

        return np.asarray(samples)


@dataclass
class KTULASampler:
    drift: DriftFn
    step_size: float
    beta: float = 1.0
    a_tame: float = 0.05
    ell_tame: float = 0.0

    def step(self, x: Array, rng: np.random.Generator) -> tuple[Array, bool]:
        h = self.drift(x)
        h_tamed, active = adaptive_tamed_drift(
            drift=h,
            state=x,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )

        noise = rng.normal(size=x.shape)
        x_next = (
            x
            - self.step_size * h_tamed
            + np.sqrt(2.0 * self.step_size / self.beta) * noise
        )

        return x_next, active

    def sample(
        self,
        x0: Array,
        n_steps: int,
        burn_in: int = 0,
        seed: int | None = None,
        return_active: bool = False,
    ):
        rng = np.random.default_rng(seed)
        x = np.asarray(x0, dtype=float).copy()

        samples = []
        active_fractions = []

        for step in range(n_steps):
            x, active = self.step(x, rng)

            if not np.isfinite(x).all():
                raise FloatingPointError(f"kTULA diverged at step {step}")

            if step >= burn_in:
                samples.append(x.copy())
                if return_active:
                    active_fractions.append(float(active))

        samples_array = np.asarray(samples)

        if return_active:
            return samples_array, np.asarray(active_fractions)

        return samples_array


@dataclass
class TRLMCSampler:
    drift: DriftFn
    step_size: float
    beta: float = 1.0
    a_tame: float = 0.05
    ell_tame: float = 0.0

    def step(self, x: Array, rng: np.random.Generator) -> tuple[Array, bool]:
        tau = rng.uniform(0.0, 1.0)

        z1 = rng.normal(size=x.shape)
        z2 = rng.normal(size=x.shape)

        dW_tau = np.sqrt(tau * self.step_size) * z1
        dW_full = dW_tau + np.sqrt((1.0 - tau) * self.step_size) * z2

        brownian_scale = np.sqrt(2.0 / self.beta)

        h_x = self.drift(x)
        h_tamed_x, active = adaptive_tamed_drift(
            drift=h_x,
            state=x,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )

        x_tau = (
            x
            - self.step_size * tau * h_tamed_x
            + brownian_scale * dW_tau
        )

        h_tau = self.drift(x_tau)
        h_tamed_tau, _ = adaptive_tamed_drift(
            drift=h_tau,
            state=x_tau,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )

        x_next = (
            x
            - self.step_size * h_tamed_tau
            + brownian_scale * dW_full
        )

        return x_next, active

    def sample(
        self,
        x0: Array,
        n_steps: int,
        burn_in: int = 0,
        seed: int | None = None,
        return_active: bool = False,
    ):
        rng = np.random.default_rng(seed)
        x = np.asarray(x0, dtype=float).copy()

        samples = []
        active_fractions = []

        for step in range(n_steps):
            x, active = self.step(x, rng)

            if not np.isfinite(x).all():
                raise FloatingPointError(f"tRLMC diverged at step {step}")

            if step >= burn_in:
                samples.append(x.copy())
                if return_active:
                    active_fractions.append(float(active))

        samples_array = np.asarray(samples)

        if return_active:
            return samples_array, np.asarray(active_fractions)

        return samples_array
