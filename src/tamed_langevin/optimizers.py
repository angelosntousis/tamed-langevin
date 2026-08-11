from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tamed_langevin.taming import adaptive_tamed_drift


Array = np.ndarray
GradFn = Callable[[Array], Array]


@dataclass
class KTULAOptimizer:
    step_size: float
    beta: float = 1.0e6
    a_tame: float = 0.05
    ell_tame: float = 0.0

    def step(
        self,
        theta: Array,
        grad: Array,
        rng: np.random.Generator,
    ) -> tuple[Array, bool]:
        theta = np.asarray(theta, dtype=float)
        grad = np.asarray(grad, dtype=float)

        tamed_grad, active = adaptive_tamed_drift(
            drift=grad,
            state=theta,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )

        noise = rng.normal(size=theta.shape)
        theta_next = (
            theta
            - self.step_size * tamed_grad
            + np.sqrt(2.0 * self.step_size / self.beta) * noise
        )

        return theta_next, active


@dataclass
class TRLMCOptimizer:
    step_size: float
    beta: float = 1.0e6
    a_tame: float = 0.05
    ell_tame: float = 0.0

    def step(
        self,
        theta: Array,
        grad: Array,
        grad_at: GradFn,
        rng: np.random.Generator,
    ) -> tuple[Array, bool]:
        theta = np.asarray(theta, dtype=float)
        grad = np.asarray(grad, dtype=float)

        tau = rng.uniform(0.0, 1.0)

        z1 = rng.normal(size=theta.shape)
        z2 = rng.normal(size=theta.shape)

        dW_tau = np.sqrt(tau * self.step_size) * z1
        dW_full = dW_tau + np.sqrt((1.0 - tau) * self.step_size) * z2

        brownian_scale = np.sqrt(2.0 / self.beta)

        tamed_grad, active = adaptive_tamed_drift(
            drift=grad,
            state=theta,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )

        theta_tau = (
            theta
            - self.step_size * tau * tamed_grad
            + brownian_scale * dW_tau
        )

        grad_tau = np.asarray(grad_at(theta_tau), dtype=float)

        tamed_grad_tau, _ = adaptive_tamed_drift(
            drift=grad_tau,
            state=theta_tau,
            step_size=self.step_size,
            a_tame=self.a_tame,
            ell=self.ell_tame,
        )

        theta_next = (
            theta
            - self.step_size * tamed_grad_tau
            + brownian_scale * dW_full
        )

        return theta_next, active
