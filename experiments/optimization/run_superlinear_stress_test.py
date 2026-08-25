#!/usr/bin/env python3
"""
Super-linear drift stress test for adaptive-tamed Langevin optimizers.

This script compares untamed SGD, Adam, RMSProp, AMSGrad with adaptive-tamed
kTULA and tRLMC on a nonlinear objective with super-linear drift.

The reusable kTULA/tRLMC implementations are imported from src/tamed_langevin.
This file keeps only experiment-specific code: data generation, neural-network
objective, plotting, CSV summaries, and experiment orchestration.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from tamed_langevin import KTULAOptimizer, TRLMCOptimizer
from tamed_langevin.taming import check_g_c1, g_switch


# ============================================================
# Publication styling
# ============================================================
mpl.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 12.5,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "figure.dpi": 130,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
})

COLORS = {
    "SGD": "#4C72B0",
    "ADAM": "#DD8452",
    "RMSPROP": "#937860",
    "AMSGRAD": "#55A868",
    "KTULA": "#C44E52",
    "TRLMC": "#8172B3",
}

LABELS = {
    "SGD": "SGD",
    "ADAM": "Adam",
    "RMSPROP": "RMSProp",
    "AMSGRAD": "AMSGrad",
    "KTULA": "adTULA",
    "TRLMC": "adTRLMC",
}

METHODS = ["SGD", "ADAM", "RMSPROP", "AMSGRAD", "KTULA", "TRLMC"]
TAMED_METHODS = ["KTULA", "TRLMC"]


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class Config:
    din: int = 20
    width: int = 64
    depth: int = 1
    n_train: int = 2000
    n_test: int = 1000
    batch: int = 128
    n_epochs: int = 60
    seeds: Tuple[int, ...] = (1, 2, 3, 4, 5)

    lr: float = 0.2 #this lr produces the best results for all methods in the main stress test
    sweep_lrs: Tuple[float, ...] = (0.05, 0.10, 0.20)
    eta: float = 0.25
    init_scale: float = 2.0

    beta: float = 1.0e6
    a_tame: float = 0.05
    ell_tame: float = 3.0

    momentum: float = 0.9
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1.0e-8
    rms_alpha: float = 0.99
    rms_eps: float = 1.0e-8

    explode_cap: float = 1.0e12
    theta_cap: float = 1.0e4

    regime_init_scales: Tuple[float, ...] = (1.50, 1.75, 2.00, 2.25, 2.50)
    regime_etas: Tuple[float, ...] = (0.25, 0.50, 1.00, 2.00)
    regime_seeds: Tuple[int, ...] = (1, 2, 3)
    regime_epochs: int = 8

    order_T: float = 1.0
    order_x0: float = 2.5
    order_beta: float = 1.0e6
    order_a_tame: float = 0.05
    order_ell_tame: float = 1.0
    order_lambdas: Tuple[float, ...] = (
        2.0 ** -4,
        2.0 ** -5,
        2.0 ** -6,
        2.0 ** -7,
        2.0 ** -8,
    )
    order_lambda_ref: float = 2.0 ** -12
    order_samples: int = 4096
    order_seed: int = 24680

    out_dir: str = "./figures/SuperLinear_SINUM"

    def quick(self) -> "Config":
        return replace(
            self,
            width=32,
            n_train=1000,
            n_test=500,
            n_epochs=8,
            seeds=(1, 2),
            sweep_lrs=(0.05, 0.10),
            regime_init_scales=(1.50, 2.00),
            regime_etas=(0.50, 1.00),
            regime_seeds=(1,),
            regime_epochs=4,
            order_lambdas=(2.0 ** -4, 2.0 ** -5, 2.0 ** -6),
            order_lambda_ref=2.0 ** -10,
            order_samples=1024,
        )


# ============================================================
# Network utilities
# ============================================================
def make_shapes(cfg: Config) -> Tuple[List[Tuple[int, int]], List[Tuple[int, ...]]]:
    weight_shapes = (
        [(cfg.din, cfg.width)]
        + [(cfg.width, cfg.width)] * (cfg.depth - 1)
        + [(cfg.width, 1)]
    )
    bias_shapes = [(cfg.width,)] * cfg.depth + [(1,)]
    return weight_shapes, bias_shapes


def parameter_dimension(cfg: Config) -> int:
    weight_shapes, bias_shapes = make_shapes(cfg)
    return sum(a * b for a, b in weight_shapes) + sum(s[0] for s in bias_shapes)


def unflatten(theta: np.ndarray, cfg: Config):
    weight_shapes, bias_shapes = make_shapes(cfg)
    weights, biases, i = [], [], 0

    for shape in weight_shapes:
        n = shape[0] * shape[1]
        weights.append(theta[i:i + n].reshape(shape))
        i += n

    for shape in bias_shapes:
        n = shape[0]
        biases.append(theta[i:i + n].reshape(shape))
        i += n

    return weights, biases


def init_params(cfg: Config, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weight_shapes, bias_shapes = make_shapes(cfg)
    weights = [rng.standard_normal(s) * scale for s in weight_shapes]
    biases = [np.zeros(s) for s in bias_shapes]
    return np.concatenate([w.ravel() for w in weights] + [b.ravel() for b in biases])


def silu(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -30.0, 30.0)
    return x / (1.0 + np.exp(-z))


def silu_prime(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -30.0, 30.0)
    s = 1.0 / (1.0 + np.exp(-z))
    return s + x * s * (1.0 - s)


def predict(theta: np.ndarray, z: np.ndarray, cfg: Config) -> np.ndarray:
    weights, biases = unflatten(theta, cfg)

    h = z
    for ell, (w, b) in enumerate(zip(weights, biases)):
        pre = h @ w + b
        h = silu(pre) if ell < len(weights) - 1 else pre

    return h.ravel()


def reg_value(theta: np.ndarray, eta: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        val = (eta / 6.0) * np.sum(np.abs(theta) ** 6)
    return float(val)


def data_grad(
    theta: np.ndarray,
    z_batch: np.ndarray,
    y_batch: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    weights, biases = unflatten(theta, cfg)

    acts = [z_batch]
    pres = []
    h = z_batch

    for ell, (w, b) in enumerate(zip(weights, biases)):
        pre = h @ w + b
        pres.append(pre)
        h = silu(pre) if ell < len(weights) - 1 else pre
        acts.append(h)

    pred = h.ravel()
    resid = pred - y_batch
    batch_n = len(y_batch)
    delta = (2.0 / batch_n * resid)[:, None]

    grad_weights_reversed: List[np.ndarray] = []
    grad_biases_reversed: List[np.ndarray] = []

    for ell in reversed(range(len(weights))):
        grad_w = acts[ell].T @ delta
        grad_b = delta.sum(axis=0)

        grad_weights_reversed.append(grad_w)
        grad_biases_reversed.append(grad_b)

        if ell > 0:
            delta = (delta @ weights[ell].T) * silu_prime(pres[ell - 1])

    grad_weights = list(reversed(grad_weights_reversed))
    grad_biases = list(reversed(grad_biases_reversed))

    return np.concatenate(
        [gw.ravel() for gw in grad_weights]
        + [gb.ravel() for gb in grad_biases]
    )


def full_grad(
    theta: np.ndarray,
    z_batch: np.ndarray,
    y_batch: np.ndarray,
    cfg: Config,
    eta: float,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return data_grad(theta, z_batch, y_batch, cfg) + eta * theta * np.abs(theta) ** 4


def full_objective(
    theta: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    eta: float,
) -> float:
    if (not np.isfinite(theta).all()) or np.max(np.abs(theta)) > cfg.theta_cap:
        return cfg.explode_cap

    with np.errstate(over="ignore", invalid="ignore"):
        pred = predict(theta, z, cfg)
        mse = np.mean((y - pred) ** 2)
        obj = mse + reg_value(theta, eta)

    if not np.isfinite(obj):
        return cfg.explode_cap

    return float(min(obj, cfg.explode_cap))


def make_dataset(cfg: Config, seed: int):
    rng = np.random.default_rng(seed)

    x = rng.standard_normal((cfg.n_train + cfg.n_test, cfg.din))
    y = (
        np.sin(3.0 * x[:, 0])
        + 0.5 * x[:, 1] ** 2
        - 0.3 * np.cos(2.0 * x[:, 2])
        + 0.1 * rng.standard_normal(cfg.n_train + cfg.n_test)
    )

    return (x[:cfg.n_train], y[:cfg.n_train]), (x[cfg.n_train:], y[cfg.n_train:])

def compute_null_baselines(cfg: Config) -> Dict[str, float]:
    train_mses = []
    test_mses = []
    constants = []

    for seed in cfg.seeds:
        (z_train, y_train), (z_test, y_test) = make_dataset(cfg, seed)

        null_pred = float(np.mean(y_train))
        null_train_mse = float(np.mean((y_train - null_pred) ** 2))
        null_test_mse = float(np.mean((y_test - null_pred) ** 2))

        constants.append(null_pred)
        train_mses.append(null_train_mse)
        test_mses.append(null_test_mse)

    return {
        "median_constant": float(np.median(constants)),
        "median_train_mse": float(np.median(train_mses)),
        "median_test_mse": float(np.median(test_mses)),
    }


# ============================================================
# Training
# ============================================================
def method_seed_offset(method: str) -> int:
    return 1000 + 97 * METHODS.index(method)


def is_diverged(theta: np.ndarray, cfg: Config) -> bool:
    return (not np.isfinite(theta).all()) or np.max(np.abs(theta)) > cfg.theta_cap


def train_one(
    method: str,
    seed: int,
    cfg: Config,
    lr: Optional[float] = None,
    eta: Optional[float] = None,
    init_scale: Optional[float] = None,
    n_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")

    lr = cfg.lr if lr is None else lr
    eta = cfg.eta if eta is None else eta
    init_scale = cfg.init_scale if init_scale is None else init_scale
    n_epochs = cfg.n_epochs if n_epochs is None else n_epochs

    (z_train, y_train), (z_test, y_test) = make_dataset(cfg, seed)

    theta = init_params(cfg, init_scale, seed + 100)
    dim = theta.size

    nb = cfg.n_train // cfg.batch
    total_steps = n_epochs * nb

    rng_perm = np.random.default_rng(10_000 + seed)
    rng_noise = np.random.default_rng(20_000 + seed + method_seed_offset(method))

    vel = np.zeros(dim)
    m = np.zeros(dim)
    v = np.zeros(dim)
    vmax = np.zeros(dim)
    rms_avg = np.zeros(dim)
    t_adam = 0

    obj_h, test_h, norm_h, active_h = [], [], [], []
    diverged_at = None

    for _ in range(n_epochs):
        perm = rng_perm.permutation(cfg.n_train)

        for bi in range(nb):
            idx = perm[bi * cfg.batch:(bi + 1) * cfg.batch]
            z_batch, y_batch = z_train[idx], y_train[idx]

            with np.errstate(over="ignore", invalid="ignore"):
                grad = full_grad(theta, z_batch, y_batch, cfg, eta)

            active_frac = 0.0

            if (not np.isfinite(grad).all()) or np.max(np.abs(grad)) > cfg.explode_cap:
                diverged_at = len(obj_h) + 1

            else:
                if method == "SGD":
                    vel = cfg.momentum * vel + grad
                    theta = theta - lr * vel

                elif method in ("ADAM", "AMSGRAD"):
                    t_adam += 1
                    m = cfg.adam_b1 * m + (1.0 - cfg.adam_b1) * grad
                    v = cfg.adam_b2 * v + (1.0 - cfg.adam_b2) * grad ** 2

                    m_hat = m / (1.0 - cfg.adam_b1 ** t_adam)
                    v_hat = v / (1.0 - cfg.adam_b2 ** t_adam)

                    if method == "AMSGRAD":
                        vmax = np.maximum(vmax, v_hat)
                        denom = np.sqrt(vmax) + cfg.adam_eps
                    else:
                        denom = np.sqrt(v_hat) + cfg.adam_eps

                    theta = theta - lr * m_hat / denom

                elif method == "RMSPROP":
                    rms_avg = cfg.rms_alpha * rms_avg + (1.0 - cfg.rms_alpha) * grad ** 2
                    theta = theta - lr * grad / (np.sqrt(rms_avg) + cfg.rms_eps)

                elif method == "KTULA":
                    optimizer = KTULAOptimizer(
                        step_size=lr,
                        beta=cfg.beta,
                        a_tame=cfg.a_tame,
                        ell_tame=cfg.ell_tame,
                    )

                    theta, active = optimizer.step(theta, grad, rng_noise)
                    active_frac = float(active)

                elif method == "TRLMC":
                    optimizer = TRLMCOptimizer(
                        step_size=lr,
                        beta=cfg.beta,
                        a_tame=cfg.a_tame,
                        ell_tame=cfg.ell_tame,
                    )

                    def grad_at(theta_mid: np.ndarray) -> np.ndarray:
                        return full_grad(theta_mid, z_batch, y_batch, cfg, eta)

                    theta, active = optimizer.step(
                        theta=theta,
                        grad=grad,
                        grad_at=grad_at,
                        rng=rng_noise,
                    )

                    active_frac = float(active)

            if diverged_at is not None or is_diverged(theta, cfg):
                if diverged_at is None:
                    diverged_at = len(obj_h) + 1

                remaining = total_steps - len(obj_h)
                obj_h.extend([cfg.explode_cap] * remaining)
                test_h.extend([cfg.explode_cap] * remaining)
                norm_h.extend([cfg.theta_cap] * remaining)
                active_h.extend([active_frac] * remaining)
                break

            obj = full_objective(theta, z_train, y_train, cfg, eta)
            pred_test = predict(theta, z_test, cfg)

            if np.isfinite(pred_test).all():
                test_mse = float(np.mean((y_test - pred_test) ** 2))
            else:
                test_mse = cfg.explode_cap

            if np.isfinite(theta).all():
                theta_norm = float(np.linalg.norm(theta))
            else:
                theta_norm = cfg.theta_cap

            obj_h.append(min(obj, cfg.explode_cap))
            test_h.append(min(test_mse, cfg.explode_cap))
            norm_h.append(min(theta_norm, cfg.theta_cap))
            active_h.append(active_frac)

        if diverged_at is not None:
            break

    while len(obj_h) < total_steps:
        obj_h.append(obj_h[-1] if obj_h else cfg.explode_cap)
        test_h.append(test_h[-1] if test_h else cfg.explode_cap)
        norm_h.append(norm_h[-1] if norm_h else cfg.theta_cap)
        active_h.append(active_h[-1] if active_h else 0.0)

    return {
        "obj": np.asarray(obj_h),
        "test": np.asarray(test_h),
        "norm": np.asarray(norm_h),
        "active": np.asarray(active_h),
        "diverged_at": diverged_at,
        "final_obj": float(obj_h[-1]),
        "final_test": float(test_h[-1]),
        "final_norm": float(norm_h[-1]),
    }


# ============================================================
# Observed-order experiment
# ============================================================
def order_potential(x: np.ndarray) -> np.ndarray:
    return (x ** 6) / 6.0 - 0.5 * x ** 2 + 1.0 / 3.0


def order_drift(x: np.ndarray) -> np.ndarray:
    return x ** 5 - x


def tamed_order_drift(
    h: np.ndarray,
    x: np.ndarray,
    lam: float,
    a_tame: float,
    ell_tame: float,
) -> np.ndarray:
    # Each element is an independent one-dimensional Monte Carlo path. In one
    # dimension the global norm is |x|, so apply the formula pathwise.
    switch = g_switch(lam * np.abs(x) ** (2.0 * (ell_tame + 1.0)))
    return a_tame * x + (h - a_tame * x) / np.sqrt(1.0 + switch)


def simulate_order_method(method: str, lam: float, cfg: Config, seed: int) -> Dict[str, float]:
    if method not in TAMED_METHODS:
        raise ValueError(
            "Observed-order experiment only supports adaptive kTULA and tRLMC."
        )

    n_steps = int(round(cfg.order_T / lam))
    if abs(n_steps * lam - cfg.order_T) > 1.0e-12:
        raise ValueError("order_T must be an integer multiple of every lambda.")

    rng = np.random.default_rng(seed)
    x = np.full(cfg.order_samples, cfg.order_x0, dtype=float)
    sigma = np.sqrt(2.0 / cfg.order_beta)

    for _ in range(n_steps):
        h = order_drift(x)

        if method == "KTULA":
            h_t = tamed_order_drift(
                h, x, lam, cfg.order_a_tame, cfg.order_ell_tame
            )
            x = x - lam * h_t + sigma * np.sqrt(lam) * rng.standard_normal(cfg.order_samples)

        elif method == "TRLMC":
            tau = rng.uniform(0.0, 1.0, size=cfg.order_samples)
            z1 = rng.standard_normal(cfg.order_samples)
            z2 = rng.standard_normal(cfg.order_samples)

            d_w_tau = np.sqrt(tau * lam) * z1
            d_w_full = d_w_tau + np.sqrt((1.0 - tau) * lam) * z2

            h_t1 = tamed_order_drift(
                h, x, lam, cfg.order_a_tame, cfg.order_ell_tame
            )
            x_mid = x - tau * lam * h_t1 + sigma * d_w_tau

            h_mid = order_drift(x_mid)
            h_t2 = tamed_order_drift(
                h_mid, x_mid, lam, cfg.order_a_tame, cfg.order_ell_tame
            )

            x = x - lam * h_t2 + sigma * d_w_full

        x = np.clip(x, -1.0e6, 1.0e6)

    u = order_potential(x)

    return {
        "mean_x": float(np.mean(x)),
        "mean_x2": float(np.mean(x ** 2)),
        "mean_u": float(np.mean(u)),
        "std_u": float(np.std(u)),
    }


def run_observed_order(cfg: Config):
    print("\nObserved-order experiment on a 1D super-linear SDE:")

    rows = []
    refs = {}

    for method in TAMED_METHODS:
        ref = simulate_order_method(
            method,
            cfg.order_lambda_ref,
            cfg,
            cfg.order_seed + 31 * METHODS.index(method),
        )
        refs[method] = ref

        print(
            f"  reference {LABELS[method]}: "
            f"lambda_ref={cfg.order_lambda_ref:.4e}, "
            f"E[u]={ref['mean_u']:.6e}, "
            f"E[X^2]={ref['mean_x2']:.6e}"
        )

        errors_u, errors_x2 = [], []

        for lam in cfg.order_lambdas:
            out = simulate_order_method(
                method,
                lam,
                cfg,
                cfg.order_seed + int(1.0 / lam) + 211 * METHODS.index(method),
            )

            err_u = abs(out["mean_u"] - ref["mean_u"])
            err_x2 = abs(out["mean_x2"] - ref["mean_x2"])

            errors_u.append(max(err_u, 1.0e-16))
            errors_x2.append(max(err_x2, 1.0e-16))

            rows.append({
                "method": method,
                "lambda": lam,
                "mean_u": out["mean_u"],
                "mean_x2": out["mean_x2"],
                "err_mean_u": err_u,
                "err_mean_x2": err_x2,
            })

        slope_u = float(np.polyfit(np.log(cfg.order_lambdas), np.log(errors_u), 1)[0])
        slope_x2 = float(np.polyfit(np.log(cfg.order_lambdas), np.log(errors_x2), 1)[0])

        print(
            f"  {LABELS[method]} fitted slopes: "
            f"E[u] error ≈ {slope_u:.2f}, "
            f"E[X^2] error ≈ {slope_x2:.2f}"
        )

    return rows, refs


# ============================================================
# Runners
# ============================================================
def run_main(cfg: Config):
    results = {method: [] for method in METHODS}

    print("\nMain per-iteration study:")

    for method in METHODS:
        for seed in cfg.seeds:
            results[method].append(
                train_one(
                    method=method,
                    seed=seed,
                    cfg=cfg,
                    lr=cfg.lr,
                    eta=cfg.eta,
                    init_scale=cfg.init_scale,
                    n_epochs=cfg.n_epochs,
                )
            )

        ndiv = sum(1 for r in results[method] if r["diverged_at"] is not None)
        print(f"  {LABELS[method]:8s}: diverged {ndiv}/{len(cfg.seeds)}")

    return results


def run_lr_sweep(cfg: Config):
    sweep = {method: {lr: [] for lr in cfg.sweep_lrs} for method in METHODS}

    print("\nLearning-rate robustness sweep:")

    for lr in cfg.sweep_lrs:
        for method in METHODS:
            for seed in cfg.seeds:
                sweep[method][lr].append(
                    train_one(
                        method=method,
                        seed=seed,
                        cfg=cfg,
                        lr=lr,
                        eta=cfg.eta,
                        init_scale=cfg.init_scale,
                        n_epochs=cfg.n_epochs,
                    )
                )

        print(f"  lr={lr:g} done")

    return sweep


def run_regime_map(cfg: Config):
    maps = {
        method: np.zeros((len(cfg.regime_etas), len(cfg.regime_init_scales)))
        for method in METHODS
    }
    survival = {
        method: np.zeros((len(cfg.regime_etas), len(cfg.regime_init_scales)))
        for method in METHODS
    }

    print("\nRegime map over (eta, initialization scale):")

    for i, eta in enumerate(cfg.regime_etas):
        for j, init_scale in enumerate(cfg.regime_init_scales):
            for method in METHODS:
                finals = []
                survived = 0

                for seed in cfg.regime_seeds:
                    out = train_one(
                        method=method,
                        seed=seed,
                        cfg=cfg,
                        lr=cfg.lr,
                        eta=eta,
                        init_scale=init_scale,
                        n_epochs=cfg.regime_epochs,
                    )

                    finals.append(float(out["final_obj"]))
                    survived += int(out["diverged_at"] is None)

                maps[method][i, j] = float(np.median(finals))
                survival[method][i, j] = survived / len(cfg.regime_seeds)

        print(f"  eta={eta:g} row done")

    return maps, survival


def stack_metric(results, method: str, key: str) -> np.ndarray:
    return np.vstack([np.asarray(r[key]) for r in results[method]])


# ============================================================
# Figure helpers
# ============================================================
def savefig(fig: Figure, cfg: Config, name: str) -> None:
    path = os.path.join(cfg.out_dir, name)
    fig.savefig(path)
    plt.close(fig)


def overflow_line(ax, cfg: Config, label: str = "numerical overflow") -> None:
    ax.axhline(cfg.explode_cap, ls="--", lw=1.2, color="0.40", zorder=1)
    ax.text(
        0.012,
        cfg.explode_cap,
        f" {label}",
        va="bottom",
        ha="left",
        transform=ax.get_yaxis_transform(),
        fontsize=9.0,
        color="0.35",
    )


def plot_metric_over_iterations(
    results,
    cfg: Config,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    ylog: bool = True,
    reference_value: Optional[float] = None,
    reference_label: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.3))

    first_mat = stack_metric(results, METHODS[0], metric)
    iterations = np.arange(1, first_mat.shape[1] + 1)

    for method in METHODS:
        mat = stack_metric(results, method, metric)

        med = np.median(mat, axis=0)
        lo = np.percentile(mat, 25, axis=0)
        hi = np.percentile(mat, 75, axis=0)

        ax.plot(iterations, med, color=COLORS[method], label=LABELS[method])
        ax.fill_between(iterations, lo, hi, color=COLORS[method], alpha=0.10, linewidth=0)

    if reference_value is not None:
        ax.axhline(
            reference_value,
            color="0.25",
            linestyle=":",
            linewidth=1.6,
            label=reference_label,
            zorder=2,
        )

    if ylog:
        ax.set_yscale("log")

    overflow_line(ax, cfg)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, iterations[-1])
    ax.legend(frameon=False, ncol=2, loc="best")

    savefig(fig, cfg, filename)


def fig_objective(results, cfg: Config):
    plot_metric_over_iterations(
        results,
        cfg,
        metric="obj",
        ylabel=r"Empirical objective $u_n(\theta_k)$",
        title=(
            r"Empirical objective under super-linear drift "
            rf"($\eta={cfg.eta:g}$, init scale $={cfg.init_scale:g}$, lr $={cfg.lr:g}$)"
        ),
        filename="fig1_empirical_objective_all_methods.png",
        ylog=True,
    )


def fig_test_mse(results, cfg: Config, null_test_mse: float):
    plot_metric_over_iterations(
        results,
        cfg,
        metric="test",
        ylabel="Test MSE",
        title=(
            r"Test error under super-linear drift "
            rf"($\eta={cfg.eta:g}$, init scale $={cfg.init_scale:g}$, lr $={cfg.lr:g}$)"
        ),
        filename="fig2_test_mse_all_methods.png",
        ylog=True,
        reference_value=null_test_mse,
        reference_label=rf"Null predictor ({null_test_mse:.3f})",
    )


def fig_parameter_norm(results, cfg: Config):
    fig, ax = plt.subplots(figsize=(9.0, 5.3))

    first_mat = stack_metric(results, METHODS[0], "norm")
    iterations = np.arange(1, first_mat.shape[1] + 1)

    for method in METHODS:
        mat = stack_metric(results, method, "norm")

        med = np.median(mat, axis=0)
        lo = np.percentile(mat, 25, axis=0)
        hi = np.percentile(mat, 75, axis=0)

        ax.plot(iterations, med, color=COLORS[method], label=LABELS[method])
        ax.fill_between(iterations, lo, hi, color=COLORS[method], alpha=0.10, linewidth=0)

    ax.axhline(cfg.theta_cap, ls="--", lw=1.2, color="0.40")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Parameter norm $\|\theta_k\|$")
    ax.set_title("Parameter growth: all methods on the same stress test")
    ax.set_xlim(1, iterations[-1])
    ax.legend(frameon=False, ncol=2, loc="best")

    savefig(fig, cfg, "fig3_parameter_norm_all_methods.png")


def fig_lr_robustness(sweep, cfg: Config):
    fig, ax = plt.subplots(figsize=(9.0, 5.3))

    for method in METHODS:
        meds, lows, highs, divs = [], [], [], []

        for lr in cfg.sweep_lrs:
            vals = np.asarray([r["final_obj"] for r in sweep[method][lr]], dtype=float)

            meds.append(np.median(vals))
            lows.append(np.percentile(vals, 25))
            highs.append(np.percentile(vals, 75))
            divs.append(sum(1 for r in sweep[method][lr] if r["diverged_at"] is not None))

        meds_arr = np.asarray(meds)

        ax.plot(
            cfg.sweep_lrs,
            meds_arr,
            "-o",
            color=COLORS[method],
            label=LABELS[method],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.fill_between(cfg.sweep_lrs, lows, highs, color=COLORS[method], alpha=0.10, linewidth=0)

        for lr, med, nd in zip(cfg.sweep_lrs, meds_arr, divs):
            if nd > 0:
                ax.annotate(
                    f"{nd}/{len(cfg.seeds)} div.",
                    (lr, med),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8.0,
                    color=COLORS[method],
                )

    ax.set_yscale("log")
    overflow_line(ax, cfg)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel(r"Final empirical objective")
    ax.set_title("Learning-rate robustness: all methods on the same objective")
    ax.set_xticks(list(cfg.sweep_lrs))
    ax.legend(frameon=False, ncol=2, loc="best")

    savefig(fig, cfg, "fig4_learning_rate_robustness_all_methods.png")


def fig_taming_activation(results, cfg: Config):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    first_mat = stack_metric(results, TAMED_METHODS[0], "active")
    iterations = np.arange(1, first_mat.shape[1] + 1)

    for method in TAMED_METHODS:
        mat = stack_metric(results, method, "active")
        med = np.mean(mat, axis=0) * 100.0

        if len(med) > 11:
            med = np.convolve(med, np.ones(11) / 11, mode="same")

        ax.plot(iterations, med, color=COLORS[method], label=LABELS[method])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Updates with taming active (%)")
    ax.set_title("Adaptive taming activates during the super-linear transient")
    ax.set_xlim(1, iterations[-1])
    ax.set_ylim(-2, 102)
    ax.legend(frameon=False, ncol=1, loc="best")

    savefig(fig, cfg, "fig5_taming_active_fraction.png")


def fig_regime_map(regime_maps, survival_maps, cfg: Config):
    n_methods = len(METHODS)
    ncols = 3
    nrows = int(np.ceil(n_methods / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12.8, 6.8), constrained_layout=True)
    axes = axes.ravel()

    all_vals = np.concatenate([
        np.ravel(np.clip(regime_maps[m], 1.0e-6, cfg.explode_cap))
        for m in METHODS
    ])

    positive = all_vals[all_vals > 0]

    if len(positive):
        vmin = float(max(1.0e-4, float(np.percentile(positive, 5))))
    else:
        vmin = 1.0e-4

    vmax = float(cfg.explode_cap)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = None

    for ax, method in zip(axes, METHODS):
        dat = np.clip(regime_maps[method], vmin, vmax)

        im = ax.imshow(
            dat,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn_r",
            norm=norm,
            extent=[
                cfg.regime_init_scales[0],
                cfg.regime_init_scales[-1],
                cfg.regime_etas[0],
                cfg.regime_etas[-1],
            ],
        )

        surv = survival_maps[method]

        for i, eta in enumerate(cfg.regime_etas):
            for j, init_scale in enumerate(cfg.regime_init_scales):
                if surv[i, j] < 1.0:
                    ax.text(
                        init_scale,
                        eta,
                        f"{surv[i, j]:.0%}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8.2,
                        fontweight="bold",
                    )

        ax.set_title(LABELS[method])
        ax.set_xlabel("Init. scale")
        ax.set_ylabel(r"$\eta$")

    for ax in axes[n_methods:]:
        ax.axis("off")

    if im is not None:
        cb = fig.colorbar(im, ax=axes[:n_methods], shrink=0.92, pad=0.015)
        cb.set_label("Median final objective; cells marked by survival rate if <100%")

    fig.suptitle(
        "Robustness regime map: all algorithms on the same super-linear stress grid",
        fontsize=14,
    )

    savefig(fig, cfg, "fig6_regime_map_all_methods.png")


def fig_observed_order(order_rows, cfg: Config):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for method in TAMED_METHODS:
        rows = [r for r in order_rows if r["method"] == method]
        lam = np.asarray([r["lambda"] for r in rows])
        err = np.asarray([max(r["err_mean_u"], 1.0e-16) for r in rows])
        slope = float(np.polyfit(np.log(lam), np.log(err), 1)[0])

        ax.loglog(
            lam,
            err,
            "-o",
            color=COLORS[method],
            label=f"{LABELS[method]} (slope {slope:.2f})",
        )

    ax.invert_xaxis()
    ax.set_xlabel(r"Stepsize $\lambda$")
    ax.set_ylabel(r"Weak error in $\mathbb{E}[u(X_T)]$")
    ax.set_title("Observed stepsize scaling on a one-dimensional super-linear SDE")
    ax.legend(frameon=False, loc="best")

    savefig(fig, cfg, "fig7_observed_order_mean_potential.png")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for method in TAMED_METHODS:
        rows = [r for r in order_rows if r["method"] == method]
        lam = np.asarray([r["lambda"] for r in rows])
        err = np.asarray([max(r["err_mean_x2"], 1.0e-16) for r in rows])
        slope = float(np.polyfit(np.log(lam), np.log(err), 1)[0])

        ax.loglog(
            lam,
            err,
            "-o",
            color=COLORS[method],
            label=f"{LABELS[method]} (slope {slope:.2f})",
        )

    ax.invert_xaxis()
    ax.set_xlabel(r"Stepsize $\lambda$")
    ax.set_ylabel(r"Weak error in $\mathbb{E}[X_T^2]$")
    ax.set_title("Observed stepsize scaling in a second moment")
    ax.legend(frameon=False, loc="best")

    savefig(fig, cfg, "fig8_observed_order_second_moment.png")


# ============================================================
# CSV outputs
# ============================================================
def write_summary_csv(
    results,
    sweep,
    regime_maps,
    survival_maps,
    order_rows,
    order_refs,
    cfg: Config,
):
    path = os.path.join(cfg.out_dir, "summary_main.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("method,median_final_obj,median_final_test,median_final_norm,diverged,total\n")

        for method in METHODS:
            final_obj = np.asarray([r["final_obj"] for r in results[method]], dtype=float)
            final_test = np.asarray([r["final_test"] for r in results[method]], dtype=float)
            final_norm = np.asarray([r["final_norm"] for r in results[method]], dtype=float)
            div = sum(1 for r in results[method] if r["diverged_at"] is not None)

            f.write(
                f"{method},"
                f"{np.median(final_obj):.10e},"
                f"{np.median(final_test):.10e},"
                f"{np.median(final_norm):.10e},"
                f"{div},"
                f"{len(cfg.seeds)}\n"
            )

    path = os.path.join(cfg.out_dir, "summary_lr_sweep.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("method,lr,median_final_obj,median_final_test,diverged,total\n")

        for method in METHODS:
            for lr in cfg.sweep_lrs:
                final_obj = np.asarray([r["final_obj"] for r in sweep[method][lr]], dtype=float)
                final_test = np.asarray([r["final_test"] for r in sweep[method][lr]], dtype=float)
                div = sum(1 for r in sweep[method][lr] if r["diverged_at"] is not None)

                f.write(
                    f"{method},"
                    f"{lr:.10g},"
                    f"{np.median(final_obj):.10e},"
                    f"{np.median(final_test):.10e},"
                    f"{div},"
                    f"{len(cfg.seeds)}\n"
                )

    path = os.path.join(cfg.out_dir, "summary_regime_map.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("method,eta,init_scale,median_final_obj,survival_rate\n")

        for method in METHODS:
            for i, eta in enumerate(cfg.regime_etas):
                for j, init_scale in enumerate(cfg.regime_init_scales):
                    f.write(
                        f"{method},"
                        f"{eta:.10g},"
                        f"{init_scale:.10g},"
                        f"{regime_maps[method][i, j]:.10e},"
                        f"{survival_maps[method][i, j]:.10e}\n"
                    )

    path = os.path.join(cfg.out_dir, "summary_observed_order.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("method,lambda,mean_u,mean_x2,err_mean_u,err_mean_x2\n")

        for r in order_rows:
            f.write(
                f"{r['method']},"
                f"{r['lambda']:.10e},"
                f"{r['mean_u']:.10e},"
                f"{r['mean_x2']:.10e},"
                f"{r['err_mean_u']:.10e},"
                f"{r['err_mean_x2']:.10e}\n"
            )

    path = os.path.join(cfg.out_dir, "reference_observed_order.csv")

    with open(path, "w", encoding="utf-8") as f:
        f.write("method,lambda_ref,mean_u,mean_x2\n")

        for method, ref in order_refs.items():
            f.write(
                f"{method},"
                f"{cfg.order_lambda_ref:.10e},"
                f"{ref['mean_u']:.10e},"
                f"{ref['mean_x2']:.10e}\n"
            )


# ============================================================
# Console summaries
# ============================================================
def print_summaries(results, sweep, cfg: Config):
    print("\nMain run -- final objective/test/norm and divergence:")

    for method in METHODS:
        fobj = np.asarray([r["final_obj"] for r in results[method]], dtype=float)
        ftst = np.asarray([r["final_test"] for r in results[method]], dtype=float)
        fnrm = np.asarray([r["final_norm"] for r in results[method]], dtype=float)
        div = sum(1 for r in results[method] if r["diverged_at"] is not None)

        print(
            f"  {LABELS[method]:8s}  "
            f"obj={np.median(fobj):10.3e}  "
            f"test={np.median(ftst):10.3e}  "
            f"norm={np.median(fnrm):10.3e}  "
            f"div={div}/{len(cfg.seeds)}"
        )

    print("\nLearning-rate sweep -- median final objective; divergence in parentheses:")

    header = "  " + " " * 8 + "  ".join([f"lr={lr:<8g}" for lr in cfg.sweep_lrs])
    print(header)

    for method in METHODS:
        cells = []

        for lr in cfg.sweep_lrs:
            fobj = np.asarray([r["final_obj"] for r in sweep[method][lr]], dtype=float)
            div = sum(1 for r in sweep[method][lr] if r["diverged_at"] is not None)

            cell = f"{np.median(fobj):.2e}"

            if div:
                cell += f"(d{div})"

            cells.append(f"{cell:>12s}")

        print(f"  {LABELS[method]:8s}" + "  ".join(cells))


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Super-linear drift stress test for tamed Langevin optimizers."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures and CSV files.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller quick test.",
    )
    parser.add_argument(
        "--skip-order",
        action="store_true",
        help="Skip the one-dimensional observed-order experiment.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override main learning rate.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Override regularization eta.",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=None,
        help="Override initialization scale.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()

    if args.quick:
        cfg = cfg.quick()

    if args.out_dir is not None:
        cfg = replace(cfg, out_dir=args.out_dir)

    if args.lr is not None:
        cfg = replace(cfg, lr=args.lr)

    if args.eta is not None:
        cfg = replace(cfg, eta=args.eta)

    if args.init_scale is not None:
        cfg = replace(cfg, init_scale=args.init_scale)

    os.makedirs(cfg.out_dir, exist_ok=True)
    check_g_c1()

    print("Configuration")
    print("-------------")
    print(f"parameter dimension d = {parameter_dimension(cfg)}")
    print(
        f"n_train={cfg.n_train}, "
        f"n_test={cfg.n_test}, "
        f"batch={cfg.batch}, "
        f"epochs={cfg.n_epochs}"
    )
    print(f"seeds={cfg.seeds}")
    print(f"main lr={cfg.lr}, eta={cfg.eta}, init_scale={cfg.init_scale}")
    print(f"methods={', '.join(LABELS[m] for m in METHODS)}")
    print(f"output directory: {cfg.out_dir}")
    
    null = compute_null_baselines(cfg)

    print("\nNull baseline")
    print("-------------")
    print(f"median constant prediction = {null['median_constant']:.6e}")
    print(f"median null train MSE      = {null['median_train_mse']:.6e}")
    print(f"median null test MSE       = {null['median_test_mse']:.6e}")
    t0 = time.time()

    results = run_main(cfg)
    sweep = run_lr_sweep(cfg)
    regime_maps, survival_maps = run_regime_map(cfg)

    if args.skip_order:
        order_rows, order_refs = [], {}
        print("\nObserved-order experiment skipped.")
    else:
        order_rows, order_refs = run_observed_order(cfg)

    print("\nGenerating figures...")

    fig_objective(results, cfg)
    fig_test_mse(results, cfg, null["median_test_mse"])
    fig_parameter_norm(results, cfg)
    fig_lr_robustness(sweep, cfg)
    fig_taming_activation(results, cfg)
    fig_regime_map(regime_maps, survival_maps, cfg)

    if order_rows:
        fig_observed_order(order_rows, cfg)

    write_summary_csv(
        results,
        sweep,
        regime_maps,
        survival_maps,
        order_rows,
        order_refs,
        cfg,
    )
    print_summaries(results, sweep, cfg)

    print(f"\nTotal compute time: {time.time() - t0:.1f}s")
    print(f"Saved figures and CSV summaries to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
