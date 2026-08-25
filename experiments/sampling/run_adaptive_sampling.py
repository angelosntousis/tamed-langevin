#!/usr/bin/env python3
"""
Aggressive-regime sampling experiment for adaptive-tamed Langevin schemes.

This script compares:
    ULA,
    kTULA without the adaptive switch,
    adaptive kTULA,
    adaptive tRLMC,

on the high-dimensional double-well target

    U(x) = (x_1^2 - 1)^2 / 4 + kappa * sum_{i=2}^d x_i^2 / 2,
    h(x) = (x_1^3 - x_1, kappa*x_2, ..., kappa*x_d).

The reusable kTULA/tRLMC implementations are imported from src/tamed_langevin.
This file keeps only experiment-specific code: ULA baseline, target density,
plotting, statistics, and experiment orchestration.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tamed_langevin import KTULASampler, StandardKTULASampler, TRLMCSampler
from tamed_langevin.taming import check_g_c1


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class Config:
    beta: float = 1.0
    dim: int = 100
    kappa: float = 100.0

    lambdas: Tuple[float, ...] = (0.002, 0.001, 0.0005)

    n_steps: int = 1_000_000
    burn_in: int = 200_000
    n_reps: int = 30

    seed: int = 1234
    illustrative_seed_offset: int = 999

    a_tame: float = 0.05
    ell_tame: float = 2.0

    density_xmin: float = -5.0
    density_xmax: float = 5.0
    density_grid_n: int = 3000
    hist_alpha: float = 0.35

    out_dir: str = "./figures/Sampling_Experiments_Adaptive_rLMC"

    def quick(self) -> "Config":
        return replace(
            self,
            lambdas=(0.002, 0.001),
            n_steps=20_000,
            burn_in=10_000,
            n_reps=3,
            density_grid_n=1000,
        )


METHODS = ("ULA", "kTULA", "adaptive kTULA", "tRLMC")
TAMED_METHODS = METHODS[1:]

# Keep the internal method keys stable for saved data and sampler dispatch, while
# using the preferred algorithm names everywhere text is rendered in a figure.
LABELS = {
    "ULA": "ULA",
    "kTULA": "kTULA",
    "adaptive kTULA": "adTULA",
    "tRLMC": "adTRLMC",
}


# ============================================================
# Potential and drift
# ============================================================
def potential(x: np.ndarray, kappa: float = 100.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.25 * (x[..., 0] ** 2 - 1.0) ** 2 + 0.5 * kappa * np.sum(
        x[..., 1:] ** 2, axis=-1
    )


def drift(x: np.ndarray, kappa: float = 100.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    out[0] = x[0] ** 3 - x[0]
    out[1:] = kappa * x[1:]
    return out


def make_drift(cfg: Config):
    def target_drift(x: np.ndarray) -> np.ndarray:
        return drift(x, cfg.kappa)

    return target_drift


# ============================================================
# Target density and second moment
# ============================================================
def integrate(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def true_density(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Evaluate the normalized double-well marginal of the first coordinate."""
    x = np.asarray(x, dtype=float)
    logp = -cfg.beta * 0.25 * (x**2 - 1.0) ** 2
    logp -= np.max(logp)
    density = np.exp(logp)
    return density / integrate(density, x)


def true_second_moment(cfg: Config) -> float:
    x = np.linspace(cfg.density_xmin, cfg.density_xmax, 200_000)
    density = true_density(x, cfg)
    return integrate(x**2 * density, x)


# ============================================================
# Samplers
# ============================================================
def initial_state(cfg: Config) -> np.ndarray:
    x = np.zeros(cfg.dim)
    x[0] = 200.0
    return x


def make_sampler(method: str, lam: float, cfg: Config):
    target_drift = make_drift(cfg)

    if method == "kTULA":
        return StandardKTULASampler(
            drift=target_drift,
            step_size=lam,
            beta=cfg.beta,
            a_tame=cfg.a_tame,
            ell_tame=cfg.ell_tame,
        )

    if method == "adaptive kTULA":
        return KTULASampler(
            drift=target_drift,
            step_size=lam,
            beta=cfg.beta,
            a_tame=cfg.a_tame,
            ell_tame=cfg.ell_tame,
        )

    if method == "tRLMC":
        return TRLMCSampler(
            drift=target_drift,
            step_size=lam,
            beta=cfg.beta,
            a_tame=cfg.a_tame,
            ell_tame=cfg.ell_tame,
        )

    raise ValueError(f"Unknown sampler method: {method}")


def step_ula(x: np.ndarray, lam: float, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    noise = rng.normal(size=x.shape)
    return x - lam * drift(x, cfg.kappa) + np.sqrt(2.0 * lam / cfg.beta) * noise


# ============================================================
# Chain runners
# ============================================================
def run_chain(
    method: str,
    lam: float,
    seed: int,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    rng = np.random.default_rng(seed)
    x = initial_state(cfg)

    trajectory = []
    norm2 = []

    if method == "ULA":
        for step in range(cfg.n_steps):
            with np.errstate(over="ignore", invalid="ignore"):
                x = step_ula(x, lam, rng, cfg)

            if not np.isfinite(x).all():
                return None, None, step

            norm2.append(float(np.sum(x**2)))

            if step >= cfg.burn_in:
                trajectory.append(float(x[0]))

        return np.asarray(trajectory), np.asarray(norm2), None

    sampler = make_sampler(method, lam, cfg)

    for step in range(cfg.n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            x, _ = sampler.step(x, rng)

        if not np.isfinite(x).all():
            return None, None, step

        norm2.append(float(np.sum(x**2)))

        if step >= cfg.burn_in:
            trajectory.append(float(x[0]))

    return np.asarray(trajectory), np.asarray(norm2), None


def explosion_time_ula(lam: float, seed: int, cfg: Config) -> int:
    rng = np.random.default_rng(seed)
    x = initial_state(cfg)

    for step in range(cfg.n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            x = step_ula(x, lam, rng, cfg)

        if not np.isfinite(x).all():
            return step

    return cfg.n_steps + 1


def moment_errors(
    method: str,
    lam: float,
    seed: int,
    true_first_m2: float,
    true_last_m2: float,
    cfg: Config,
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    x = initial_state(cfg)
    sampler = make_sampler(method, lam, cfg)

    first_moment_sum = 0.0
    last_moment_sum = 0.0
    sample_count = 0

    for step in range(cfg.n_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            x, _ = sampler.step(x, rng)

        if not np.isfinite(x).all():
            nan = float("nan")
            return nan, nan, nan, nan

        if step >= cfg.burn_in:
            first_moment_sum += float(x[0] ** 2)
            last_moment_sum += float(x[-1] ** 2)
            sample_count += 1

    empirical_first_m2 = first_moment_sum / sample_count
    empirical_last_m2 = last_moment_sum / sample_count
    first_bias = float(empirical_first_m2 - true_first_m2)
    last_bias = float(empirical_last_m2 - true_last_m2)
    return abs(first_bias), abs(last_bias), first_bias, last_bias


# ============================================================
# Plot helpers
# ============================================================
def save_moment_growth_plot(
    lam: float,
    norm2_by_method: Dict[str, Optional[np.ndarray]],
    cfg: Config,
) -> None:
    plt.figure()

    for method in METHODS:
        norm2 = norm2_by_method[method]
        if norm2 is not None:
            plt.plot(norm2, label=LABELS[method])

    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel(r"$\|X_n\|^2$")
    plt.title(f"Second-moment growth, d={cfg.dim}, λ={lam}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, f"moment_d{cfg.dim}_lam_{lam}.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_trajectory_plot(
    trajectory: np.ndarray,
    lam: float,
    name: str,
    cfg: Config,
) -> None:
    file_name = name.lower().replace(" ", "_")

    plt.figure()
    plt.plot(trajectory[:5000])
    plt.xlabel("post burn-in iteration")
    plt.ylabel(r"$X_n^{(1)}$")
    plt.title(f"{LABELS[name]} trajectory, λ={lam}")
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, f"traj_{file_name}_d{cfg.dim}_lam_{lam}.png")
    plt.savefig(path, dpi=300)
    plt.close()


def save_density_plot(
    trajectory: np.ndarray,
    lam: float,
    name: str,
    target_grid: np.ndarray,
    target_density: np.ndarray,
    cfg: Config,
) -> None:
    file_name = name.lower().replace(" ", "_")
    plt.figure()
    plt.hist(
        trajectory,
        bins=120,
        range=(cfg.density_xmin, cfg.density_xmax),
        density=True,
        alpha=cfg.hist_alpha,
        label=LABELS[name],
    )
    plt.plot(target_grid, target_density, lw=2, label="Target density")
    plt.xlim(cfg.density_xmin, cfg.density_xmax)
    plt.xlabel("x, first coordinate")
    plt.ylabel("density")
    plt.title(f"Empirical density, d={cfg.dim}, λ={lam}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, f"density_{file_name}_d{cfg.dim}_lam_{lam}.png")
    plt.savefig(path, dpi=300)
    plt.close()


def boxplot_by_lambda(
    data_by_lambda: Dict[float, List[float]],
    title: str,
    file_name: str,
    ylabel: str,
    cfg: Config,
    logy: bool = False,
) -> None:
    data = [data_by_lambda[lam] for lam in cfg.lambdas]

    plt.figure(figsize=(6.0, 4.0))
    plt.boxplot(data)
    plt.xticks(
        range(1, len(cfg.lambdas) + 1),
        [f"λ={lam}" for lam in cfg.lambdas],
    )

    if logy:
        plt.yscale("log")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    path = os.path.join(cfg.out_dir, file_name)
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# Experiment
# ============================================================
def run_experiment(cfg: Config):
    check_g_c1()
    os.makedirs(cfg.out_dir, exist_ok=True)

    true_first_m2 = true_second_moment(cfg)
    true_last_m2 = 1.0 / (cfg.beta * cfg.kappa)
    target_grid = np.linspace(
        cfg.density_xmin, cfg.density_xmax, cfg.density_grid_n
    )
    target_pdf = true_density(target_grid, cfg)

    ula_explosion_times: Dict[float, List[float]] = {}
    second_moment_errors: Dict[str, Dict[float, List[float]]] = {
        method: {} for method in TAMED_METHODS
    }
    last_second_moment_errors: Dict[str, Dict[float, List[float]]] = {
        method: {} for method in TAMED_METHODS
    }
    signed_biases: Dict[str, Dict[float, List[float]]] = {
        method: {} for method in TAMED_METHODS
    }
    last_signed_biases: Dict[str, Dict[float, List[float]]] = {
        method: {} for method in TAMED_METHODS
    }

    illustrative_seed = cfg.seed + cfg.illustrative_seed_offset

    print("Adaptive sampling experiment")
    print("----------------------------")
    print(f"dimension d = {cfg.dim}")
    print(f"transverse confinement kappa = {cfg.kappa}")
    print(f"beta = {cfg.beta}")
    print(f"a_tame = {cfg.a_tame}")
    print(f"ell_tame = {cfg.ell_tame}")
    print(f"n_steps = {cfg.n_steps}")
    print(f"burn_in = {cfg.burn_in}")
    print(f"n_reps = {cfg.n_reps}")
    print(f"true E[X_1^2] = {true_first_m2:.8f}")
    print(f"true E[X_{cfg.dim}^2] = {true_last_m2:.8f}")
    print(f"output directory: {cfg.out_dir}")

    for lam_index, lam in enumerate(cfg.lambdas):
        print(f"\nRunning λ={lam}")

        illustrative = {
            method: run_chain(method, lam, illustrative_seed, cfg)
            for method in METHODS
        }
        save_moment_growth_plot(
            lam,
            {method: illustrative[method][1] for method in METHODS},
            cfg,
        )

        for method in TAMED_METHODS:
            trajectory = illustrative[method][0]
            if trajectory is not None:
                save_trajectory_plot(trajectory, lam, method, cfg)
                save_density_plot(
                    trajectory, lam, method, target_grid, target_pdf, cfg
                )

        ula_times = []
        errors = {method: [] for method in TAMED_METHODS}
        last_errors = {method: [] for method in TAMED_METHODS}
        biases = {method: [] for method in TAMED_METHODS}
        last_biases = {method: [] for method in TAMED_METHODS}

        for rep in range(cfg.n_reps):
            rep_seed = cfg.seed + 10_000 * lam_index + rep

            ula_times.append(float(explosion_time_ula(lam, rep_seed, cfg)))
            for method in TAMED_METHODS:
                first_error, last_error, first_bias, last_bias = moment_errors(
                    method,
                    lam,
                    rep_seed,
                    true_first_m2,
                    true_last_m2,
                    cfg,
                )
                errors[method].append(first_error)
                last_errors[method].append(last_error)
                biases[method].append(first_bias)
                last_biases[method].append(last_bias)

        ula_explosion_times[lam] = ula_times
        for method in TAMED_METHODS:
            second_moment_errors[method][lam] = errors[method]
            last_second_moment_errors[method][lam] = last_errors[method]
            signed_biases[method][lam] = biases[method]
            last_signed_biases[method][lam] = last_biases[method]

        n_exploded = sum(t < cfg.n_steps for t in ula_times)
        print(f"  ULA exploded in {n_exploded}/{cfg.n_reps} runs")

    return (
        ula_explosion_times,
        second_moment_errors,
        last_second_moment_errors,
        signed_biases,
        last_signed_biases,
    )


# ============================================================
# Summary
# ============================================================
def mean_std(values: List[float]) -> Tuple[float, float]:
    xs = np.asarray(values, dtype=float)
    return float(np.nanmean(xs)), float(np.nanstd(xs, ddof=1))


def paired_difference_summary(
    left: List[float],
    right: List[float],
) -> Tuple[float, float, float, float]:
    """Summarize matched-seed differences left - right."""
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    differences = differences[np.isfinite(differences)]
    mean = float(np.mean(differences))
    std = float(np.std(differences, ddof=1))
    half_width = 1.96 * std / np.sqrt(len(differences))
    return mean, std, mean - half_width, mean + half_width


def print_summary(
    ula_explosion_times: Dict[float, List[float]],
    second_moment_errors: Dict[str, Dict[float, List[float]]],
    last_second_moment_errors: Dict[str, Dict[float, List[float]]],
    signed_biases: Dict[str, Dict[float, List[float]]],
    last_signed_biases: Dict[str, Dict[float, List[float]]],
    cfg: Config,
) -> None:
    print("\nSummary table values:\n")

    for lam in cfg.lambdas:
        mean, std = mean_std(ula_explosion_times[lam])
        print(f"ULA explosion time, lambda={lam}: {mean:.2f} ± {std:.2f}")

    for method in TAMED_METHODS:
        for lam in cfg.lambdas:
            mean, std = mean_std(second_moment_errors[method][lam])
            print(
                f"{method} first-coordinate second-moment error, "
                f"lambda={lam}: {mean:.6f} ± {std:.6f}"
            )

    print("\nSigned second-moment biases (positive means overestimation):\n")

    for method in TAMED_METHODS:
        for lam in cfg.lambdas:
            mean, std = mean_std(signed_biases[method][lam])
            print(
                f"{method} first-coordinate signed bias, "
                f"lambda={lam}: {mean:.6f} ± {std:.6f}"
            )

    for method in TAMED_METHODS:
        for lam in cfg.lambdas:
            mean, std = mean_std(last_signed_biases[method][lam])
            print(
                f"{method} last-coordinate signed bias, "
                f"lambda={lam}: {mean:.6f} ± {std:.6f}"
            )

    print("\nMatched-seed absolute-error differences (negative favors first method):\n")
    comparisons = (
        ("adaptive kTULA", "kTULA"),
        ("tRLMC", "kTULA"),
        ("tRLMC", "adaptive kTULA"),
    )

    for coordinate, errors in (
        ("first", second_moment_errors),
        ("last", last_second_moment_errors),
    ):
        for left, right in comparisons:
            for lam in cfg.lambdas:
                mean, std, ci_low, ci_high = paired_difference_summary(
                    errors[left][lam], errors[right][lam]
                )
                print(
                    f"{coordinate} coordinate, {left} - {right}, lambda={lam}: "
                    f"mean={mean:.6f}, sd={std:.6f}, "
                    f"95% CI=[{ci_low:.6f}, {ci_high:.6f}]"
                )


def write_sampling_csvs(
    second_moment_errors: Dict[str, Dict[float, List[float]]],
    last_second_moment_errors: Dict[str, Dict[float, List[float]]],
    signed_biases: Dict[str, Dict[float, List[float]]],
    last_signed_biases: Dict[str, Dict[float, List[float]]],
    cfg: Config,
) -> None:
    summary_path = os.path.join(cfg.out_dir, "summary_sampling_moments.csv")
    with open(summary_path, "w", encoding="utf-8") as output:
        output.write(
            "method,lambda,coordinate,mean_absolute_error,sd_absolute_error,"
            "mean_signed_bias,sd_signed_bias\n"
        )
        for coordinate, errors, biases in (
            ("first", second_moment_errors, signed_biases),
            ("last", last_second_moment_errors, last_signed_biases),
        ):
            for method in TAMED_METHODS:
                for lam in cfg.lambdas:
                    mean_abs, sd_abs = mean_std(errors[method][lam])
                    mean_bias, sd_bias = mean_std(biases[method][lam])
                    output.write(
                        f"{method},{lam:.10g},{coordinate},{mean_abs:.10e},"
                        f"{sd_abs:.10e},{mean_bias:.10e},{sd_bias:.10e}\n"
                    )

    paired_path = os.path.join(cfg.out_dir, "summary_paired_differences.csv")
    comparisons = (
        ("adaptive kTULA", "kTULA"),
        ("tRLMC", "kTULA"),
        ("tRLMC", "adaptive kTULA"),
    )
    with open(paired_path, "w", encoding="utf-8") as output:
        output.write(
            "coordinate,left_method,right_method,lambda,mean_difference,"
            "sd_difference,ci95_low,ci95_high\n"
        )
        for coordinate, errors in (
            ("first", second_moment_errors),
            ("last", last_second_moment_errors),
        ):
            for left, right in comparisons:
                for lam in cfg.lambdas:
                    mean, std, ci_low, ci_high = paired_difference_summary(
                        errors[left][lam], errors[right][lam]
                    )
                    output.write(
                        f"{coordinate},{left},{right},{lam:.10g},{mean:.10e},"
                        f"{std:.10e},{ci_low:.10e},{ci_high:.10e}\n"
                    )

    for method in TAMED_METHODS:
        for lam in cfg.lambdas:
            mean, std = mean_std(last_second_moment_errors[method][lam])
            print(
                f"{method} last-coordinate second-moment error, "
                f"lambda={lam}: {mean:.6f} ± {std:.6f}"
            )


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggressive-regime sampling experiment for adaptive-tamed Langevin schemes."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller quick test.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override number of chain steps.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=None,
        help="Override burn-in.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=None,
        help="Override number of repetitions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config()

    if args.quick:
        cfg = cfg.quick()

    if args.out_dir is not None:
        cfg = replace(cfg, out_dir=args.out_dir)

    if args.n_steps is not None:
        cfg = replace(cfg, n_steps=args.n_steps)

    if args.burn_in is not None:
        cfg = replace(cfg, burn_in=args.burn_in)

    if args.n_reps is not None:
        cfg = replace(cfg, n_reps=args.n_reps)

    if cfg.burn_in >= cfg.n_steps:
        raise ValueError("burn_in must be smaller than n_steps.")

    (
        ula_explosion_times,
        second_moment_errors,
        last_second_moment_errors,
        signed_biases,
        last_signed_biases,
    ) = run_experiment(cfg)

    boxplot_by_lambda(
        ula_explosion_times,
        "ULA stability: explosion time vs step size",
        "ula_explosion_time_boxplot.png",
        "Explosion time, iterations",
        cfg,
        logy=True,
    )

    for method in TAMED_METHODS:
        file_stem = method.lower().replace(" ", "_")
        boxplot_by_lambda(
            second_moment_errors[method],
            f"{LABELS[method]} accuracy: second-moment error vs step size",
            f"{file_stem}_second_moment_error_boxplot.png",
            r"$|\widehat{\mathbb{E}}[X_1^2] - \mathbb{E}_\pi[X_1^2]|$",
            cfg,
        )

        boxplot_by_lambda(
            last_second_moment_errors[method],
            f"{LABELS[method]} accuracy: last-coordinate moment error",
            f"{file_stem}_last_coordinate_second_moment_error_boxplot.png",
            rf"$|\widehat{{\mathbb{{E}}}}[X_{{{cfg.dim}}}^2] - 1/(\beta\kappa)|$",
            cfg,
        )

    print_summary(
        ula_explosion_times,
        second_moment_errors,
        last_second_moment_errors,
        signed_biases,
        last_signed_biases,
        cfg,
    )

    write_sampling_csvs(
        second_moment_errors,
        last_second_moment_errors,
        signed_biases,
        last_signed_biases,
        cfg,
    )

    print(f"\nSaved figures to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
