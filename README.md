# Tamed Langevin Experiments

Reproducible numerical experiments and reusable Python implementations for adaptive-tamed kTULA and tRLMC under super-linear drift.

This repository contains:

1. **Reusable algorithms:** importable kTULA/tRLMC samplers and optimizers.
2. **Diagnostics:** utilities for active taming fraction, running second moments, drift distortion, and divergence detection.
3. **Sampling experiments:** high-dimensional double-well target comparing ULA, adaptive kTULA, and adaptive tamed rLMC.
4. **Optimization experiments:** neural-network stress test with untamed SGD/Adam/RMSProp/AMSGrad baselines and adaptive-tamed kTULA/tRLMC.

The adaptive taming used throughout is

```math
h_\lambda^{\mathrm{ad}}(x)
=
a x+
\frac{h(x)-a x}
{\sqrt{1+g(\lambda\|x\|^{2(\ell+1)})}}.
```

The residual vector is divided by one scalar denominator shared by every
coordinate. Taming is inactive when `sqrt(lambda) * ||x||^(ell+1) < 1` and
activates smoothly as the Euclidean norm of the full state increases. The switch is

```math
g(t)=
\begin{cases}
0, & 0\le t<1,\\
5(t-1)^2-3(t-1)^3, & 1\le t<2,\\
t, & t\ge 2.
\end{cases}
```

## Repository Layout

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
├── configs/
│   ├── optimization_default.json
│   └── sampling_default.json
├── examples/
│   ├── diagnostics_demo.py
│   ├── double_well_sampling.py
│   └── superlinear_optimization.py
├── experiments/
│   ├── optimization/
│   │   └── run_superlinear_stress_test.py
│   └── sampling/
│       └── run_adaptive_sampling.py
├── scripts/
│   ├── run_optimization.sh
│   ├── run_sampling.sh
│   └── reproduce_all.sh
├── src/
│   └── tamed_langevin/
│       ├── __init__.py
│       ├── diagnostics.py
│       ├── optimizers.py
│       ├── samplers.py
│       └── taming.py
├── tests/
│   ├── test_diagnostics.py
│   ├── test_optimizers.py
│   ├── test_samplers.py
│   └── test_taming.py
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

The `src/tamed_langevin/` directory contains the reusable package code. The `experiments/` directory contains the paper-style reproduction scripts. The `examples/` directory contains small usage examples for users who want to call the algorithms directly.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies and the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Alternatively:

```bash
python -m pip install -e .
```

## Run Tests

```bash
python -m pytest
```

Expected result:

```text
20 passed
```

## Python API

The package exposes reusable adaptive-tamed Langevin samplers, optimizers, and diagnostics.

### Sampling with kTULA

```python
import numpy as np

from tamed_langevin import KTULASampler


def drift(x):
    out = np.empty_like(x)
    out[0] = x[0]**3 - x[0]
    out[1:] = 100.0 * x[1:]
    return out


x0 = np.zeros(100)
x0[0] = 200.0

sampler = KTULASampler(
    drift=drift,
    step_size=0.01,
    beta=1.0,
    a_tame=0.05,
    ell_tame=2.0,
)

samples, active = sampler.sample(
    x0,
    n_steps=10000,
    burn_in=1000,
    seed=1,
    return_active=True,
)

print(samples.shape)
print(active.mean())
```

### Sampling with tRLMC

```python
import numpy as np

from tamed_langevin import TRLMCSampler


def drift(x):
    out = np.empty_like(x)
    out[0] = x[0]**3 - x[0]
    out[1:] = 100.0 * x[1:]
    return out


x0 = np.zeros(100)
x0[0] = 200.0

sampler = TRLMCSampler(
    drift=drift,
    step_size=0.01,
    beta=1.0,
    a_tame=0.05,
    ell_tame=2.0,
)

samples, active = sampler.sample(
    x0,
    n_steps=10000,
    burn_in=1000,
    seed=1,
    return_active=True,
)

print(samples.shape)
print(active.mean())
```

### Optimization with kTULA

```python
import numpy as np

from tamed_langevin import KTULAOptimizer


def grad(theta):
    return theta**5 - theta


rng = np.random.default_rng(1)

theta = np.zeros(20)
theta[0] = 10.0

optimizer = KTULAOptimizer(
    step_size=0.01,
    beta=1.0e6,
    a_tame=0.05,
    ell_tame=4.0,
)

for _ in range(1000):
    theta, active = optimizer.step(theta, grad(theta), rng)

print(np.linalg.norm(theta))
print(active)
```

### Optimization with tRLMC

```python
import numpy as np

from tamed_langevin import TRLMCOptimizer


def grad(theta):
    return theta**5 - theta


rng = np.random.default_rng(1)

theta = np.zeros(20)
theta[0] = 10.0

optimizer = TRLMCOptimizer(
    step_size=0.01,
    beta=1.0e6,
    a_tame=0.05,
    ell_tame=4.0,
)

for _ in range(1000):
    theta, active = optimizer.step(theta, grad(theta), grad, rng)

print(np.linalg.norm(theta))
print(active)
```

## Diagnostics

The package includes lightweight diagnostics:

```python
from tamed_langevin import (
    active_fraction,
    divergence_time,
    relative_drift_distortion,
    running_second_moment,
    second_moment_error,
)
```

Available diagnostics:

```text
active_fraction(active)
running_second_moment(samples)
relative_drift_distortion(original_drift, tamed_drift)
divergence_time(path)
second_moment_error(samples, reference_second_moment)
```

Example:

```python
import numpy as np

from tamed_langevin import active_fraction, running_second_moment


samples = np.random.normal(size=(1000, 10))
active = np.array([True, False, False, True])

print(active_fraction(active))
print(running_second_moment(samples)[-1])
```

## Run Example Scripts

```bash
python examples/double_well_sampling.py
python examples/superlinear_optimization.py
python examples/diagnostics_demo.py
```

Expected behavior:

```text
double_well_sampling.py          produces kTULA/tRLMC samples
superlinear_optimization.py      runs a simple super-linear optimization example
diagnostics_demo.py              reports active fraction, divergence time, and running second moment
```

## Smoke Tests for Paper Experiments

```bash
python experiments/optimization/run_superlinear_stress_test.py --quick --skip-order
```

The full sampling experiment is expensive because it runs long Markov chains with 30 repetitions.

## Run the Sampling Experiment

```bash
python experiments/sampling/run_adaptive_sampling.py
```

Default outputs are saved to:

```text
./figures/Sampling_Experiments_Adaptive_rLMC
```

Main outputs:

```text
ula_explosion_time_boxplot.png
ktula_second_moment_error_boxplot.png
trlmc_second_moment_error_boxplot.png
moment_d100_lam_*.png
density_ktula_*_d100_lam_*.png
density_tamed_rlmc_*_d100_lam_*.png
```

## Run the Optimization Experiment

```bash
python experiments/optimization/run_superlinear_stress_test.py \
  --out-dir ./figures/SuperLinear_SINUM
```

Fast check:

```bash
python experiments/optimization/run_superlinear_stress_test.py --quick --skip-order
```

Main outputs:

```text
fig1_empirical_objective_all_methods.png
fig2_test_mse_all_methods.png
fig3_parameter_norm_all_methods.png
fig4_learning_rate_robustness_all_methods.png
fig5_taming_active_fraction.png
fig6_adaptive_gradient_distortion.png
fig7_regime_map_all_methods.png
fig8_observed_order_mean_potential.png
fig9_observed_order_second_moment.png
summary_main.csv
summary_lr_sweep.csv
summary_regime_map.csv
summary_observed_order.csv
reference_observed_order.csv
```

## Reproduce All Default Experiments

```bash
bash scripts/reproduce_all.sh
```

## Experimental Interpretation

The experiments are deliberately aggressive. They test the super-linear drift regime where explicit Euler-type schemes are expected to be unstable.

The intended experimental claim is that adaptive-tamed kTULA and tRLMC remain stable in this regime, while untamed explicit methods either diverge or stagnate.

The optimization experiment is a stress test, not a general-purpose benchmark for neural-network training. Its purpose is to expose instability under super-linear drift and compare this behavior with adaptive-tamed schemes.

## Development

Run the test suite:

```bash
python -m pytest
```

Run all examples:

```bash
python examples/double_well_sampling.py
python examples/superlinear_optimization.py
python examples/diagnostics_demo.py
```

Run quick experiment checks:

```bash
python experiments/optimization/run_superlinear_stress_test.py --quick --skip-order
```

## Citation

If you use this software, please cite the associated paper and this repository. Citation metadata is provided in `CITATION.cff`.

## License

This project is released under the MIT license.
