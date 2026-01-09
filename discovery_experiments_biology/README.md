# Discovery Experiments (Biology)

This directory contains the experiment pipeline for Phase B2: Biological Discoverability Experiments.

## Purpose

The goal of this module is to evaluate how discoverability behaves in biological systems compared to deterministic, stochastic, and chaotic regimes.
It tests whether adaptation, irreversibility, and historical contingency further degrade the possibility of compact, stable explanation.

## Methodology

1.  **Biological Simulation**: A mock Agent-Based Model (ABM) simulates population dynamics with resource competition, metabolism, reproduction, and mutation (adaptation).
2.  **Input Handling**: We treat biological outputs (population size, mean energy) as raw data, without imposing biological semantics (fitness, species labels) on the discovery process.
3.  **Baselines**: We use the same set of baselines as in Phase D2 (Linear, Polynomial, Compression, Spline, KRR) to isolate the effect of the *regime* (biology) vs. the *algorithm*.
4.  **Metrics**: We track Description Length, Stability, and Consistency to detect breakdown modes.

## Experiment Contract

1.  **Biology as Regime**: Biology is treated as a dynamical system class, distinct from simple chaos or stochasticity due to adaptation and history.
2.  **Empirical Limits**: Failures to find stable representations are interpreted as empirical boundaries of explanation, not flaws in the solver.
3.  **No Inferred Intelligence**: We do not assume the agents are "smart", only that they evolve.

## Running Experiments

Execute the runner:

```bash
python3 discovery_experiments_biology/experiment_runner.py
```

Results will be populated in `discovery_experiments_biology/results/`.
