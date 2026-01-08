# Discovery Experiments

This directory contains the experiment pipeline for Phase D1: Discoverability Boundary Experiments.

## Purpose

The goal of this module is to empirically probe the limits of law discovery and compact explanation in simulated worlds.
It does **NOT** attempt to discover true laws or physical equations.
Instead, it studies when standard discovery baselines fail to produce compact, stable representations.

## Structure

*   `experiment_runner.py`: Main execution script. Loads simulator outputs, applies baselines, and logs results.
*   `baselines.py`: Implementation of discovery baselines (Linear, Polynomial, Compression). These are stress tools, not solvers.
*   `metrics.py`: Metrics for evaluating discoverability (Description Length, Stability, Consistency).
*   `experiment_config.py`: Configuration grid for experiments (Deterministic, Stochastic, Chaotic).
*   `results/`: Directory containing raw experiment outputs.

## Experiment Contract

1.  **No Discovery of Truth**: This module does not claim to find the "correct" laws of the simulation.
2.  **Proxies Only**: All metrics (DL, Stability) are proxies for explainability.
3.  **Failure as Boundary**: "Failure" of a baseline (e.g., diverging DL, instability) is a signal of an empirical boundary where compact explanation degrades.
4.  **Epistemically Neutral**: We assume nothing about the underlying ontology of the simulator.

## Running Experiments

Execute the runner:

```bash
python3 discovery_experiments/experiment_runner.py
```

Results will be populated in `discovery_experiments/results/`.
