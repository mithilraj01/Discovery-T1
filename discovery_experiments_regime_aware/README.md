# Discovery Experiments (Regime Aware)

This directory contains the experiment pipeline for Phase D2: Regime-Aware Discoverability Experiments.

## Purpose

The goal of this module is to test whether making physics regimes and fidelity limits explicit improves or stabilizes discovery.
It builds upon Phase D1/D1.5 by exposing simulator metadata (regime type, parameters like noise level) to the discovery baselines.

## Key Changes from Phase D1/D1.5

*   **Metadata Input**: `fit()` methods now accept a `metadata` dictionary containing regime descriptors.
*   **Conditioned Baselines**:
    *   `BaselineCompression`: Uses noise level (`sigma`) to determine quantization precision, preventing "fitting the noise" from bloating the description length.
    *   `BaselineKernelRidge`: Uses noise level (`sigma`) to adjust regularization strength (`alpha`), robustifying against stochasticity.
*   **Metrics**: Identical to D1/D1.5 to ensure direct comparison.

## Running Experiments

Execute the runner:

```bash
python3 discovery_experiments_regime_aware/experiment_runner.py
```

Results will be populated in `discovery_experiments_regime_aware/results/`.

## Experiment Contract

1.  **Regime Awareness**: We assume the agent knows the *context* (regime labels, limits) but not the *laws*.
2.  **No Enforced Physics**: The metadata conditions the *approximation*, it does not force a specific equation form.
3.  **Failure Analysis**: We look for cases where regime awareness fails to fix instability (especially in chaos), helping to triangulate the true discoverability boundary.
