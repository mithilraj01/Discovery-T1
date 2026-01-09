# Discovery Experiments (Ocean / Liquid Continuum)

This directory contains the experiment pipeline for the Ocean / Liquid Discoverability Experiments.

## Purpose

The goal of this module is to measure how discoverability behaves in continuous field-based liquid systems.
It tests whether the presence of smooth continuous fields (Temperature, Pressure, Density) restores stable, reusable explanation compared to chaotic or biological regimes.

## Methodology

1.  **Ocean Simulation**: A 1D vertical column model simulates thermal diffusion and optional advection/mixing.
2.  **Input Handling**:
    *   Vertical profiles of T, P, Density are generated.
    *   Discovery baselines operate on aggregate summaries (Mean Temperature) to ensure comparability with scalar time series from other regimes.
3.  **Baselines**: We reuse the standard suite (Linear, Polynomial, Compression, Spline, KRR).
4.  **Metrics**: Instability, Description Length, Consistency.

## Experiment Contract

1.  **Continuum Assumption**: Liquids are treated as continuous fields.
2.  **No Fluid Laws**: We do not infer Navier-Stokes or Heat Equation directly; we test if generic approximators can stably represent the evolution.
3.  **Smoothness vs. Explanability**: We investigate if physical smoothness implies better explainability (lower DL, higher stability).

## Running Experiments

Execute the runner:

```bash
python3 discovery_experiments_ocean/experiment_runner.py
```

Results will be populated in `discovery_experiments_ocean/results/`.
