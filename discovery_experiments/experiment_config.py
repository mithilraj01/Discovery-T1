
EXPERIMENT_CONFIG = {
    "deterministic": {
        "world_type": "deterministic",
        "params": [
            {"A": 1.0, "omega": 0.1, "phi": 0.0},
            {"A": 5.0, "omega": 0.5, "phi": 1.0}
        ],
        "seeds": [42] # Deterministic needs only one seed usually, but kept for consistency
    },
    "stochastic": {
        "world_type": "stochastic",
        "params": [
            {"alpha": 0.5, "sigma": 0.1},
            {"alpha": 0.9, "sigma": 0.5}, # Near unit root
            {"alpha": 1.0, "sigma": 0.1}  # Random walk
        ],
        "seeds": [42, 101, 999]
    },
    "chaotic": {
        "world_type": "chaotic",
        "params": [
            {"r": 3.5, "x0": 0.5}, # Period doubling
            {"r": 3.9, "x0": 0.5}, # Fully chaotic
        ],
        "seeds": [42], # Seed controls initial state if randomized, but we set x0 explicitly.
                       # However, we will use seeds to generate epsilon perturbations.
        "perturbation_epsilon": 1e-10
    }
}
