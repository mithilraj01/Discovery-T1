
EXPERIMENT_CONFIG = {
    "deterministic": {
        "world_type": "deterministic",
        "params": [
            {"A": 5.0, "omega": 0.5, "phi": 1.0}
        ],
        "seeds": [42]
    },
    "stochastic": {
        "world_type": "stochastic",
        "params": [
            {"alpha": 0.9, "sigma": 0.5}
        ],
        "seeds": [42, 101]
    },
    "chaotic": {
        "world_type": "chaotic",
        "params": [
            {"r": 3.9, "x0": 0.5}
        ],
        "seeds": [42],
        "perturbation_epsilon": 1e-10
    },
    "biological": {
        "world_type": "biological",
        "params": [
            {
                "initial_pop": 50,
                "resource_inflow": 500.0,
                "metabolic_cost": 1.0,
                "reproduction_threshold": 10.0,
                "mutation_rate": 0.05
            },
            {
                "initial_pop": 50,
                "resource_inflow": 500.0,
                "metabolic_cost": 1.0,
                "reproduction_threshold": 10.0,
                "mutation_rate": 0.2 # High mutation -> faster divergence?
            }
        ],
        "seeds": [42, 101, 999],
        "perturbation_epsilon": 0.01 # For params like metabolic_cost
    }
}
