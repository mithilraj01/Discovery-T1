
EXPERIMENT_CONFIG = {
    "deterministic": {
        "world_type": "deterministic",
        "params": [{"A": 5.0, "omega": 0.5, "phi": 1.0}],
        "seeds": [42]
    },
    "chaotic": {
        "world_type": "chaotic",
        "params": [{"r": 3.9, "x0": 0.5}],
        "seeds": [42],
        "perturbation_epsilon": 1e-10
    },
    "biological": {
        "world_type": "biological",
        "params": [{
            "initial_pop": 50, "resource_inflow": 500.0, "metabolic_cost": 1.0,
            "reproduction_threshold": 10.0, "mutation_rate": 0.05
        }],
        "seeds": [42],
        "perturbation_epsilon": 0.01
    },
    "liquid_continuum": {
        "world_type": "liquid_continuum",
        "params": [
            {
                "depth": 100.0,
                "num_layers": 20,
                "diffusion_rate": 0.2, # Stable if dz=1, dt=1, D<=0.5
                "surface_temp": 25.0,
                "bottom_temp": 5.0,
                "noise_level": 0.5
            },
            {
                "depth": 100.0,
                "num_layers": 50, # Higher res
                "diffusion_rate": 0.4, # Faster mixing
                "surface_temp": 25.0,
                "bottom_temp": 5.0,
                "noise_level": 1.0
            }
        ],
        "seeds": [42, 101, 999],
        "perturbation_epsilon": 0.1 # Perturb surface temp
    }
}
