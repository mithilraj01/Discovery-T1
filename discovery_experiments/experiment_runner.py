
import json
import os
import sys
import numpy as np

# Adjust path to find simulator and local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.world_engine import generate_world_data
from discovery_experiments.baselines import BaselineLinear, BaselinePolynomial, BaselineCompression, BaselineSpline, BaselineKernelRidge
from discovery_experiments.metrics import approximate_description_length, stability_under_perturbation, cross_run_consistency, intervention_sensitivity
from discovery_experiments.experiment_config import EXPERIMENT_CONFIG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'phase_D1_5')
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiments():
    all_results = []

    baselines = [
        BaselineLinear(),
        BaselinePolynomial(degree=3),
        BaselinePolynomial(degree=5),
        BaselineCompression(),
        BaselineSpline(knots=10, order=3),
        BaselineKernelRidge(alpha=0.1, gamma=0.5)
    ]

    for category, config in EXPERIMENT_CONFIG.items():
        world_type = config["world_type"]
        param_sets = config["params"]
        seeds = config["seeds"]

        for params in param_sets:
            # We need to collect results for consistency check later
            param_results_map = {bl.__class__.__name__ + (f"_deg{bl.degree}" if hasattr(bl, 'degree') else ""): [] for bl in baselines}

            for seed in seeds:
                # 1. Base run
                data = generate_world_data(world_type, params, steps=200, seed=seed)

                # 2. Chaotic Perturbation (for Stability)
                data_perturbed = None
                if category == "chaotic":
                    epsilon = config.get("perturbation_epsilon", 1e-10)
                    params_perturbed = params.copy()
                    params_perturbed['x0'] += epsilon
                    data_perturbed = generate_world_data(world_type, params_perturbed, steps=200, seed=seed)

                # 3. Intervention Simulation (Parameter Step Change)
                # We simulate a "post-intervention" world by slightly modifying a parameter
                data_intervention = None
                params_intervention = params.copy()
                # Modify a parameter depending on type
                if world_type == 'deterministic':
                    params_intervention['A'] *= 1.1
                elif world_type == 'stochastic':
                    params_intervention['alpha'] = max(0.1, min(0.95, params_intervention['alpha'] * 1.1))
                elif world_type == 'chaotic':
                    params_intervention['r'] = max(2.5, min(4.0, params_intervention['r'] * 0.98))

                data_intervention = generate_world_data(world_type, params_intervention, steps=200, seed=seed+9999) # New seed for post-intervention noise if stochastic

                # Run Baselines
                for baseline in baselines:
                    bl_name = baseline.__class__.__name__
                    if hasattr(baseline, 'degree'):
                        bl_name += f"_deg{baseline.degree}"

                    # Fit on base data
                    res = baseline.fit(data['time'], data['value'])

                    # Compute DL
                    dl = approximate_description_length(res)

                    # Compute Stability if applicable
                    stability = None
                    if data_perturbed:
                        res_perturbed = baseline.fit(data_perturbed['time'], data_perturbed['value'])
                        stability = stability_under_perturbation(res, res_perturbed)

                    # Compute Intervention Sensitivity
                    sensitivity = None
                    if data_intervention:
                        res_intervention = baseline.fit(data_intervention['time'], data_intervention['value'])
                        sensitivity = intervention_sensitivity(res, res_intervention)

                    # Check Failure Conditions (Per Run)
                    failed = False
                    failure_reason = []

                    # Fail if DL diverges
                    if dl == float('inf') or np.isnan(dl) or dl > 1e6:
                        failed = True
                        failure_reason.append("dl_divergence")

                    # Fail if unstable (chaotic only usually)
                    if stability is not None and (stability == float('inf') or stability > 1.0):
                        failed = True
                        failure_reason.append("instability")

                    # Store for consistency check
                    result_record = {
                        "category": category,
                        "world_type": world_type,
                        "params": params,
                        "seed": seed,
                        "baseline": bl_name,
                        "approx_dl": dl,
                        "stability": stability,
                        "intervention_sensitivity": sensitivity,
                        "failed": failed,
                        "failure_reasons": failure_reason,
                        "raw_res": res # Keep raw result for later consistency check
                    }

                    param_results_map[bl_name].append(result_record)

            # 4. Consistency Check (After all seeds for this param set)
            if len(seeds) > 1:
                for bl_name, records in param_results_map.items():
                    # Extract DLs or raw results
                    raw_results_list = [r["raw_res"] for r in records]
                    consistency_variance = cross_run_consistency(raw_results_list)

                    # Update records with consistency metric and check failure
                    for record in records:
                        record["cross_run_consistency_variance"] = consistency_variance
                        del record["raw_res"] # Remove raw_res before saving

                        # Fail if inconsistent (high variance)
                        # Threshold is arbitrary but let's say > 100 variance in DL is "inconsistent"
                        if consistency_variance > 100.0:
                            record["failed"] = True
                            record["failure_reasons"].append("inconsistency")

                        all_results.append(record)

                        # Save individual result
                        filename = f"{category}_{bl_name}_{record['seed']}_{np.random.randint(0,10000)}.json"
                        with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
                            json.dump(record, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=2)
            else:
                # Just save the records if only 1 seed
                for bl_name, records in param_results_map.items():
                    for record in records:
                        record["cross_run_consistency_variance"] = 0.0 # Single run is consistent
                        del record["raw_res"]
                        all_results.append(record)
                        filename = f"{category}_{bl_name}_{record['seed']}_{np.random.randint(0,10000)}.json"
                        with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
                            json.dump(record, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=2)

    print(f"Experiments completed. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_experiments()
