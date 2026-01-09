
import json
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulator.world_engine import generate_world_data
from discovery_experiments_biology.biological_engine import generate_biological_data
from discovery_experiments_ocean.ocean_engine import generate_ocean_data

from discovery_experiments_ocean.baselines import BaselineLinear, BaselinePolynomial, BaselineCompression, BaselineSpline, BaselineKernelRidge
from discovery_experiments_ocean.metrics import approximate_description_length, stability_under_perturbation, cross_run_consistency, intervention_sensitivity
from discovery_experiments_ocean.experiment_config import EXPERIMENT_CONFIG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_data(category, world_type, params, steps, seed):
    if world_type == 'biological':
        return generate_biological_data(params, steps=steps, seed=seed)
    elif world_type == 'liquid_continuum':
        return generate_ocean_data(params, steps=steps, seed=seed)
    else:
        return generate_world_data(world_type, params, steps=steps, seed=seed)

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
            param_results_map = {bl.__class__.__name__ + (f"_deg{bl.degree}" if hasattr(bl, 'degree') else ""): [] for bl in baselines}

            for seed in seeds:
                # 1. Base run
                data = get_data(category, world_type, params, 200, seed)

                regime_metadata = {
                    'regime': data['type'],
                    'params': data['params']
                }
                # Inject sigma for quantization if missing (e.g. liquid)
                if world_type == 'liquid_continuum':
                     # Use noise_level as sigma
                     regime_metadata['params']['sigma'] = params.get('noise_level', 0.1)

                # 2. Perturbation
                data_perturbed = None
                if category == "chaotic":
                    epsilon = config.get("perturbation_epsilon", 1e-10)
                    params_perturbed = params.copy()
                    params_perturbed['x0'] += epsilon
                    data_perturbed = get_data(category, world_type, params_perturbed, 200, seed)
                elif category == "biological":
                    epsilon = config.get("perturbation_epsilon", 0.01)
                    params_perturbed = params.copy()
                    params_perturbed['resource_inflow'] *= (1.0 + epsilon)
                    data_perturbed = get_data(category, world_type, params_perturbed, 200, seed)
                elif category == "liquid_continuum":
                    # Perturb surface temperature (Boundary Condition)
                    epsilon = config.get("perturbation_epsilon", 0.1)
                    params_perturbed = params.copy()
                    params_perturbed['surface_temp'] += epsilon
                    data_perturbed = get_data(category, world_type, params_perturbed, 200, seed)

                # 3. Intervention
                data_intervention = None
                params_intervention = params.copy()
                if world_type == 'deterministic':
                    params_intervention['A'] *= 1.1
                elif world_type == 'chaotic':
                    params_intervention['r'] = max(2.5, min(4.0, params_intervention['r'] * 0.98))
                elif world_type == 'biological':
                    params_intervention['resource_inflow'] *= 0.8
                elif world_type == 'liquid_continuum':
                    # Mixing Pulse Intervention
                    # Inject a pulse of heat halfway through
                    params_intervention['mixing_pulse_time'] = 100
                    params_intervention['mixing_pulse_amount'] = 5.0 # Large heat injection

                data_intervention = get_data(category, world_type, params_intervention, 200, seed+9999)

                # Run Baselines
                for baseline in baselines:
                    bl_name = baseline.__class__.__name__
                    if hasattr(baseline, 'degree'):
                        bl_name += f"_deg{baseline.degree}"

                    res = baseline.fit(data['time'], data['value'], metadata=regime_metadata)
                    dl = approximate_description_length(res)

                    stability = None
                    if data_perturbed:
                        res_perturbed = baseline.fit(data_perturbed['time'], data_perturbed['value'], metadata=regime_metadata)
                        stability = stability_under_perturbation(res, res_perturbed)

                    sensitivity = None
                    if data_intervention:
                        intervention_metadata = {'regime': data_intervention['type'], 'params': data_intervention['params']}
                        res_intervention = baseline.fit(data_intervention['time'], data_intervention['value'], metadata=intervention_metadata)
                        sensitivity = intervention_sensitivity(res, res_intervention)

                    failed = False
                    failure_reason = []

                    if dl == float('inf') or np.isnan(dl) or dl > 1e6:
                        failed = True
                        failure_reason.append("dl_divergence")

                    if stability is not None and (stability == float('inf') or stability > 10.0):
                        failed = True
                        failure_reason.append("instability")

                    # "Reuse across depths or time windows fails"
                    # We are testing reuse across INTERVENTION (mixing pulse) via 'sensitivity'
                    # and reuse across PERTURBATION (surface temp) via 'stability'.
                    # If stability is bad, reuse fails.

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
                        "regime_aware": True,
                        "raw_res": res
                    }
                    param_results_map[bl_name].append(result_record)

            # 4. Consistency
            if len(seeds) > 1:
                for bl_name, records in param_results_map.items():
                    raw_results_list = [r["raw_res"] for r in records]
                    consistency_variance = cross_run_consistency(raw_results_list)

                    for record in records:
                        record["cross_run_consistency_variance"] = consistency_variance
                        del record["raw_res"]

                        # Liquid might be stochastic due to init noise, but diff eq is deterministic.
                        # Unless noise is added per step? In ocean_engine, loop: T_new = ... no random noise added per step.
                        # Only Init noise. So it should be deterministic trajectory given seed.
                        # Wait, consistent across seeds? If init noise depends on seed, trajectories differ.
                        # But physics is stable. Variance should be bounded?
                        if consistency_variance > 100.0:
                             # Just record
                             pass

                        all_results.append(record)
                        filename = f"{category}_{bl_name}_{record['seed']}_{np.random.randint(0,10000)}.json"
                        with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
                            json.dump(record, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=2)
            else:
                for bl_name, records in param_results_map.items():
                    for record in records:
                        record["cross_run_consistency_variance"] = 0.0
                        del record["raw_res"]
                        all_results.append(record)
                        filename = f"{category}_{bl_name}_{record['seed']}_{np.random.randint(0,10000)}.json"
                        with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
                            json.dump(record, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=2)

    print(f"Experiments completed. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_experiments()
