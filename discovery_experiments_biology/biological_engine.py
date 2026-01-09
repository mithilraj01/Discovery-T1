
import numpy as np

def generate_biological_data(params, steps=200, seed=None):
    """
    Simulates a biological regime using a simple Agent-Based Model (ABM).

    Dynamics:
    - Resource limited environment.
    - Agents consume resources, metabolize, reproduce, and die.
    - Agents adapt (mutate) efficiency.

    Args:
        params (dict):
            - initial_pop (int): Starting population.
            - resource_inflow (float): Base resource added per step.
            - metabolic_cost (float): Energy cost per step.
            - reproduction_threshold (float): Energy needed to reproduce.
            - mutation_rate (float): SD of efficiency mutation.
            - competition_strength (float): Factor for resource sharing.
        steps (int): Simulation duration.
        seed (int): Random seed.

    Returns:
        dict: 'time', 'value' (Population), 'type'='biological', 'params', etc.
    """
    if seed is not None:
        np.random.seed(seed)

    initial_pop = params.get('initial_pop', 100)
    resource_inflow = params.get('resource_inflow', 1000.0)
    metabolic_cost = params.get('metabolic_cost', 1.0)
    reproduction_threshold = params.get('reproduction_threshold', 10.0)
    mutation_rate = params.get('mutation_rate', 0.01)

    # Agents: List of [energy, efficiency]
    # efficiency ~ 1.0 baseline. Higher = better resource extraction?
    # Or lower metabolic cost? Let's say efficiency = resource extraction multiplier.
    agents = np.ones((initial_pop, 2)) # col 0: energy, col 1: efficiency
    agents[:, 0] = reproduction_threshold / 2.0 # start with half repro energy

    pop_history = []
    mean_energy_history = []

    for t in range(steps):
        if len(agents) == 0:
            pop_history.append(0)
            mean_energy_history.append(0)
            continue

        # 1. Resource Inflow & Competition
        # Total demand = sum(efficiencies)
        total_efficiency = np.sum(agents[:, 1])

        # Available resource per efficiency unit
        # Simple logistic-like constraint or fixed pie?
        # Fixed pie: resource_inflow is distributed.
        if total_efficiency > 0:
            resource_per_eff = resource_inflow / total_efficiency
        else:
            resource_per_eff = 0

        # 2. Consumption & Metabolism
        # Gained = eff * resource_per_eff
        gains = agents[:, 1] * resource_per_eff
        agents[:, 0] += gains - metabolic_cost

        # 3. Death
        survivors = agents[agents[:, 0] > 0]

        # 4. Reproduction
        # Check who has enough energy
        if len(survivors) > 0:
            parents_mask = survivors[:, 0] > reproduction_threshold
            parents = survivors[parents_mask]

            if len(parents) > 0:
                # Spends energy to reproduce
                survivors[parents_mask, 0] -= reproduction_threshold / 2.0 # Cost

                # Offspring
                offspring = parents.copy()
                offspring[:, 0] = reproduction_threshold / 2.0 # Initial energy

                # Mutation
                mutations = np.random.normal(0, mutation_rate, size=len(offspring))
                offspring[:, 1] += mutations
                offspring[:, 1] = np.maximum(0.1, offspring[:, 1]) # Min efficiency

                survivors = np.vstack([survivors, offspring])

        agents = survivors

        pop_history.append(len(agents))
        mean_energy_history.append(np.mean(agents[:, 0]) if len(agents) > 0 else 0)

    # Return structure compatible with world_engine
    return {
        'time': list(range(steps)),
        'value': pop_history, # Primary observable is Population Size
        'mean_energy': mean_energy_history,
        'type': 'biological',
        'params': params,
        'seed': seed
    }
