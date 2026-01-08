
import numpy as np

def generate_world_data(world_type, params, steps=100, seed=None):
    """
    Generates time-series data for a simulated world.

    Args:
        world_type (str): 'deterministic', 'stochastic', or 'chaotic'.
        params (dict): Parameters for the world.
        steps (int): Number of time steps.
        seed (int): Random seed.

    Returns:
        dict: containing 'time', 'value', 'type', 'params'.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(steps)
    x = np.zeros(steps)

    if world_type == 'deterministic':
        # Simple Harmonic Oscillator: x(t) = A * sin(omega * t + phi)
        # params: A, omega, phi
        A = params.get('A', 1.0)
        omega = params.get('omega', 0.1)
        phi = params.get('phi', 0.0)
        x = A * np.sin(omega * t + phi)

    elif world_type == 'stochastic':
        # AR(1) Process: x_t = alpha * x_{t-1} + noise
        # params: alpha, sigma
        alpha = params.get('alpha', 0.9)
        sigma = params.get('sigma', 0.1)
        x[0] = np.random.normal(0, sigma)
        for i in range(1, steps):
            x[i] = alpha * x[i-1] + np.random.normal(0, sigma)

    elif world_type == 'chaotic':
        # Logistic Map: x_{n+1} = r * x_n * (1 - x_n)
        # params: r, x0
        # Typically chaotic for r > 3.57
        r = params.get('r', 3.9)
        x[0] = params.get('x0', 0.5)
        for i in range(1, steps):
            x[i] = r * x[i-1] * (1 - x[i-1])

    else:
        raise ValueError(f"Unknown world type: {world_type}")

    return {
        'time': t.tolist(),
        'value': x.tolist(),
        'type': world_type,
        'params': params,
        'seed': seed
    }
