
import numpy as np

def generate_ocean_data(params, steps=200, seed=None):
    """
    Simulates a 1D vertical column of liquid (Ocean Column).
    Dynamics: Diffusion (Heat Equation) with optional Advection/Mixing.

    State: Temperature T(z, t).
    Derived: Density rho(z, t), Pressure P(z, t).

    Args:
        params (dict):
            - depth (float): Total depth (m).
            - num_layers (int): Discretization.
            - diffusion_rate (float): Thermal diffusivity.
            - surface_temp (float): Boundary condition top.
            - bottom_temp (float): Boundary condition bottom.
            - mixing_pulse_time (int): Step to inject heat.
            - mixing_pulse_depth (float): Relative depth (0-1) to inject.
            - mixing_pulse_amount (float): Temp increase.
            - noise_level (float): Initial noise sigma.
        steps (int): Simulation duration.
        seed (int): Random seed.

    Returns:
        dict: 'time', 'value' (Mean Temperature), 'fields' (Profiles), 'type'='liquid_continuum', params.
    """
    if seed is not None:
        np.random.seed(seed)

    depth = params.get('depth', 100.0)
    nz = params.get('num_layers', 20)
    D = params.get('diffusion_rate', 0.1)
    T_surf = params.get('surface_temp', 20.0)
    T_bot = params.get('bottom_temp', 4.0)

    pulse_t = params.get('mixing_pulse_time', -1)
    pulse_z = params.get('mixing_pulse_depth', 0.5)
    pulse_amt = params.get('mixing_pulse_amount', 0.0)

    noise = params.get('noise_level', 0.1)

    dz = depth / nz
    dt = 0.5 * dz**2 / (D + 1e-9) # Stability limit approximation
    # Enforce a reasonable dt (e.g. 1.0) and adjust D or normalize steps
    # For simplicity, let's treat 'steps' as simulation steps, and D as 'diffusion per step'.
    # Stability: alpha = D * dt / dz^2 <= 0.5.
    # Let's set dt=1, dz=1 (relative units) and ensure D <= 0.4.

    # Grid
    T = np.linspace(T_surf, T_bot, nz) + np.random.normal(0, noise, nz)

    # Physics constants
    rho0 = 1025.0
    g = 9.81
    alpha_exp = 2e-4 # Thermal expansion

    history_mean_T = []
    history_profiles = []

    for t in range(steps):
        # 1. Diffusion
        # T_new = T + D * laplacian(T)
        T_new = T.copy()

        # Interior points
        for i in range(1, nz - 1):
            laplacian = (T[i+1] - 2*T[i] + T[i-1]) # / dz**2 implicitly if D is scaled
            T_new[i] += D * laplacian

        # Boundary Conditions (Fixed)
        T_new[0] = T_surf
        T_new[-1] = T_bot

        # 2. Mixing Pulse (Intervention)
        if t == pulse_t:
            idx = int(pulse_z * nz)
            idx = max(1, min(nz-2, idx))
            T_new[idx] += pulse_amt

        T = T_new

        # Derived fields
        # rho = rho0 * (1 - alpha * (T - T_mean)) approx
        rho = rho0 * (1 - alpha_exp * (T - 10.0))

        # P = integral(rho * g * dz)
        P = np.cumsum(rho * g * (depth/nz))

        history_mean_T.append(np.mean(T))
        # Store profile sparsely? Or full? 20 floats is fine.
        history_profiles.append(T.tolist())

    return {
        'time': list(range(steps)),
        'value': history_mean_T, # 1D observable for baselines
        'fields': {
            'T': history_profiles
        },
        'type': 'liquid_continuum',
        'params': params,
        'seed': seed
    }
