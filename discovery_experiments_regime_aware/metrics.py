
import numpy as np

def approximate_description_length(results):
    """
    Computes proxy DL based on model results.
    If 'approx_dl' is present (Compression baseline), returns that.
    Otherwise estimates based on parameter count + residual norm.
    """
    if "approx_dl" in results:
        return results["approx_dl"]

    if "error" in results:
        return float('inf')

    # Heuristic for other models:
    # DL ~ k * log(N) + (N/2) * log(RSS/N)
    # k: number of params
    # N: number of data points
    # RSS: sum of squared residuals

    residuals = np.array(results.get("residuals", []))
    if len(residuals) == 0:
        return float('inf')

    N = len(residuals)
    RSS = np.sum(residuals**2)

    # Estimate k from parameters
    params = results.get("parameters", [])
    if isinstance(params, dict):
        k = len(params)
    elif isinstance(params, list):
        k = len(params)
    else:
        k = 1

    if RSS <= 1e-10:
        # Avoid log(0), treat as very low entropy
        term2 = -N * 10
    else:
        term2 = (N / 2) * np.log(RSS / N)

    dl = k * np.log(N) + term2
    return dl

def stability_under_perturbation(result_a, result_b):
    """
    Measures change in fitted representation between two runs.
    Uses parameter distance or residual norm difference.
    """
    if "error" in result_a or "error" in result_b:
        return float('inf')

    # Check if same model type
    if result_a.get("model_type") != result_b.get("model_type"):
        return float('nan')

    # Compare parameters
    params_a = result_a.get("parameters")
    params_b = result_b.get("parameters")

    if isinstance(params_a, dict) and isinstance(params_b, dict):
        # Linear: slope, intercept
        diff = 0
        for k in params_a:
            diff += (params_a[k] - params_b.get(k, 0))**2
        return np.sqrt(diff)

    elif isinstance(params_a, list) and isinstance(params_b, list):
        # Polynomial: coefficients
        a = np.array(params_a)
        b = np.array(params_b)
        if len(a) != len(b):
            return float('inf')
        return np.linalg.norm(a - b)

    elif result_a.get("model_type") == "compression_proxy":
        # For compression, stability is diff in DL or ratio
        return abs(result_a["approx_dl"] - result_b["approx_dl"])

    return float('nan')

def cross_run_consistency(results_list):
    """
    Checks variance of outputs across repeated runs (on stochastic data).
    Returns variance of the metric (e.g. DL).
    """
    dls = [approximate_description_length(r) for r in results_list]
    dls = [d for d in dls if d != float('inf')]

    if len(dls) < 2:
        return 0.0

    return np.var(dls)

def intervention_sensitivity(result_pre, result_post):
    """
    Measure discontinuity in representation.
    Similar to stability but implies a larger shift.
    """
    return stability_under_perturbation(result_pre, result_post)
