
import numpy as np
import zlib
import json
from scipy.interpolate import LSQUnivariateSpline
from sklearn.kernel_ridge import KernelRidge

class BaselineLinear:
    """
    Simple Linear Regression Approximation.
    Fits x[t] = w0 + w1 * x[t-1]
    """
    def fit(self, time, values, metadata=None):
        if len(values) < 2:
            return {"error": "Not enough data"}

        # Prepare data for x[t] = a * x[t-1] + b
        X = np.array(values[:-1])
        y = np.array(values[1:])

        # Simple least squares: y = mx + c
        A = np.vstack([X, np.ones(len(X))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        # Calculate residuals
        y_pred = m * X + c
        residuals = y - y_pred

        return {
            "model_type": "linear",
            "parameters": {"slope": m, "intercept": c},
            "residuals": residuals.tolist(),
            "residual_norm": np.linalg.norm(residuals)
        }

class BaselinePolynomial:
    """
    Polynomial Regression of bounded degree.
    """
    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, time, values, metadata=None):
        if len(values) < self.degree + 1:
            return {"error": "Not enough data"}

        t = np.array(time)
        x = np.array(values)

        # Fit polynomial
        coeffs = np.polyfit(t, x, self.degree)

        # Residuals
        p = np.poly1d(coeffs)
        x_pred = p(t)
        residuals = x - x_pred

        return {
            "model_type": f"polynomial_deg_{self.degree}",
            "parameters": coeffs.tolist(),
            "residuals": residuals.tolist(),
            "residual_norm": np.linalg.norm(residuals)
        }

class BaselineCompression:
    """
    Compression-Based Proxy (MDL-like).
    Estimates complexity using zlib compression of residuals + model parameter proxy.
    """
    def fit(self, time, values, metadata=None):
        # We need a reference "model". Let's assume a simple mean model or linear model
        # as the base, and compress the residuals.

        # Strategy: Use Linear Baseline first, then compress residuals.
        lin = BaselineLinear()
        res = lin.fit(time, values)

        if "error" in res:
             return {"error": res["error"]}

        residuals = np.array(res["residuals"])

        # Quantize residuals to standard precision
        # Use metadata to adjust precision if available (e.g. knowing noise floor)
        decimals = 4
        if metadata and 'params' in metadata and 'sigma' in metadata['params']:
            sigma = metadata['params']['sigma']
            if sigma > 0:
                # If sigma is 0.1, we might want 1 decimal? No, we want enough bits.
                # If sigma is noise, quantization step should be around sigma.
                # So decimals ~ -log10(sigma).
                # sigma=0.1 -> 1 decimal. sigma=0.01 -> 2 decimals.
                decimals = max(0, int(-np.log10(sigma))) + 1

        res_str = json.dumps([round(x, decimals) for x in residuals])
        compressed_res = zlib.compress(res_str.encode('utf-8'))

        # Model complexity proxy: float string length of parameters
        params_str = json.dumps(res["parameters"])
        model_complexity = len(params_str)

        return {
            "model_type": "compression_proxy",
            "model_complexity": model_complexity,
            "compressed_residual_size": len(compressed_res),
            "approx_dl": model_complexity + len(compressed_res),
            "raw_data_size": len(json.dumps(values)),
            "compression_ratio": (model_complexity + len(compressed_res)) / len(json.dumps(values)),
            "quantization_decimals": decimals
        }

class BaselineSpline:
    """
    Piecewise polynomial spline regression (Control).
    """
    def __init__(self, knots=5, order=3):
        self.knots_count = knots
        self.order = order

    def fit(self, time, values, metadata=None):
        t = np.array(time)
        x = np.array(values)

        if len(x) < self.order + self.knots_count + 1:
             return {"error": "Not enough data for spline"}

        # Define fixed knots (excluding boundaries)
        knots = np.linspace(t[0], t[-1], self.knots_count + 2)[1:-1]

        try:
            spline = LSQUnivariateSpline(t, x, knots, k=self.order)
            x_pred = spline(t)
            residuals = x - x_pred
            coeffs = spline.get_coeffs()

            return {
                "model_type": f"spline_knots{self.knots_count}_order{self.order}",
                "parameters": coeffs.tolist(),
                "residuals": residuals.tolist(),
                "residual_norm": np.linalg.norm(residuals)
            }
        except Exception as e:
            return {"error": str(e)}

class BaselineKernelRidge:
    """
    Kernel Ridge Regression (Control).
    RBF kernel, fixed bandwidth, fixed regularization.
    """
    def __init__(self, alpha=1.0, gamma=0.1):
        self.base_alpha = alpha # Base Regularization
        self.gamma = gamma # RBF width parameter
        # We don't init model here because alpha might change per fit if regime-aware

    def fit(self, time, values, metadata=None):
        t = np.array(time).reshape(-1, 1)
        x = np.array(values)

        if len(x) < 2:
             return {"error": "Not enough data"}

        # Adjust alpha based on regime info (e.g. noise level)
        current_alpha = self.base_alpha
        if metadata and 'params' in metadata and 'sigma' in metadata['params']:
            sigma = metadata['params']['sigma']
            # Heuristic: more noise -> more regularization
            current_alpha = self.base_alpha * (1 + 10 * sigma)

        model = KernelRidge(alpha=current_alpha, kernel='rbf', gamma=self.gamma)

        model.fit(t, x)
        x_pred = model.predict(t)
        residuals = x - x_pred

        # Dual coefficients are the "parameters"
        coeffs = model.dual_coef_

        return {
            "model_type": f"krr_rbf_alpha{current_alpha:.4f}_gamma{self.gamma}",
            "parameters": coeffs.tolist(),
            "residuals": residuals.tolist(),
            "residual_norm": np.linalg.norm(residuals),
            "effective_alpha": current_alpha
        }
