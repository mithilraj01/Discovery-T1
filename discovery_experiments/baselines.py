
import numpy as np
import zlib
import json

class BaselineLinear:
    """
    Simple Linear Regression Approximation.
    Fits x[t] = w0 + w1 * x[t-1]
    """
    def fit(self, time, values):
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
    Fits x[t] against t (time) or phase space?
    Instruction says: "Fit polynomial models of fixed max degree".
    Usually for time series discovery we might map t -> x(t) or x(t-1) -> x(t).
    Given "instability under small perturbations", mapping t -> x(t) is a common baseline
    to show how it fails for chaos (sensitivity to initial conditions vs functional fit).
    However, for dynamic laws, x(t-1) -> x(t) is better.

    Let's implement t -> x(t) as a curve fitting baseline which often fails for long term chaotic/stochastic.
    """
    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, time, values):
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
    def fit(self, time, values):
        # We need a reference "model". Let's assume a simple mean model or linear model
        # as the base, and compress the residuals.
        # Or simply compress the raw series vs a naive prediction.

        # Strategy: Use Linear Baseline first, then compress residuals.
        lin = BaselineLinear()
        res = lin.fit(time, values)

        if "error" in res:
             return {"error": res["error"]}

        residuals = np.array(res["residuals"])

        # Quantize residuals to standard precision for fair compression comparison
        # (e.g., 4 decimal places)
        res_str = json.dumps([round(x, 4) for x in residuals])
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
            "compression_ratio": (model_complexity + len(compressed_res)) / len(json.dumps(values))
        }
