# src/detectors/DAWIDD/lsit.py
import numpy as np

from src.detectors.DAWIDD.hsic import hsic_gam


def test_independence(X, Y, alpha):
    """
    Performs an independence test using HSIC with Gamma approximation.

    Args:
        X (array-like): First dataset, shape (n_samples, n_features). Will be converted to 2D if needed.
        Y (array-like): Second dataset (e.g., time or labels), shape (n_samples, n_features). Will be converted to 2D if needed.
        alpha (float): Significance level for the test. The null hypothesis (independence) is rejected if p-value <= alpha.

    Returns:
        bool: True if X and Y are independent (p-value > alpha), False if dependent.
    """

    # Ensure X and Y are 2D numpy arrays
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr.reshape(-1, 1)

    p_value = hsic_gam(X_arr, Y_arr)
    return p_value > alpha
