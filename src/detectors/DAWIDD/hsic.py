# src/detectors/DAWIDD/hsic.py
import numpy as np
from scipy.stats import gamma


def rbf_dot(pattern1, pattern2, deg):
    """
    Computes the RBF (Gaussian) kernel between two datasets.

    Args:
        pattern1 (np.ndarray): First input array of shape (n_samples_1, n_features).
        pattern2 (np.ndarray): Second input array of shape (n_samples_2, n_features).
        deg (float): Kernel bandwidth parameter (sigma). Should be > 0.

    Returns:
        np.ndarray: RBF kernel matrix of shape (n_samples_1, n_samples_2).
    """

    # Ensure input are 2D arrays
    if pattern1.ndim == 1:
        pattern1 = pattern1.reshape(-1, 1)
    if pattern2.ndim == 1:
        pattern2 = pattern2.reshape(-1, 1)

    size1 = pattern1.shape
    size2 = pattern2.shape

    G = np.sum(pattern1 * pattern1, axis=1).reshape(size1[0], 1)
    H = np.sum(pattern2 * pattern2, axis=1).reshape(size2[0], 1)

    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))

    # Calculate squared Euclidean distances
    H_dist = Q + R - 2 * np.dot(pattern1, pattern2.T)
    H_dist = np.maximum(H_dist, 0)  # Ensure non-negative distances

    # RBF kernel calculation using deg (sigma)
    if deg <= 1e-10:  # Basic check to prevent division by zero
        print(f"Warning: RBF kernel sigma (deg) is near zero ({deg:.2e}). Using 1.0.")
        sigma_sq = 1.0
    else:
        sigma_sq = deg ** 2

    H_kernel = np.exp(-H_dist / (2 * sigma_sq))

    return H_kernel


def hsic_gam(X, Y):
    """
    Performs the HSIC independence test using Gamma approximation.

    Args:
        X (np.ndarray): First dataset, shape (n_samples, n_features).
        Y (np.ndarray): Second dataset (e.g., time), shape (n_samples, n_features).

    Returns:
        float: The estimated p-value for the independence test.
    """
    n = X.shape[0]

    # Ensure X and Y are 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Median heuristic for width (sigma) of X
    Xmed = X
    G = np.sum(Xmed * Xmed, axis=1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))
    dists_sq = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists_sq = np.maximum(dists_sq, 0)

    # Use upper triangle to find median pairwise distance
    upper_triangle_indices = np.triu_indices(n, k=1)
    dists_sq_vec = dists_sq[upper_triangle_indices]
    dists_sq_vec_pos = dists_sq_vec[dists_sq_vec > 1e-10]  # Avoid zeros

    width_x = 1.0  # Default value
    if len(dists_sq_vec_pos) > 0:
        median_dist_sq = np.median(dists_sq_vec_pos)

        width_x = np.sqrt(0.5 * median_dist_sq)
        if width_x < 1e-10:  # Check if width is too small
            width_x = 1.0  # Fallback if median is ~0

    # Median heuristic for width (sigma) of Y
    Ymed = Y
    G = np.sum(Ymed * Ymed, axis=1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))
    dists_sq = Q + R - 2 * np.dot(Ymed, Ymed.T)
    dists_sq = np.maximum(dists_sq, 0)
    upper_triangle_indices = np.triu_indices(n, k=1)
    dists_sq_vec = dists_sq[upper_triangle_indices]
    dists_sq_vec_pos = dists_sq_vec[dists_sq_vec > 1e-10]

    width_y = 1.0  # Default value
    if len(dists_sq_vec_pos) > 0:
        median_dist_sq = np.median(dists_sq_vec_pos)
        width_y = np.sqrt(0.5 * median_dist_sq)
        if width_y < 1e-10:
            width_y = 1.0

    # HSIC Calculation
    bone = np.ones((n, 1), dtype=float)  # Used later for mean/var calculation
    H = np.identity(n) - np.ones((n, n), dtype=float) / n  # Centering matrix

    # Use width_x/y (sigma) as the 'deg' argument for rbf_dot
    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)

    Kc = np.dot(np.dot(H, K), H)  # Centered kernel K
    Lc = np.dot(np.dot(H, L), H)  # Centered kernel L

    # Test Statistic
    testStat = np.sum(Kc.T * Lc) / n

    # Null Distribution Parameters
    # Variance calculation - direct copy from original snippet
    varHSIC = (Kc * Lc / 6) ** 2
    varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)
    varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    # Mean calculation
    K_mod = K - np.diag(np.diag(K))
    L_mod = L - np.diag(np.diag(L))
    muX = np.dot(np.dot(bone.T, K_mod), bone) / n / (n - 1) if n > 1 else 0
    muY = np.dot(np.dot(bone.T, L_mod), bone) / n / (n - 1) if n > 1 else 0
    mHSIC = (1 + muX * muY - muX - muY) / n

    # Check for potential issues before Gamma calculation
    if varHSIC <= 1e-10 or mHSIC <= 1e-10 or np.isnan(varHSIC) or np.isnan(mHSIC):
        raise UserWarning("HSIC mean or variance is invalid. Cannot compute Gamma threshold.")

    # Gamma parameters
    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * n / mHSIC

    if al <= 0 or bet <= 0 or np.isnan(al) or np.isnan(bet):
        raise UserWarning("Invalid Gamma parameters calculated. Cannot compute threshold.")

    p_value = 1 - gamma.cdf(testStat, a=al, scale=bet)
    return p_value


def hsic_permutation_test(X, Y, n_permutations=199):
    """
    Performs the HSIC independence test using permutation testing.

    Args:
        X (np.ndarray): First dataset, shape (n_samples, n_features).
        Y (np.ndarray): Second dataset (e.g., time), shape (n_samples, n_features).
        n_permutations (int): Number of permutations to use for null distribution estimation.

    Returns:
        float: The estimated p-value for the independence test.
    """
    n = X.shape[0]
    if n <= 1:
        print(f"Warning: Sample size n={n} is too small for permutation test.")
        return 1.0  # Cannot perform test, assume independent

    # Ensure X and Y are 2D
    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)

    # Median heuristic for width (sigma) of X
    Xmed = X
    G = np.sum(Xmed * Xmed, axis=1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))
    dists_sq = Q + R - 2 * np.dot(Xmed, Xmed.T)
    dists_sq = np.maximum(dists_sq, 0)
    upper_triangle_indices = np.triu_indices(n, k=1)
    dists_sq_vec = dists_sq[upper_triangle_indices]
    dists_sq_vec_pos = dists_sq_vec[dists_sq_vec > 1e-10]
    width_x = 1.0
    if len(dists_sq_vec_pos) > 0:
        median_dist_sq = np.median(dists_sq_vec_pos)
        width_x = np.sqrt(0.5 * median_dist_sq)
        if width_x < 1e-10: width_x = 1.0

    # Median heuristic for width (sigma) of Y
    Ymed = Y
    G = np.sum(Ymed * Ymed, axis=1).reshape(n, 1)
    Q = np.tile(G, (1, n))
    R = np.tile(G.T, (n, 1))
    dists_sq = Q + R - 2 * np.dot(Ymed, Ymed.T)
    dists_sq = np.maximum(dists_sq, 0)
    upper_triangle_indices = np.triu_indices(n, k=1)
    dists_sq_vec = dists_sq[upper_triangle_indices]
    dists_sq_vec_pos = dists_sq_vec[dists_sq_vec > 1e-10]
    width_y = 1.0
    if len(dists_sq_vec_pos) > 0:
        median_dist_sq = np.median(dists_sq_vec_pos)
        width_y = np.sqrt(0.5 * median_dist_sq)
        if width_y < 1e-10: width_y = 1.0

    # Centering Matrix
    H = np.identity(n) - np.ones((n, n), dtype=float) / n

    # Calculate Kernels and Centered Kernels for ORIGINAL data
    K = rbf_dot(X, X, width_x)
    L = rbf_dot(Y, Y, width_y)
    Kc = H @ K @ H
    Lc = H @ L @ H

    # Calculate ORIGINAL Test Statistic
    testStat_orig = np.sum(
        Kc * Lc) / n  # Biased version used before - stick to this? Let's keep previous for consistency

    # Permutation Loop
    count = 0

    # Generate permutations of indices for Y
    indices = np.arange(n)
    for i in range(n_permutations):
        # Shuffle Y by shuffling indices
        perm_indices = np.random.permutation(indices)
        Y_perm = Y[perm_indices, :]  # Permute rows of Y

        L_perm = rbf_dot(Y_perm, Y_perm, width_y)
        Lc_perm = H @ L_perm @ H

        # Calculate HSIC statistic for this permutation
        testStat_perm = np.sum(Kc * Lc_perm) / n

        if testStat_perm >= testStat_orig:
            count += 1

    # Calculate p-value
    p_value = (count + 1.0) / (n_permutations + 1.0)

    return p_value
