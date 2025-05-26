# src/detectors/HDDDM/hdddm.py
import pandas as pd
import numpy as np
import math
from math import floor
from scipy.stats import t as t_distribution


class Distance:
    """
    Helper class to compute Hellinger distance between two probability distributions.
    """

    @staticmethod
    def hellinger_dist(P, Q):
        """
        Computes Hellinger distance between two probability distributions

        Args:
            P (dict): First probability distribution (value: probability).
            Q (dict): Second probability distribution (value: probability)

        Returns:
            float: Hellinger distance between P and Q.
        """
        diff = 0.0
        for k in P:
            diff += (math.sqrt(P.get(k, 0.0)) - math.sqrt(Q.get(k, 0.0))) ** 2
        return (1.0 / math.sqrt(2.0)) * math.sqrt(diff)


def discretizer(data, n_bins, method='equalsize'):
    """
    Discretizes a numerical pandas Series using equal-size or equal-quantile binning.

    Args:
        data (pd.Series): Continuous numerical data.
        n_bins (int): Number of bins to discretize into.
        method (str): Binning method ('equalsize' or 'equalquantile').

    Returns:
        pd.Series: Discretized categorical series.
    """
    if method == 'equalsize':
        b = pd.cut(data, bins=n_bins, labels=False,
                   include_lowest=True, duplicates='drop')
    elif method == 'equalquantile':
        b = pd.qcut(data, q=n_bins, labels=False, duplicates='drop')
    else:
        raise ValueError("method must be 'equalsize' or 'equalquantile'")
    return b.astype(str).astype('category')


def process_numerical_features(df_num, n_bins, method='equalsize'):
    """
    Discretizes all numerical columns in a DataFrame.

    Args:
        df_num (pd.DataFrame): DataFrame with only numerical columns.
        n_bins (int): Number of bins for discretization.
        method (str): Binning method.

    Returns:
        pd.DataFrame: Discretized version of df_num.
    """
    df_out = pd.DataFrame(index=df_num.index)
    for col in df_num.columns:
        df_out[col] = discretizer(df_num[col], n_bins, method)
    return df_out


def generate_proper_dic(series, union_vals):
    """
    Generates a probability distribution dictionary over provided values.

    Args:
        series (pd.Series): Categorical values to count.
        union_vals (set): Set of all possible categorical values.

    Returns:
        dict: Dictionary with frequencies normalized to probabilities.
    """
    props = series.value_counts(normalize=True)
    return {v: float(props.get(v, 0.0)) for v in union_vals}


class HDDDM:
    def __init__(self, hdddm_batch_size, gamma=1.0, alpha=None, n_bins=None, discretization_method='equalsize'):
        """
        Hellinger Distance Drift Detection Method (HDDDM).
        Uses statistical monitoring of distributional change based on Hellinger distance between
        discretized reference and incoming data batches. Optionally supports SHAP-style feature filtering.

        Exactly one of `gamma` or `alpha` must be specified.

        Args:
            hdddm_batch_size (int): Expected size of incoming batches.
            gamma (float, optional): Sensitivity multiplier for thresholding. Used if `alpha` is None.
            alpha (float, optional): Significance level (two-tailed). Used if `gamma` is None.
            n_bins (int, optional): Fixed number of bins for discretization. If None, computed from sqrt(n).
            discretization_method (str): Binning strategy; one of {'equalsize', 'equalquantile'}. Defaults to 'equalsize'.
        """

        # must specify exactly one of gamma or alpha
        if (gamma is None) == (alpha is None):
            raise ValueError("Specify exactly one of gamma or alpha.")

        self.gamma = gamma
        self.alpha = alpha
        self.hdddm_batch_size = hdddm_batch_size

        self.n_bins_config = n_bins
        self.discretization_method = discretization_method

        # distance helper
        self.dist_func = Distance().hellinger_dist

        # SHAP extension:
        self.feature_indices = None
        self.cols_to_monitor = None  # will become list of column‑names

        self.initialized = False

    def initialize(self, X_ref):
        """
        Initializes the detector with a reference dataset.

        Args:
            X_ref (pd.DataFrame or np.ndarray): Initial reference window.
        """
        if isinstance(X_ref, np.ndarray):
            X_ref = pd.DataFrame(X_ref)
        if X_ref.empty:
            raise ValueError("Initial reference cannot be empty.")

        # set up columns
        self.all_cols = list(X_ref.columns)

        self.num_cols = list(X_ref.select_dtypes(include=np.number).columns)
        self.cat_cols = [c for c in self.all_cols if c not in self.num_cols]

        # reference size and bin‐count
        self.n_reference_samples = X_ref.shape[0]
        self.n_bins = (self.n_bins_config if self.n_bins_config else floor(math.sqrt(self.n_reference_samples))) or 1

        # discretize reference
        self.reference_window_cat = self._prepare_data(X_ref)

        # initially self all indices for monitoring
        self.feature_indices = None
        self.cols_to_monitor = self.all_cols.copy()

        # reset stats
        self.t_denom = 0
        self.old_total_dist = 0.0
        self.epsilons = []
        self.drift_detected = False
        self.initialized = True

    def set_feature_indices(self, indices):
        """
        Selects a subset of features (by index) to monitor.

        Args:
            indices (list[int] or None): List of integer indices into all features. None resets to all.
        """

        if not self.initialized:
            raise RuntimeError("Call initialize() before setting feature indices.")

        if indices is None:
            self.feature_indices = None
            self.cols_to_monitor = self.all_cols.copy()
        else:
            for i in indices:
                if i < 0 or i >= len(self.all_cols):
                    raise IndexError(f"Feature index {i} out of bounds.")
            self.feature_indices = indices
            self.cols_to_monitor = [self.all_cols[i] for i in indices]
        print(f"HDDDM will monitor columns: {self.cols_to_monitor}")

    def _prepare_data(self, X_batch):
        """
        Discretizes numerical columns, preserves categorical ones.

        Args:
            X_batch (pd.DataFrame or np.ndarray): Input batch.

        Returns:
            pd.DataFrame: Aligned and discretized data.
        """
        if isinstance(X_batch, np.ndarray):
            X_batch = pd.DataFrame(X_batch, columns=self.all_cols)
        else:
            X_batch = X_batch.copy()
        num = X_batch[self.num_cols]
        cat = X_batch[self.cat_cols]
        if not num.empty:
            num_disc = process_numerical_features(
                num, self.n_bins, self.discretization_method
            )
            out = pd.concat([num_disc, cat], axis=1)
        else:
            out = cat
        return out[self.all_cols]

    def _calculate_distance_sum(self, ref_cat, cur_cat):
        """
        Computes average Hellinger distance across monitored features.

        Args:
            ref_cat (pd.DataFrame): Reference categorical data.
            cur_cat (pd.DataFrame): Current batch categorical data.

        Returns:
            float: Average Hellinger distance.
        """
        total = 0.0
        for feat in self.cols_to_monitor:
            union_vals = set(ref_cat[feat].unique()) | set(cur_cat[feat].unique())
            p = generate_proper_dic(ref_cat[feat], union_vals)
            q = generate_proper_dic(cur_cat[feat], union_vals)
            total += self.dist_func(p, q)
        return total / len(self.cols_to_monitor)

    def add_new_batch(self, X_new):
        """
        Adds a new batch and checks for drift. Resets reference if drift is detected.

        Args:
            X_new (pd.DataFrame): New batch of data.
        """
        if not self.initialized:
            raise RuntimeError("Call initialize() first.")
        n_new = X_new.shape[0]
        self.drift_detected = False

        # discretize
        cur_cat = self._prepare_data(X_new)

        # compute Hellinger
        cur_dist = self._calculate_distance_sum(self.reference_window_cat, cur_cat)

        if self.t_denom > 0:
            eps = abs(cur_dist - self.old_total_dist)
            self.epsilons.append(eps)

            mean_eps = np.mean(self.epsilons)
            sigma_eps = np.std(self.epsilons) if len(self.epsilons) > 1 else 0.0

            if self.gamma is not None:
                beta = mean_eps + self.gamma * sigma_eps
            else:
                df = len(self.epsilons) - 1
                if df > 0:
                    tval = t_distribution.ppf(1 - self.alpha / 2, df)
                    se = sigma_eps / math.sqrt(len(self.epsilons))
                    beta = mean_eps + tval * se
                else:
                    beta = np.inf

            if eps > beta:
                self.drift_detected = True
                print(f"HDDDM Drift Detected! (eps={eps:.4f} > beta={beta:.4f})")

                # reset reference in this batch
                self.n_bins = max(1, floor(math.sqrt(n_new)))
                self.reference_window_cat = self._prepare_data(X_new)
                self.n_reference_samples = n_new

                # reset stats
                self.t_denom = 0
                self.old_total_dist = 0.0
                self.epsilons = []
            else:
                # if no drift, then append to reference
                combined = pd.concat(
                    [self.reference_window_cat, cur_cat], ignore_index=True
                )
                self.reference_window_cat = combined
                self.n_reference_samples += n_new

        # update for next iteration
        if not self.drift_detected:
            self.old_total_dist = cur_dist
            self.t_denom += 1

    def detected_change(self):
        return self.drift_detected
