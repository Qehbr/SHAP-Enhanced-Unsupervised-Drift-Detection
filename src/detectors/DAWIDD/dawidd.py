# src/detectors/DAWIDD/dawidd.py
import numpy as np
from src.detectors.DAWIDD.lsit import test_independence


class DAWIDD:

    def __init__(self, max_window_size, min_window_size, alpha):
        """
        Dynamic Adapting Window Independence Drift Detector (DAWIDD).
        Maintains a dynamic window of data and uses independence testing (LSIT) to detect concept drift.
        SHAP-compatible for feature filtering

        Args:
            max_window_size (int): Maximum number of samples in the sliding window.
            min_window_size (int): Minimum number of samples required to start drift detection.
            alpha (float): Significance level (alpha) for the independence test.
        """

        if not (1 < min_window_size <= max_window_size):
            raise ValueError("Window sizes must satisfy 1 < min_window_size <= max_window_size")

        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.alpha = alpha  # threshold for independence test

        self.X = []
        self.n_items = 0
        self.drift_detected = False

        self.feature_indices = None  # SHAP support

    def set_feature_indices(self, indices):
        """
        Sets the feature indices for the independence test.

        Args:
            indices (list[int] or None): Indices to track in drift detection. If None, all features are used.
        """
        if indices is None:
            self.feature_indices = None
        else:
            if not all(isinstance(i, int) and i >= 0 for i in indices):
                raise ValueError("SHAP indices must be non-negative integers.")
            self.feature_indices = sorted(set(indices))

    def _test_for_independence(self):
        """
        Runs the underlying independence test on the current window.

        Returns:
            bool: True if the data and time are independent, False otherwise.
        """
        # time vector
        t = np.arange(self.n_items) / float(self.n_items)
        t = (t / np.std(t)).reshape(-1, 1)

        # data matrix
        X_arr = np.vstack(self.X)
        if self.feature_indices is not None:
            X_arr = X_arr[:, self.feature_indices]

        return test_independence(X_arr, t, self.alpha)

    def add_record(self, x):
        """
        Adds a new sample, tests for drift once enough data is collected,
        and adapts the window dynamically on overflow or when drift is detected.

        Args:
            x (np.ndarray): New data instance of shape (dim, ) or (1, dim).
        """
        self.drift_detected = False

        x_flat = np.asarray(x).flatten()
        self.X.append(x_flat)
        self.n_items += 1

        # overflow: keep window size at max
        if self.n_items > self.max_window_size:
            idx = np.random.randint(0, self.n_items)
            self.X.pop(idx)
            self.n_items -= 1

        # drift detection: test on every new sample once enough data
        if self.n_items >= self.min_window_size:
            is_indep = self._test_for_independence()
            if not is_indep:
                # drift flagged
                self.drift_detected = True

                # shrink window iteratively until independence is restored or min size reached
                while not is_indep and self.n_items >= self.min_window_size:
                    self.X.pop(0)
                    self.n_items -= 1
                    is_indep = self._test_for_independence()

    def detected_change(self):
        """
        Indicates whether drift was detected during the last `add_record` call.

        Returns:
            bool: True if drift was detected, False otherwise.
        """

        return self.drift_detected
