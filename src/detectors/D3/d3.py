# src/detectors/D3/d3.py
import numpy as np
from src.detectors.D3.utils import d3_auc_drift_check


class D3:

    def __init__(self, dim, w=100, rho=0.1, auc=0.7, classifier=None):
        """
        Domain Discrimination Drift Detection (D3) - Class Implementation.
        Manages reference (W) and recent (R) data windows and uses an AUC-based check function to detect drift.

        Args:
        dim (int): The number of features in the input data.
        w (int): Size of the reference window (W).
                 Defaults to 100.
        rho (float): Ratio determining the size of the recent window (R = W * rho).
                     Defaults to 0.1.
        auc (float): AUC threshold for drift detection. Defaults to 0.7.
        classifier (sklearn classifier, optional): Classifier passed to the check function.
                                                   Defaults to Logistic Regression.
        """

        if w <= 0 or rho <= 0:
            raise ValueError("Window size (w) and ratio (rho) must be positive.")
        if dim <= 0:
            raise ValueError("Dimension (dim) must be positive.")

        self.dim = dim
        self.w = w
        self.rho = rho
        self.size_r = max(1, int(self.w * self.rho))  # R size is at least 1
        self.size = self.w + self.size_r  # Total buffer size W + R

        self.win_data = np.zeros((self.size, dim))  # Data buffer
        self.auc_threshold = auc
        self.classifier = classifier  # Classifier for the check function

        self.drift_count = 0
        self.window_index = 0  # Tracks how much of the entire buffer is filled
        self.ready_to_check = False  # Flag: True when R window part is full

        self.feature_indices = None  # For SHAP

    def set_feature_indices(self, indices):
        """
        Sets the feature indices for the drift check function.

        Args:
            indices (list[int]): Indices of features to track in drift detection.
        """
        self.feature_indices = indices
        print(f"D3 using feature indices: {self.feature_indices}")

    def add_initial_reference(self, X_ref):
        """
        Fills the initial reference window (W) part of the buffer.

        Args:
            X_ref (np.ndarray): Initial reference window data with shape (w, dim).
        """
        if X_ref.shape[0] != self.w:
            raise ValueError(f"Initial reference data size mismatch: expected {self.w}, got {X_ref.shape[0]}")
        if X_ref.shape[1] != self.dim:
            raise ValueError(f"Initial reference data dimension mismatch: expected {self.dim}, got {X_ref.shape[1]}")

        self.win_data[:self.w] = X_ref
        self.window_index = self.w  # Buffer filled up to W
        self.ready_to_check = False  # Need to fill R next

    def add_record(self, x):
        """
        Adds a new data instance (1D array) and manages the buffer.

        Args:
            x (np.ndarray): New data instance of shape (dim, ) or (1, dim).
        """
        x_flat = x.flatten()

        # Still filling the buffer initially (W then R)
        if self.window_index < self.size:
            self.win_data[self.window_index] = x_flat
            self.window_index += 1

            # Check if R part just became full
            if self.window_index == self.size:
                self.ready_to_check = True
        # Buffer full: Slide W by R samples, insert new sample into R section
        else:
            self.win_data = np.roll(self.win_data, -self.size_r, axis=0)
            # Insert new sample at the start of the now-empty R section (index w)
            self.win_data[self.w] = x_flat
            # Reset index to start filling the R section again
            self.window_index = self.w + 1
            self.ready_to_check = False  # R section is not full yet

    def detected_change(self):
        """
        Checks for drift if the R window is full.

        Returns:
            bool: True if drift detected, False otherwise.
        """
        if not self.ready_to_check:
            return False

        detected = d3_auc_drift_check(
            S=self.win_data[:self.w],  # Reference W
            T=self.win_data[self.w:],  # Recent R
            threshold=self.auc_threshold,
            feature_indices=self.feature_indices,
            classifier=self.classifier
        )
        self.ready_to_check = False  # Checked performed, reset readiness

        if detected:
            self.drift_count += 1
            print(f"D3 Drift Detected! (AUC > {self.auc_threshold})")
            self.win_data = np.roll(self.win_data, -self.w, axis=0)  # Shift R to start

            # Reset index to start filling the now empty R part
            self.window_index = self.size_r
            self.ready_to_check = False  # Need to fill empty part
            return True
        else:
            # No drift. Buffer sliding is handled in add_record.
            return False
