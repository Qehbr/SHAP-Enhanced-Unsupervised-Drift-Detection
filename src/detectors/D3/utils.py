# src/detectors/D3/utils.py
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import StandardScaler

# Default Logistic Regression Classifier for D3
default_d3_classifier = LogisticRegression(solver='liblinear', max_iter=1000)


def d3_auc_drift_check(S, T, threshold, feature_indices=None, classifier=None):
    """
    Compares Source (S) and Target (T) data distribution using a classifier.
    Returns True if drift is detected (AUC > threshold), False otherwise.
    (This is the core check function used internally by the D3 class).

    Args:
        S (np.ndarray):
        T (np.ndarray): Target data (current window).
        threshold (float): AUC threshold for drift detection.
        feature_indices (list, optional): Indices of features to use. If None, use all.
        classifier (sklearn classifier, optional): Classifier for discrimination. Defaults to LogisticRegression defined above.

    Returns:
        bool: True if drift detected, False otherwise.
    """

    if S.shape[0] == 0 or T.shape[0] == 0:
        raise ValueError('S and T must have at least one sample.')

    if classifier is None:
        classifier = default_d3_classifier

    # If shap based features
    if feature_indices is not None:
        if not feature_indices:  # Handle empty list explicitly
            raise ValueError("feature_indices cannot be None")

        max_idx = max(feature_indices)
        if max_idx >= S.shape[1] or max_idx >= T.shape[1]:
            raise ValueError(f"Feature index {max_idx} out of bounds (S shape: {S.shape}, T shape: {T.shape}).")

        S_filtered = S[:, feature_indices]
        T_filtered = T[:, feature_indices]
        if S_filtered.shape[1] == 0:
            raise ValueError("No features selected by SHAP for D3 check. Skipping.")
    else:
        S_filtered = S
        T_filtered = T

    # Labels: 1 for Source (old), 0 for Target (new)
    labels_S = np.ones(S_filtered.shape[0])
    labels_T = np.zeros(T_filtered.shape[0])

    ST = np.vstack((T_filtered, S_filtered))
    labels = np.concatenate((labels_T, labels_S))
    scaler = StandardScaler()
    ST_scaled = scaler.fit_transform(ST)

    # Classification
    probs = np.zeros(labels.shape)
    skf = StratifiedKFold(n_splits=2, shuffle=True)

    for train_idx, test_idx in skf.split(ST_scaled, labels):
        X_train, X_test = ST_scaled[train_idx], ST_scaled[test_idx]
        y_train = labels[train_idx]
        clf = copy.deepcopy(classifier)
        clf.fit(X_train, y_train)
        probs[test_idx] = clf.predict_proba(X_test)[:, 1]

    # Get AUC score
    auc_score = AUC(labels, probs)

    drift_detected = auc_score > threshold
    return drift_detected
