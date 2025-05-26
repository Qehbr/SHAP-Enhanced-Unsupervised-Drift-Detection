# src/shap_utils.py
import shap
import numpy as np
import pandas as pd


def select_shap_data(X_full, shap_subset_size):
    """
    Selects a subset of data for SHAP value computation.

    Args:
        X_full (np.ndarray): Full dataset to sample from.
        shap_subset_size (int): Number of instances to select. If <= 0 or >= len(X_full), returns full dataset.

    Returns:
        np.ndarray: Subsample of the dataset.
    """
    n_instances = X_full.shape[0]
    if shap_subset_size <= 0 or shap_subset_size >= n_instances:
        print(f"      Using full dataset ({n_instances} instances) for SHAP.")
        return X_full
    else:
        print(f"      Selecting random subset of {shap_subset_size} instances (out of {n_instances}) for SHAP.")
        random_indices = np.random.choice(n_instances, shap_subset_size, replace=False)
        return X_full[random_indices]


def get_shap_explainer(model, X_background):
    """
    Creates and returns a SHAP explainer appropriate for the model type.

    Args:
        model: Fitted model (e.g., from scikit-learn).
        X_background (pd.DataFrame or np.ndarray): Background data for SHAP initialization.
        model_type (str): One of 'tree', 'linear', 'kernel', or 'auto'. Defaults to 'auto'.

    Returns:
        shap.Explainer: Initialized SHAP explainer.
    """
    model_name = type(model).__name__.lower()
    is_classification = hasattr(model, 'classes_')

    if isinstance(X_background, np.ndarray):
        try:
            feature_names = model.feature_names_in_
        except AttributeError:
            feature_names = [f'f_{i}' for i in range(X_background.shape[1])]
        X_background = pd.DataFrame(X_background, columns=feature_names)

    if any(substring in model_name for substring in ['tree', 'randomforest', 'extratrees', 'xgb', 'lgbm']):
        return shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
    elif any(substring in model_name for substring in ['linear', 'logisticregression', 'linearregression']):
        return shap.LinearExplainer(model, masker=X_background)
    else:
        n_clusters = min(X_background.shape[0], 100)
        background_summary = X_background
        if n_clusters > 1:
            background_summary = shap.kmeans(X_background, n_clusters)

        # KernelExplainer needs the prediction function (predict_proba preferred for classification)
        if is_classification and hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict

        return shap.KernelExplainer(predict_fn, data=background_summary)


def calculate_shap_values(explainer, X_sample):
    """
    Computes SHAP values for a sample of data.

    Args:
        explainer (shap.Explainer): Initialized SHAP explainer.
        X_sample (pd.DataFrame or np.ndarray): Data to explain.

    Returns:
        shap.Explanation: SHAP explanation object.
    """
    if not isinstance(X_sample, pd.DataFrame) and isinstance(explainer, shap.explainers.Linear):
        if hasattr(explainer.masker, 'columns'):
            col_names = explainer.masker.columns
        else:
            col_names = [f'f_{i}' for i in range(X_sample.shape[1])]
        X_sample = pd.DataFrame(X_sample, columns=col_names)

    shap_values = explainer(X_sample)
    return shap_values


def select_important_features(shap_values, method='top_k', k=10, feature_names=None):
    """
    Identifies the most important features based on SHAP values.

    Args:
        shap_values (shap.Explanation or list or np.ndarray): SHAP values for the dataset.
        method (str): Feature selection strategy. Currently supports 'top_k'. Defaults to 'top_k'.
        k (int): Number of top features to return if method is 'top_k'.
        feature_names (list, optional): List of feature names corresponding to SHAP columns.

    Returns:
        tuple:
            - List[int]: Indices of the selected features.
            - List[str] or None: Names of the selected features, if names were provided.
    """
    if isinstance(shap_values, shap.Explanation):
        shap_vals_data = shap_values.values
    else:
        shap_vals_data = shap_values

    if isinstance(shap_vals_data, list):  # Multi-output case (classification)
        mean_abs_shap = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals_data], axis=0)
    elif isinstance(shap_vals_data,
                    np.ndarray) and shap_vals_data.ndim == 2:  # Single output case (regression or binary handled by explainer)
        mean_abs_shap = np.mean(np.abs(shap_vals_data), axis=0)
    elif isinstance(shap_vals_data,
                    np.ndarray) and shap_vals_data.ndim == 3:
        mean_abs_shap = np.mean(np.mean(np.abs(shap_vals_data), axis=0), axis=1)  # Mean over samples, then classes
    else:
        raise TypeError(
            f"Unsupported shap_values structure: {type(shap_vals_data)}, ndim: {getattr(shap_vals_data, 'ndim', None)}")

    if method == 'top_k':
        # Ensure k is not larger than the number of features
        k = min(k, len(mean_abs_shap))
        # indices_of_top_k = np.argpartition(mean_abs_shap, -k)[-k:]
        # selected_indices = indices_of_top_k[np.argsort(mean_abs_shap[indices_of_top_k])][::-1].tolist()
        indices_of_bottom_k = np.argpartition(mean_abs_shap, k)[:k]
        selected_indices = indices_of_bottom_k[np.argsort(mean_abs_shap[indices_of_bottom_k])].tolist()
        print(f"Selected Top {k} features (indices): {selected_indices}")
    else:
        # TODO - This is for future work with extending SHAP methods
        raise ValueError(f"Unsupported feature selection method: {method}")

    # Get names if provided
    selected_names = None
    if feature_names is not None:
        selected_names = [feature_names[i] for i in selected_indices]
        print(f"Selected feature names: {selected_names}")

    return selected_indices, selected_names
