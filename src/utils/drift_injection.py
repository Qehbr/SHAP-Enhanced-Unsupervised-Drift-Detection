# src/utils/drift_injection.py
import numpy as np
import pandas as pd
import random


def num_cols(df):
    """
    Identify indices of numerical columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        np.ndarray: Array of column indices with numeric data types.
    """
    num_indices = [i for i, col in enumerate(df.columns) if pd.api.types.is_numeric_dtype(df[col])]
    return np.array(num_indices)


def create_drift_points(X, min_point=0.7, max_point=0.9):
    """
    Determines the row index where drift should start and selects pairs of numerical columns to swap.

    Args:
        X (pd.DataFrame): Input features.
        min_point (float): Minimum relative drift start location (0 to 1). Defaults to 0.7.
        max_point (float): Maximum relative drift start location (0 to 1). Defaults to 0.9.

    Returns:
        dict: Dictionary with 'row' (drift start index) and 'cols' (list of column indices to swap).
    """
    driftpoint_perc = np.random.uniform(min_point, max_point)
    driftpoint = int(driftpoint_perc * X.shape[0])
    num_ids = num_cols(X)

    if len(num_ids) < 2:
        print("Warning: Not enough numerical columns to swap for drift injection.")
        return {"row": driftpoint, "cols": []}

    # Ensure we select an even number of columns to swap in pairs
    l = len(num_ids) // 2
    num_to_sample = l * 2
    if num_to_sample == 0 and len(num_ids) >= 2:
        num_to_sample = 2
    elif num_to_sample == 0:
        return {"row": driftpoint, "cols": []}

    ids = random.sample(list(num_ids), num_to_sample)
    dpoints = {"row": driftpoint, "cols": ids}
    return dpoints


def swap_columns(X, y, selected_cols, starting_row, classification=True):
    """
    Injects drift into selected columns by swapping values post-drift point for a specific class (classification)
    or target range (regression).

    Args:
        X (pd.DataFrame): Input features.
        y (Union[pd.Series, np.ndarray]): Target variable (for class or regression-based conditioning).
        selected_cols (list[int]): Indices of columns to swap (must be even).
        starting_row (int): Row index at which drift should start.
        classification (bool): Whether the task is classification (True) or regression (False).

    Returns:
        pd.DataFrame: Drift-injected feature matrix with a new 'drifted' column.
    """
    if not selected_cols:  # If no columns selected
        X['drifted'] = 0
        return X

    if classification:
        unique_classes = list(np.unique(y))
        selected_class = random.choice(unique_classes)
        print(f"Injecting drift, swapping cols {selected_cols} for class {selected_class} starting row {starting_row}")

    else:  # Regression
        bins = pd.qcut(y.iloc[starting_row:], q=10, duplicates='drop')
        value_counts = bins.value_counts()
        good_bins = value_counts[value_counts >= 10].index.tolist()
        if not good_bins:
            print(
                "Warning: Could not find suitable bins in regression target for drift injection. Using random sample.")
            selected_class = None
        else:
            selected_class = random.choice(good_bins)
        print(
            f"Injecting drift by swapping cols {selected_cols} for target range {selected_class} starting row {starting_row}")

    df = X.copy()
    df['target'] = y.values if isinstance(y, pd.Series) else y

    df_before = df.iloc[:starting_row, :].copy()
    df_after = df.iloc[starting_row:, :].copy()

    df_before['drifted'] = 0

    if len(selected_cols) % 2 != 0:
        print(f"Warning: Odd number of columns selected for swapping ({len(selected_cols)}). Skipping last column.")
        selected_cols = selected_cols[:-1]

    column_pairs = list(zip(selected_cols[::2], selected_cols[1::2]))

    df_after['drifted'] = 0

    col_names = X.columns

    for idx in df_after.index:
        apply_swap = False
        current_target = df_after.loc[idx, 'target']

        if classification:
            if current_target == selected_class:
                apply_swap = True
        elif selected_class is not None:  # Regression with targeted bin
            if current_target in selected_class:
                apply_swap = True
        else:  # Regression without specific target (or if binning failed)
            apply_swap = True

        if apply_swap:
            df_after.loc[idx, 'drifted'] = 1
            for col_idx1, col_idx2 in column_pairs:
                col_name1 = col_names[col_idx1]
                col_name2 = col_names[col_idx2]

                val1 = df_after.loc[idx, col_name1]
                val2 = df_after.loc[idx, col_name2]
                df_after.loc[idx, col_name1] = val2
                df_after.loc[idx, col_name2] = val1

    # Combine parts and drop temporary target column
    result_df = pd.concat([df_before, df_after], ignore_index=False)
    result_df = result_df.drop(columns=['target'])

    return result_df


def inject_drift(X, y, min_point=0.7, max_point=0.9, classification=True):
    """
    Main function to inject feature drift into a dataset based on target conditioning.

    Args:
        X (pd.DataFrame): Input feature data.
        y (Union[pd.Series, np.ndarray]): Target labels or values.
        min_point (float): Minimum percentage of data to skip before injecting drift. Defaults to 0.7.
        max_point (float): Maximum percentage of data to skip before injecting drift. Defaults to 0.9.
        classification (bool): If True, inject drift conditioned on class labels. Otherwise, use regression bins.

    Returns:
        tuple:
            - pd.DataFrame: Drifted feature set with 'drifted' indicator column.
            - pd.Series or np.ndarray: Unchanged target values.
            - int: Row index at which drift began.
    """
    d_point = create_drift_points(X, min_point, max_point)
    if not d_point['cols']:
        X_drifted = X.copy()
        X_drifted['drifted'] = 0
        return X_drifted, y, d_point["row"]

    X_drifted = swap_columns(X, y, d_point['cols'], d_point['row'], classification)
    return X_drifted, y, d_point["row"]
