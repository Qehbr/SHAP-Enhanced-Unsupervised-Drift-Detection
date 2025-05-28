# src/data_loader.py
import pandas as pd
import numpy as np
from skmultiflow.data.data_stream import DataStream
from sklearn.preprocessing import LabelEncoder
from src.utils.drift_injection import inject_drift


def read_data_electricity_market(dataset_path):
    """
    Reads and processes the Electricity Market dataset.

    Args:
        dataset_path (str): Path to the CSV file.

    Returns:
        tuple: Features (X) as DataFrame and encoded labels (y) as ndarray.
    """
    df = pd.read_csv(dataset_path, dtype=str, encoding='utf-8')

    # Separate features and label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Cleaning helper for numeric features
    def clean_float(val):
        try:
            if isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
                val = val[2:-1]
            return float(val)
        except Exception as e:
            print(f"Error converting value: {val}")
            raise e

    # Cleaning helper for labels
    def clean_label(val):
        if isinstance(val, str) and val.startswith("b'") and val.endswith("'"):
            val = val[2:-1]
        return val.strip()

    X = X.applymap(clean_float)
    y = y.apply(clean_label)

    # Encode labels
    label = ["UP", "DOWN"]
    le = LabelEncoder()
    le.fit(label)
    y_encoded = le.transform(y)

    return X, y_encoded


def read_data_weather(data_path, class_path):
    """
    Reads and merges weather features and class labels from separate files.

    Args:
        data_path (str): Path to the feature data CSV file.
        class_path (str): Path to the class labels CSV file.

    Returns:
        tuple: Features (X) as DataFrame and labels (y) as Series.
    """
    df_labels = pd.read_csv(class_path, header=None)
    y = df_labels.values.flatten()

    df_data = pd.read_csv(data_path, header=None)
    df = df_data.copy()
    df['y'] = y

    X = df.iloc[:, :-1]
    y = df['y']

    return X, y


def read_data_forest_cover_type(data_path):
    """
    Reads the Forest Cover Type dataset and selects relevant features.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        tuple: Features (X) as DataFrame and labels (y) as ndarray.
    """
    df = pd.read_csv(data_path)

    X = df.drop('class', axis=1)
    y = df['class']

    # Select only the first 10 numerical + 4 wilderness area features
    X = X.iloc[:, 0:10]
    y = y.values.flatten()

    return X, y


def load_data_stream(dataset_conf, drift=False, min_drift_point=0.7, max_drift_point=0.9):
    """
    Loads a dataset and converts it into a stream-compatible format.
    Optionally injects synthetic concept drift.

    Args:
        dataset_conf (dict): Dictionary containing dataset metadata and file paths.
            Keys: 'name' (str), and one or more of:
                - 'file_path' (for electricity/forest)
                - 'file_path_data', 'file_path_class' (for weather)
        drift (bool): Whether to inject drift using `inject_drift`.
        min_drift_point (float): Minimum fraction of data after which drift may start.
        max_drift_point (float): Maximum fraction of data after which drift may start.

    Returns:
        tuple:
            - pd.DataFrame: Features.
            - pd.Series: Labels.
            - DataStream: skmultiflow DataStream object.
            - np.ndarray: Binary array indicating drifted rows.
            - int or float: Row index where drift started, or np.nan if no drift.
    """
    drift_start_point = np.nan
    name = dataset_conf['name']

    if name == 'electricity':
        X, y = read_data_electricity_market(dataset_conf['file_path'])
        is_classification = True
    elif name == 'weather' or name == 'constant' or name == 'prob' or name == 'drift_important' or name == 'drift_unimportant':
        X, y = read_data_weather(dataset_conf['file_path_data'], dataset_conf['file_path_class'])
        is_classification = True
    elif name == 'forest':
        X, y = read_data_forest_cover_type(dataset_conf['file_path'])
        is_classification = True
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    if drift:
        print(f"Injecting drift into {name} dataset...")
        X, y, drift_start_point = inject_drift(X, pd.Series(y),  # inject_drift expects Series for y
                                               min_point=min_drift_point,
                                               max_point=max_drift_point,
                                               classification=is_classification)

        drifted_rows = X['drifted'].values
        X = X.drop(columns=['drifted'])
    else:
        drifted_rows = np.zeros(X.shape[0])  # No drift injected

    y_stream = y.values if isinstance(y, pd.Series) else y
    stream = DataStream(X, y_stream)

    return X, pd.Series(y_stream), stream, drifted_rows, drift_start_point
