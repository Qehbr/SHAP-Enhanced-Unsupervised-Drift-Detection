# src/simulation.py
import os
import random
import time
import numpy as np
import pandas as pd
import yaml
from copy import deepcopy
import gc
from src.data_loader import load_data_stream
from src.models.utils import get_model_instance
from src.shap_utils import get_shap_explainer, calculate_shap_values, select_important_features, select_shap_data
from src.detectors.D3.d3 import D3
from src.detectors.DAWIDD.dawidd import DAWIDD
from src.detectors.HDDDM.hdddm import HDDDM
from src.detectors.STUDD.student_teacher import StudentTeacherDriftDetector


def define_scenarios(config):
    """
    Defines all baseline and detector scenarios based on the configuration file.

    Args:
        config (dict): Parsed YAML configuration dict.

    Returns:
        dict: Scenario dictionary mapping scenario name to configuration details.
    """
    scenario_definitions = {}
    detector_config = config.get('detectors', {})
    shap_glob_enabled = config.get('shap_params', {}).get('enabled', False)

    print("Defining detector scenarios...")

    # Baseline
    scenario_definitions['baseline'] = {'detector_type': None, 'params': {}, 'is_shap': False}
    print("  Defined baseline scenario.")

    # D3
    d3_conf = detector_config.get('d3', {})
    if d3_conf.get('enabled', False):
        params = d3_conf.get('params', {})
        scenario_definitions['D3_standard'] = {'detector_type': D3, 'params': params, 'is_shap': False}
        print("  Defined D3_standard scenario.")
        if d3_conf.get('shap_variant', {}).get('enabled', False) and shap_glob_enabled:
            scenario_definitions['D3_shap'] = {'detector_type': D3, 'params': params, 'is_shap': True}
            print("  Defined D3_shap scenario.")

    # Dawidd
    dawidd_conf = detector_config.get('dawidd', {})
    if dawidd_conf.get('enabled', False):
        params = dawidd_conf.get('params', {})
        scenario_definitions['Dawidd_standard'] = {'detector_type': DAWIDD, 'params': params, 'is_shap': False}
        print("  Defined Dawidd_standard scenario.")
        if dawidd_conf.get('shap_variant', {}).get('enabled', False) and shap_glob_enabled:
            scenario_definitions['Dawidd_shap'] = {'detector_type': DAWIDD, 'params': params, 'is_shap': True}
            print("  Defined Dawidd_shap scenario.")

    # HDDDM
    hdddm_conf = detector_config.get('hdddm', {})
    if hdddm_conf.get('enabled', False):
        params = hdddm_conf.get('params', {})
        scenario_definitions['HDDDM_standard'] = {'detector_type': HDDDM, 'params': params, 'is_shap': False}
        print("  Defined HDDDM_standard scenario.")
        if hdddm_conf.get('shap_variant', {}).get('enabled', False) and shap_glob_enabled:
            scenario_definitions['HDDDM_shap'] = {'detector_type': HDDDM, 'params': params, 'is_shap': True}
            print("  Defined HDDDM_shap scenario.")

    # Student-Teacher
    st_conf = detector_config.get('student_teacher', {})
    if st_conf.get('enabled', False):
        params = st_conf.get('params', {})
        scenario_definitions['ST_standard'] = {'detector_type': StudentTeacherDriftDetector, 'params': params,
                                               'is_shap': False}
        print("  Defined ST_standard scenario.")
        if st_conf.get('shap_variant', {}).get('enabled', False) and shap_glob_enabled:
            scenario_definitions['ST_shap'] = {'detector_type': StudentTeacherDriftDetector, 'params': params,
                                               'is_shap': True}
            print("  Defined ST_shap scenario.")

    return scenario_definitions


def update_detector_shap_features(detector, is_shap_scenario, indices):
    """
    Updates detector with selected SHAP features if the scenario is SHAP-enabled.

    Args:
        detector: Drift detector instance.
        is_shap_scenario (bool): Whether the scenario uses SHAP-based filtering.
        indices (list[int]): Feature indices to be tracked.
    """
    if not is_shap_scenario or detector is None or indices is None:
        return  # Only update SHAP variants with valid indices

    print(f"  Updating SHAP features for {type(detector).__name__}...")
    detector.set_feature_indices(indices)


def reset_detector_state(scenario_state, main_model_wrapper, X_ref, feature_names):
    """
    Resets and reinitializes the detector after retraining due to detected drift.

    Args:
        scenario_state (dict): Contains current scenario info and detector instance.
        main_model_wrapper: The retrained main model.
        X_ref (np.ndarray or pd.DataFrame): Reference data to reinitialize detector.
        feature_names (list[str]): Feature names for DataFrame reconstruction.
    """
    detector = scenario_state['detector']
    scenario_name = scenario_state['name']
    shap_indices = scenario_state.get('important_feature_indices')  # Get current indices if SHAP scenario
    original_dim = scenario_state['original_dim']

    if detector is None: return  # Baseline scenario

    print(f"  Resetting detector state for {scenario_name}...")

    # Ensure X_ref is numpy array
    if isinstance(X_ref, pd.DataFrame):
        X_ref_np = X_ref.values
    elif isinstance(X_ref, list):
        X_ref_np = np.array(X_ref)
    else:
        X_ref_np = X_ref

    # D3
    if isinstance(detector, D3):
        w = detector.w
        if X_ref_np.shape[0] >= w:
            ref_data = X_ref_np[-w:]
            params = {'w': detector.w, 'rho': detector.rho, 'auc': detector.auc_threshold,
                      'classifier': detector.classifier}
            scenario_state['detector'] = D3(dim=original_dim, **params)
            scenario_state['detector'].add_initial_reference(ref_data)
            if scenario_state['is_shap'] and shap_indices is not None:
                update_detector_shap_features(scenario_state['detector'], True, shap_indices)
        else:
            print(f"Warning: Not enough data for D3 reset ref window ({X_ref_np.shape[0]} < {w}) for {scenario_name}")
    # DAWIDD
    elif isinstance(detector, DAWIDD):
        params = {'max_window_size': detector.max_window_size, 'min_window_size': detector.min_window_size,
                  'alpha': detector.alpha}
        scenario_state['detector'] = DAWIDD(**params)
        if scenario_state['is_shap'] and shap_indices is not None:
            update_detector_shap_features(scenario_state['detector'], True, shap_indices)
    # HDDDM
    elif isinstance(detector, HDDDM):
        if X_ref_np.shape[0] > 0:  # Need some data to initialize
            ref_df = pd.DataFrame(X_ref_np, columns=feature_names if X_ref_np.shape[1] == len(feature_names) else None)
            params = {'gamma': detector.gamma, 'alpha': detector.alpha, 'n_bins': detector.n_bins_config,
                      'discretization_method': detector.discretization_method,
                      'hdddm_batch_size': detector.hdddm_batch_size}
            scenario_state['detector'] = HDDDM(**params)
            scenario_state['detector'].initialize(ref_df)
            if scenario_state['is_shap'] and shap_indices is not None:
                update_detector_shap_features(scenario_state['detector'], True, shap_indices)
        else:
            print(f"Warning: Not enough data for HDDDM reset for {scenario_name}")
    # Student-Teacher
    elif isinstance(detector, StudentTeacherDriftDetector):
        if scenario_state['is_shap'] and shap_indices is not None:
            update_detector_shap_features(detector, True, shap_indices)
        detector.update_student(main_model_wrapper, X_ref_np)
        # ADWIN is reset within update_student
    else:
        print(f"Warning: Unknown detector type {type(detector)} during reset for {scenario_name}")


def run_simulation(config_path="config.yaml"):
    """
    Runs the full simulation pipeline:
    - Loads datasets and models.
    - Initializes and executes scenarios with or without SHAP.
    - Monitors for drift and performs retraining.
    - Returns collected results.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        tuple:
            - master_results (list[dict]): A list of dictionaries summarizing each scenario run.
            - config (dict): The parsed configuration used for this run.
    """
    # Load Config
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded.")
    shap_config = config.get('shap_params', {})

    seed = config['random_seed']
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    shap_glob_enabled = shap_config.get('enabled', False)
    recalc_shap = shap_config.get('recalculate_on_retrain', True)
    sel_method = shap_config.get('feature_selection_method', 'top_k')

    master_results = []

    # Loop through Datasets
    for dataset_conf in config['datasets']:
        dataset_name = dataset_conf['name']
        k_features = dataset_conf.get('shap_k', -1)
        initial_train_size = dataset_conf['initial_train_size']
        retrain_window_size = dataset_conf['retrain_window_size']
        drift = dataset_conf.get('drift', False)
        shap_subset_size = dataset_conf.get('shap_subset_size', -1)

        print(f"\n--- Starting Dataset: {dataset_name} ---")
        try:
            X_all, y_all, stream, _, _ = load_data_stream(dataset_conf, drift=drift)
            feature_names = list(X_all.columns)
            n_features = X_all.shape[1]
            n_total_instances = X_all.shape[0]
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}. Skipping.")
            continue

        # Loop through Models
        for model_conf in config['models']:
            model_name = model_conf['name']
            model_params = model_conf.get('params', {})
            print(f"\n  --- Model: {model_name} ---")

            # Perform setup once per Dataset/Model
            if n_total_instances < initial_train_size:
                print(
                    f"Warning: Dataset size {n_total_instances} < initial_train_size {initial_train_size}. Skipping model.")
                continue

            X_train_init_df = X_all.iloc[:initial_train_size]
            y_train_init_series = y_all.iloc[:initial_train_size]
            X_train_init = X_train_init_df.values
            y_train_init = y_train_init_series.values

            initial_base_model_wrapper = get_model_instance(model_name, model_params, seed=seed)
            print("Training initial base model...")
            initial_base_model_wrapper.fit(X_train_init, y_train_init)

            initial_important_indices = None
            if shap_glob_enabled:
                print("Calculating initial SHAP features...")
                X_shap_init_data = select_shap_data(X_train_init, shap_subset_size)
                base_model_for_shap = initial_base_model_wrapper.get_model()
                shap_explainer = get_shap_explainer(base_model_for_shap, X_shap_init_data)
                shap_values = calculate_shap_values(shap_explainer, X_shap_init_data)  # Use same data for calculation
                initial_important_indices, _ = select_important_features(shap_values, method=sel_method, k=k_features,
                                                                         feature_names=feature_names)
                del shap_explainer, shap_values, base_model_for_shap, X_shap_init_data
                gc.collect()

            # Get Scenario Definitions
            scenario_definitions = define_scenarios(config)

            # Iterate through each scenario
            for scenario_name, scenario_conf in scenario_definitions.items():
                print(f"\n    --- Running Scenario: {scenario_name} ---")
                scenario_start_time = time.time()  # Start timer for this scenario
                stream.restart()
                gc.collect()  # Restart stream for each scenario

                is_shap_scenario = scenario_conf['is_shap']
                if is_shap_scenario and not shap_glob_enabled: print(
                    "      SHAP disabled globally, skipping SHAP scenario."); continue
                if is_shap_scenario and initial_important_indices is None:
                    print(f"      Initial SHAP failed, skipping SHAP scenario {scenario_name}.")
                    continue

                print(f"      Cloning initial model for {scenario_name}...")
                ml_model_wrapper = deepcopy(initial_base_model_wrapper)

                detector_type = scenario_conf['detector_type']
                detector_params = scenario_conf['params']
                detector = None
                important_feature_indices = deepcopy(initial_important_indices) if is_shap_scenario else None

                # Initialize the specific detector instance for this scenario
                if detector_type:
                    print(f"      Initializing detector {detector_type.__name__}...")
                    if detector_type == D3:
                        w = detector_params['w']
                        if X_train_init.shape[0] >= w:
                            ref_data = X_train_init[-w:]
                            detector = D3(dim=n_features, **detector_params)
                            detector.add_initial_reference(ref_data)
                        else:
                            print("Warning: Cannot init D3, data too small")
                            continue
                    elif detector_type == DAWIDD:
                        detector = DAWIDD(**detector_params)
                    elif detector_type == HDDDM:
                        ref_size = dataset_conf['retrain_window_size']
                        if X_train_init.shape[0] >= ref_size:
                            ref_data = X_train_init[-ref_size:]
                            ref_df = pd.DataFrame(ref_data, columns=feature_names)
                            detector = HDDDM(**detector_params)
                            detector.initialize(ref_df)
                        else:
                            print("Warning: Cannot init HDDDM, data too small")
                            continue
                    elif detector_type == StudentTeacherDriftDetector:
                        detector = StudentTeacherDriftDetector(**detector_params)
                        # Set SHAP indices before initial student training
                        if is_shap_scenario and important_feature_indices is not None:
                            update_detector_shap_features(detector, True, important_feature_indices)
                        # Train the initial student model using the combined method
                        detector.update_student(ml_model_wrapper, X_train_init)

                    else:
                        print(f"Warning: Unknown detector type for {scenario_name}")
                        continue
                    if is_shap_scenario and not isinstance(detector, StudentTeacherDriftDetector):
                        update_detector_shap_features(detector, True, important_feature_indices)

                # Scenario-specific State Variables
                accuracy_history = []
                retraining_points = []
                correct_count = 0
                total_count = 0
                recent_data_window_X = list(X_train_init[-retrain_window_size:])
                recent_data_window_y = list(y_train_init[-retrain_window_size:])
                current_batch_X = []
                current_batch_y = []
                hdddm_batch_size = config.get('detectors', {}).get('hdddm', {}).get('params', {}).get(
                    'hdddm_batch_size', retrain_window_size)

                print(f"      Starting stream processing for {scenario_name}...")
                instance_counter = initial_train_size
                stream.restart()
                stream.next_sample(initial_train_size)  # Reset stream position

                while stream.has_more_samples():
                    X_instance_raw, y_instance = stream.next_sample()
                    if X_instance_raw is None: break

                    if isinstance(X_instance_raw, pd.DataFrame):
                        X_instance_np = X_instance_raw.values
                    elif isinstance(X_instance_raw, pd.Series):
                        X_instance_np = X_instance_raw.values.reshape(1, -1)
                    else:
                        X_instance_np = X_instance_raw.reshape(1, -1) if X_instance_raw.ndim == 1 else X_instance_raw
                    y_instance_scalar = y_instance[0] if isinstance(y_instance, (np.ndarray, list)) else y_instance

                    # Predict
                    pred_result = ml_model_wrapper.predict(X_instance_np)
                    prediction = pred_result[0] if isinstance(pred_result, (np.ndarray, list)) else pred_result
                    is_correct_flag = bool(prediction == y_instance_scalar)
                    correct_count += is_correct_flag
                    total_count += 1

                    # Track Accuracy
                    current_accuracy = correct_count / total_count if total_count > 0 else 0.0
                    accuracy_history.append((instance_counter, current_accuracy))

                    # Update Retraining Window
                    recent_data_window_X.append(X_instance_np[0])
                    recent_data_window_y.append(y_instance_scalar)
                    if len(recent_data_window_X) > retrain_window_size:
                        recent_data_window_X.pop(0)
                        recent_data_window_y.pop(0)

                    # Check Drift
                    drift_detected_this_scenario = False
                    if detector:
                        current_scenario_state_for_reset = {
                            'detector': detector,
                            'name': scenario_name,
                            'is_shap': is_shap_scenario,
                            'important_feature_indices': important_feature_indices,
                            'original_dim': n_features
                        }
                        # Handle detectors
                        if isinstance(detector, (D3, DAWIDD)):
                            detector.add_record(X_instance_np[0])
                            drift_detected_this_scenario = detector.detected_change()
                        elif isinstance(detector, StudentTeacherDriftDetector):
                            drift_detected_this_scenario = detector.process_instance(ml_model_wrapper, X_instance_np)
                        elif isinstance(detector, HDDDM):
                            current_batch_X.append(X_instance_np[0])
                            current_batch_y.append(y_instance_scalar)
                            if len(current_batch_X) == hdddm_batch_size:
                                batch_df = pd.DataFrame(np.array(current_batch_X), columns=feature_names)
                                detector.add_new_batch(batch_df)
                                drift_detected_this_scenario = detector.drift_detected
                                current_batch_X = []
                                current_batch_y = []  # Clear batch

                    # Handle Drift and Retraining
                    if drift_detected_this_scenario:
                        print(f"      Drift detected for {scenario_name} at index {instance_counter}!")
                        retraining_points.append(instance_counter)

                        X_retrain = np.array(recent_data_window_X)
                        y_retrain_raw = np.array(recent_data_window_y)
                        y_retrain = y_retrain_raw.astype(int)

                        print(f"        Retraining model for {scenario_name}...")
                        ml_model_wrapper.fit(X_retrain, y_retrain)  # Retrain this scenario's model

                        if is_shap_scenario and recalc_shap:
                            print(f"        Recalculating SHAP for {scenario_name}...")
                            try:
                                X_shap_retrain_data = select_shap_data(X_retrain, shap_subset_size)
                                scenario_model = ml_model_wrapper.get_model()

                                scenario_explainer = get_shap_explainer(scenario_model, X_shap_retrain_data)
                                scenario_shap_values = calculate_shap_values(scenario_explainer,
                                                                             X_shap_retrain_data)  # Use same data
                                # Update the scenario's important_feature_indices
                                important_feature_indices, _ = select_important_features(
                                    scenario_shap_values, method=sel_method, k=k_features,
                                    feature_names=feature_names
                                )

                                # Update the detector after recalculating indices
                                update_detector_shap_features(detector, True, important_feature_indices)
                                del scenario_explainer, scenario_shap_values, scenario_model, X_shap_retrain_data
                                gc.collect()
                            except Exception as e:
                                print(f"Error recalculating SHAP for {scenario_name}: {e}")
                        current_scenario_state_for_reset['important_feature_indices'] = important_feature_indices
                        reset_detector_state(current_scenario_state_for_reset, ml_model_wrapper, X_retrain,
                                             feature_names)
                        detector = current_scenario_state_for_reset['detector']
                        gc.collect()

                    instance_counter += 1
                    if instance_counter % 10000 == 0:
                        print(f"Processed {instance_counter} instances for {scenario_name}")

                scenario_end_time = time.time()
                scenario_duration = scenario_end_time - scenario_start_time
                print(f"    Scenario {scenario_name} finished. Duration: {scenario_duration:.2f} seconds.")

                master_results.append({
                    "dataset": dataset_name if not drift else f'{dataset_name}_drifted',
                    "model": model_name,
                    "scenario": scenario_name,
                    "accuracy_history": accuracy_history,
                    "retraining_points": retraining_points,
                    "total_instances": instance_counter,
                    "final_accuracy": accuracy_history[-1][1] if accuracy_history else np.nan,
                    "scenario_duration_seconds": scenario_duration
                })

                del ml_model_wrapper, detector, accuracy_history, retraining_points
                del recent_data_window_X, recent_data_window_y, current_batch_X, current_batch_y
                if important_feature_indices is not None: del important_feature_indices
                gc.collect()

    print("\nSimulation Complete.")
    return master_results, config
