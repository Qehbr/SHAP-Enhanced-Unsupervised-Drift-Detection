# src/evaluation.py
import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def numpy_converter(obj):
    """
    Converts numpy objects to native Python types for JSON serialization.

    Args:
        obj (Any): Object to convert.

    Returns:
        int, float, list, or str: Serializable representation of the object.
    """
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def aggregate_and_summarize_results(simulation_results):
    """
    Aggregates results across multiple runs and computes summary statistics.

    Args:
        simulation_results (list): List of dictionaries containing results from `run_simulation`.

    Returns:
        tuple:
            - agg_by_scenario (dict): Raw aggregation of retraining counts, final accuracy, curves, durations.
            - scenario_summary (dict): Mean values of retraining counts, final accuracy, and duration.
    """
    if not simulation_results:
        return {}, {}

    agg_by_scenario = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'retraining_counts': [], 'final_accuracy': [], 'accuracy_curves': [], 'scenario_durations': []
    })))

    for res in simulation_results:
        ds, mdl, scenario = res['dataset'], res['model'], res['scenario']
        agg_by_scenario[ds][mdl][scenario]['retraining_counts'].append(len(res['retraining_points']))
        agg_by_scenario[ds][mdl][scenario]['final_accuracy'].append(res['final_accuracy'])
        agg_by_scenario[ds][mdl][scenario]['accuracy_curves'].append(res['accuracy_history'])
        agg_by_scenario[ds][mdl][scenario]['scenario_durations'].append(res['scenario_duration_seconds'])

    scenario_summary = defaultdict(lambda: defaultdict(dict))
    for ds, models in agg_by_scenario.items():
        for mdl, scenarios in models.items():
            for scenario, metrics in scenarios.items():
                scenario_summary[ds][mdl][scenario] = {
                    'retraining_counts': np.mean(metrics['retraining_counts']),
                    'final_accuracy': np.mean(metrics['final_accuracy']),
                    'scenario_duration': np.mean(metrics['scenario_durations'])
                }

    agg_by_scenario_dict = {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in agg_by_scenario.items()}
    scenario_summary_dict = {k: {k2: dict(v2) for k2, v2 in v.items()} for k, v in scenario_summary.items()}

    return agg_by_scenario_dict, scenario_summary_dict


def save_summary_results(summary_data, config_used, base_filename, output_dir):
    """
    Saves the scenario summary results to a JSON file.

    Args:
        summary_data (dict): Aggregated summary results.
        config_used (dict): Configuration dictionary used in the simulation.
        base_filename (str): Prefix for the output file.
        output_dir (str): Directory where the JSON file will be saved.
    """
    summary_results_path = os.path.join(output_dir, f"{base_filename}_summary_scenarios.json")
    full_summary_data = {
        'config': config_used,
        'summary': summary_data
    }
    try:
        with open(summary_results_path, 'w') as f:
            json.dump(full_summary_data, f, indent=4, default=numpy_converter)
        print(f"Aggregated scenario summary saved to: {summary_results_path}")
    except Exception as e:
        print(f"Error saving summary results: {e}")


def plot_scenario_comparison(simulation_results, output_dir):
    """
    Generates line plots comparing accuracy curves across baseline and adaptation scenarios.

    Args:
        simulation_results (list): Simulation output containing 'accuracy_history' and metadata.
        output_dir (str): Root directory for saving plots.
    """
    print("\nGenerating comparison plots...")
    if not simulation_results:
        print("No simulation results provided for plotting.")
        return

    sns.set_theme(style="whitegrid")

    scenario_groups = {
        'D3': ['baseline', 'D3_standard', 'D3_shap'],
        'Dawidd': ['baseline', 'Dawidd_standard', 'Dawidd_shap'],
        'HDDDM': ['baseline', 'HDDDM_standard', 'HDDDM_shap'],
        'ST': ['baseline', 'ST_standard', 'ST_shap']
    }

    # Reorganize data by dataset & model
    data_grouped_by_ds_model = defaultdict(list)
    for res in simulation_results:
        data_grouped_by_ds_model[(res['dataset'], res['model'])].append(res)

    # Create plots for each dataset/model
    for (ds, mdl), results_for_combo in data_grouped_by_ds_model.items():
        print(f"\nProcessing Dataset: {ds}, Model: {mdl}")

        for group_name, scenarios_in_group in scenario_groups.items():
            print(f"Checking plot generation for group: {group_name}")

            results_filtered_for_group = [res for res in results_for_combo if res['scenario'] in scenarios_in_group]

            # Skip if no data found at all for this group
            if not results_filtered_for_group:
                print(f"No data found for scenarios {scenarios_in_group}. Skipping plot {group_name}.")
                continue

            scenarios_found_in_group_set = {res['scenario'] for res in results_filtered_for_group}
            scenarios_other_than_baseline = scenarios_found_in_group_set - {'baseline'}

            if not scenarios_other_than_baseline:
                print(f"Only 'baseline' data (or none) found besides baseline for group '{group_name}'. Skipping plot.")
                continue

            print(f"Generating plot for group: {group_name} (Scenarios: {', '.join(scenarios_found_in_group_set)})")
            plot_data_for_group = []

            # Prepare data points for the plot
            for res in results_filtered_for_group:
                scenario = res['scenario']
                acc_hist = res['accuracy_history']
                if not acc_hist:
                    print(f"Warning: No accuracy history for {scenario} in {ds}/{mdl}.")
                    continue

                for index, accuracy in acc_hist:
                    plot_data_for_group.append({
                        'Scenario': scenario,
                        'Instance': index,
                        'Accuracy': accuracy
                    })

            if not plot_data_for_group:
                print(f"No valid accuracy data points generated for group {group_name}. Skipping plot.")
                continue

            df_plot = pd.DataFrame(plot_data_for_group)

            plot_output_dir_specific = os.path.join(output_dir, ds, mdl)
            os.makedirs(plot_output_dir_specific, exist_ok=True)

            plt.figure(figsize=(14, 8))

            hue_order = sorted(list(scenarios_found_in_group_set))
            sns.lineplot(data=df_plot, x='Instance', y='Accuracy',
                         hue='Scenario', style='Scenario',
                         hue_order=hue_order, style_order=hue_order,
                         markers=False, dashes=True, errorbar='sd')

            plt.title(f'Accuracy Comparison ({group_name}) - {ds} / {mdl}')
            plt.xlabel('Instances Processed')
            plt.ylabel('Accuracy')

            min_acc = df_plot['Accuracy'].min()
            max_acc = df_plot['Accuracy'].max()
            plt.ylim(bottom=max(0, min_acc - 0.05), top=min(1.05, max_acc + 0.05))

            plt.legend(title='Scenario', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            plot_filename = f"{ds}_{mdl}_{group_name}.png"
            plot_path = os.path.join(plot_output_dir_specific, plot_filename)

            try:
                plt.savefig(plot_path)
                print(f"   Comparison plot saved to: {plot_path}")
            except Exception as e:
                print(f"   Error saving plot {plot_path}: {e}")
            plt.close()

    print("\nPlot generation finished.")
