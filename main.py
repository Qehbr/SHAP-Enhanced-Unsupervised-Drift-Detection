# main.py
import argparse
import os
import json
from datetime import datetime
from src.simulation import run_simulation
from src.evaluation import aggregate_and_summarize_results, save_summary_results, plot_scenario_comparison, \
    numpy_converter


def main():
    # Get Arguments
    parser = argparse.ArgumentParser(description="Run SHAP Concept Drift Experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    print(f"Using configuration file: {args.config}")

    # Run Simulation
    simulation_results, config_used = run_simulation(config_path=args.config)
    if not simulation_results:
        print("Simulation did not produce results.")
        return

    # Prepare for saving
    output_dir = config_used['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_filename = ''
    for dataset in config_used['datasets']:
        if dataset['drift']:
            base_results_filename += f'{dataset["name"]}_drifted_'
        else:
            base_results_filename += f'{dataset["name"]}_'
    for model in config_used['models']:
        base_results_filename += f'{model["name"]}_'
    for detector in config_used['detectors']:
        if config_used['detectors'][detector]['enabled']:
            base_results_filename += f'{detector}_'
    base_results_filename += f"{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    raw_results_path = os.path.join(output_dir, f"{base_results_filename}_raw.json")
    try:
        full_raw_data = {'config': config_used, 'results': simulation_results}
        with open(raw_results_path, 'w') as f:
            json.dump(full_raw_data, f, indent=4, default=numpy_converter)
        print(f"Raw scenario results saved to: {raw_results_path}")
    except Exception as e:
        print(f"Error saving raw results: {e}")

    # Aggregate and Summarize Results
    agg_by_scenario_detailed, scenario_summary = aggregate_and_summarize_results(simulation_results)

    # Save Aggregated Summary
    save_summary_results(scenario_summary, config_used, base_results_filename, output_dir)

    # Generate Plots
    plot_scenario_comparison(simulation_results, output_dir)
    print("\nExperiment finished.")


if __name__ == "__main__":
    main()
