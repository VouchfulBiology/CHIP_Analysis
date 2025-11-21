"""
ChIP Analysis Tool

A tool for analyzing and visualizing ChIP (Chromatin Immunoprecipitation) qPCR data.
Processes Cq values, calculates dilution factors, and generates publication-quality plots.
"""

import math
import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Constants
DEFAULT_IP_BUFFER_VOLUME = 1500  # ul
DEFAULT_INPUT_BUFFER_VOLUME = 50  # ul
CQ_DIFFERENCE_THRESHOLD = 1.0  # Maximum acceptable Cq difference between replicates
ERROR_BAR_WIDTH = 1
ERROR_BAR_CAPSIZE = 3
BAR_WIDTH_TWO_SAMPLES = 0.4
BAR_WIDTH_FOUR_SAMPLES = 0.2


def get_file_path(prompt: str, default_filename: str = "") -> str:
    """
    Prompt user for a file path with optional default.

    Args:
        prompt: The prompt message to display to the user
        default_filename: Optional default filename to suggest

    Returns:
        The file path entered by the user
    """
    if default_filename:
        print(f"Default: {default_filename}")
    file_path = input(f"{prompt}: ").strip()
    if not file_path and default_filename:
        file_path = default_filename
    return file_path


def calculate_dilution_factor(
    total_ip_volume: int,
    total_input_volume: int,
    antibody_count: int
) -> float:
    """
    Calculate the dilution factor for ChIP analysis.

    Args:
        total_ip_volume: Total volume of IP buffer per cell line in ul
        total_input_volume: Total volume of Input buffer taken per cell line in ul
        antibody_count: Number of different antibodies used for this input

    Returns:
        The calculated dilution factor

    Raises:
        ValueError: If any input value is invalid (zero or negative)
    """
    if total_ip_volume <= 0 or total_input_volume <= 0 or antibody_count <= 0:
        raise ValueError("All volume and antibody count values must be positive")

    volume_ratio = total_input_volume / total_ip_volume
    antibody_factor = 1 / antibody_count
    dilution_factor = math.log(antibody_factor / volume_ratio, 2)

    return dilution_factor


def load_and_clean_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[float]]:
    """
    Load CSV file and extract cleaned Cq data.

    Args:
        file_path: Path to the CSV file containing ChIP results

    Returns:
        Tuple of (dataframe, sample_types, cq_values)

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df_read = pd.read_csv(file_path)

    # Verify required columns exist
    required_columns = ['Cq', 'Target', 'Sample']
    missing_columns = [col for col in required_columns if col not in df_read.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")

    # Remove rows with null Cq values
    df_clean = df_read[df_read.Cq.notnull()]

    # Create unique identifier combining target and sample
    sample_types = [f" {target} {sample} ".lower()
                   for target, sample in zip(df_clean.Target, df_clean.Sample)]
    cq_values = list(df_clean.Cq)

    return df_clean, sample_types, cq_values


def organize_replicates(
    sample_types: List[str],
    cq_values: List[float]
) -> Dict[str, List[float]]:
    """
    Organize Cq values by sample type and check for replicate consistency.

    Args:
        sample_types: List of sample type identifiers
        cq_values: List of corresponding Cq values

    Returns:
        Dictionary mapping sample types to lists of Cq values
    """
    organized_data = {}

    for i, sample_type in enumerate(sample_types):
        if sample_type in organized_data:
            # Check if new value differs significantly from existing values
            if abs(organized_data[sample_type][0] - cq_values[i]) > CQ_DIFFERENCE_THRESHOLD:
                print(f"Warning: More than {CQ_DIFFERENCE_THRESHOLD} Cq difference for {sample_type}")
            organized_data[sample_type].append(cq_values[i])
        else:
            organized_data[sample_type] = [cq_values[i]]

    return organized_data


def calculate_statistics(
    organized_data: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    """
    Calculate mean and standard deviation for each sample type.

    Args:
        organized_data: Dictionary mapping sample types to Cq value lists

    Returns:
        Dictionary mapping sample types to [mean, std] lists
    """
    analyzed_data = {}

    for key, values in organized_data.items():
        values_array = np.array(values)
        cq_mean = np.mean(values_array)
        cq_std = np.std(values_array)
        analyzed_data[key] = [cq_mean, cq_std]

    return analyzed_data


def separate_input_and_ip(
    analyzed_data: Dict[str, List[float]]
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Separate samples into Input and IP categories.

    Args:
        analyzed_data: Dictionary with sample statistics

    Returns:
        Tuple of (input_details, ip_details) dictionaries
    """
    input_details = {}
    ip_details = {}

    for key, values in analyzed_data.items():
        if "input" in key.lower():
            input_details[key] = values.copy()
        else:
            ip_details[key] = values.copy()

    return input_details, ip_details


def apply_dilution_correction(
    input_details: Dict[str, List[float]],
    dilution_factor: float
) -> None:
    """
    Apply dilution factor correction to input Cq values (in-place).

    Args:
        input_details: Dictionary of input sample statistics
        dilution_factor: The calculated dilution factor
    """
    for key in input_details:
        input_details[key][0] = input_details[key][0] - dilution_factor


def get_user_identifiers(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Get target genes and sample names from user input.

    Args:
        df: The dataframe containing the data

    Returns:
        Tuple of (target_ids, primer_ids, sample_ids) with centered strings
    """
    targets_input = input("Name of target genes, separated by comma: ").lower().split(',')
    primer_ids_raw = list(set(df.Target))
    samples_input = input("Name of samples, separated by comma: ").lower().split(',')

    # Add spaces around identifiers to ensure uniqueness (e.g., P53 vs SHP53)
    target_ids = [x.strip().center(len(x.strip()) + 2) for x in targets_input]
    primer_ids = [x.center(len(x) + 2) for x in primer_ids_raw]
    sample_ids = [x.strip().center(len(x.strip()) + 2) for x in samples_input]

    return target_ids, primer_ids, sample_ids


def calculate_delta_cq(
    ip_details: Dict[str, List[float]],
    input_details: Dict[str, List[float]],
    target_ids: List[str],
    sample_ids: List[str],
    primer_ids: List[str]
) -> Dict[str, List]:
    """
    Calculate IP minus Input delta Cq values.

    Args:
        ip_details: Dictionary of IP sample statistics
        input_details: Dictionary of input sample statistics
        target_ids: List of target gene identifiers
        sample_ids: List of sample identifiers
        primer_ids: List of primer identifiers

    Returns:
        Dictionary mapping sample names to [delta_cq, combined_sd, sample, target, primer]
    """
    ip_minus_input = {}

    for key in ip_details:
        # Initialize variables to avoid undefined variable errors
        target = None
        sample = None
        primer = None

        # Match target
        for target_id in target_ids:
            if target_id.lower() in key.lower():
                target = target_id
                break

        # Match sample
        for sample_id in sample_ids:
            if sample_id.lower() in key.lower():
                sample = sample_id
                print(f"Sample Check: {sample}")
                break

        # Match primer
        for primer_id in primer_ids:
            if primer_id.lower() in key.lower():
                primer = primer_id
                break

        # Skip if any identifier wasn't matched
        if target is None or sample is None or primer is None:
            print(f"Warning: Could not match all identifiers for {key}")
            continue

        # Find corresponding input
        input_info = None
        for input_key in input_details:
            if sample.lower() in input_key.lower() and primer.lower() in input_key.lower():
                input_info = input_key
                break

        if input_info is None:
            print(f"Warning: No matching input found for {key}")
            continue

        # Calculate delta Cq and combined standard deviation
        delta_cq = ip_details[key][0] - input_details[input_info][0]
        combined_sd = ip_details[key][1] + input_details[input_info][1]

        result_name = sample + target + primer
        print(f"{result_name}: {delta_cq}")

        ip_minus_input[result_name] = [delta_cq, combined_sd, sample, target, primer]

    return ip_minus_input


def calculate_percent_ip(
    ip_minus_input: Dict[str, List]
) -> Dict[str, List[float]]:
    """
    Calculate IP/Input percentage with error bounds.

    Args:
        ip_minus_input: Dictionary with delta Cq values

    Returns:
        Dictionary mapping sample names to [percent, lower_bound, upper_bound]
    """
    percent_of_ip = {}

    for key, values in ip_minus_input.items():
        delta_cq = values[0]
        combined_sd = values[1]

        ip_over_input = 100 * math.pow(2, -delta_cq)
        lower_bound = 100 * math.pow(2, -(delta_cq + combined_sd))
        upper_bound = 100 * math.pow(2, -(delta_cq - combined_sd))

        percent_of_ip[key] = [ip_over_input, lower_bound, upper_bound]

    return percent_of_ip


def export_results(
    ip_minus_input: Dict[str, List],
    percent_of_ip: Dict[str, List[float]],
    output_path: str,
    target_ids: List[str]
) -> None:
    """
    Export analysis results to CSV file.

    Args:
        ip_minus_input: Dictionary with delta Cq values
        percent_of_ip: Dictionary with percentage values
        output_path: Directory path for output file
        target_ids: List of target identifiers for filename
    """
    # Prepare data for export
    samples = []
    delta_cqs = []
    stds = []
    ip_in_percents = []
    upper_errors = []
    lower_errors = []
    sample_outputs = []
    target_outputs = []
    primer_outputs = []

    for key in ip_minus_input:
        samples.append(key)
        delta_cqs.append(ip_minus_input[key][0])
        stds.append(ip_minus_input[key][1])
        ip_in_percents.append(percent_of_ip[key][0])
        upper_errors.append(percent_of_ip[key][2])
        lower_errors.append(percent_of_ip[key][1])
        sample_outputs.append(ip_minus_input[key][2])
        target_outputs.append(ip_minus_input[key][3])
        primer_outputs.append(ip_minus_input[key][4])

    # Create dataframe
    data = {
        "Samples": samples,
        "dCQ": delta_cqs,
        "Std": stds,
        'IP/IN (%)': ip_in_percents,
        "upper_bound": upper_errors,
        'lower_bound': lower_errors,
        'sample': sample_outputs,
        'target': target_outputs,
        'primer': primer_outputs
    }
    df_export = pd.DataFrame(data)

    # Create output filename - use first target or "multiple" if more than one
    target_name = target_ids[0].strip() if len(target_ids) == 1 else "multiple_targets"
    output_filename = f"Python_CHIP_output_{target_name}.csv"
    output_file_path = os.path.join(output_path, output_filename)

    # Export to CSV
    df_export.to_csv(output_file_path, index=False)
    print(f"Results exported to: {output_file_path}")


def chip_analyser() -> None:
    """
    Main ChIP analysis function.

    Calculates dilution factors, processes Cq data, and exports results.
    """
    print("\n=== ChIP Analyser ===\n")

    try:
        # 1. Determine Dilution Factor
        print("Step 1: Calculate Dilution Factor")
        total_ip_volume = int(input(f"Total volume of IP buffer per cell line in ul (default {DEFAULT_IP_BUFFER_VOLUME}): ") or DEFAULT_IP_BUFFER_VOLUME)
        total_input_volume = int(input(f"Total volume of Input buffer taken per cell line in ul (default {DEFAULT_INPUT_BUFFER_VOLUME}): ") or DEFAULT_INPUT_BUFFER_VOLUME)
        antibody_count = int(input("How many different antibodies were used for this input (assuming even split): "))

        dilution_factor = calculate_dilution_factor(total_ip_volume, total_input_volume, antibody_count)
        print(f"Dilution Factor Calculated: {dilution_factor:.4f}\n")

        # 2. Read CSV Files
        print("Step 2: Load Data")
        print("Please only use 1 input at a time! Thanks")
        input_csv_path = get_file_path("Enter path to CHIP results CSV file")

        df_clean, sample_types, cq_values = load_and_clean_data(input_csv_path)
        print(f"Loaded {len(cq_values)} data points\n")

        # 3. Organize replicates
        print("Step 3: Organize Replicates")
        organized_data = organize_replicates(sample_types, cq_values)

        # 4. Calculate statistics
        print("Step 4: Calculate Statistics")
        analyzed_data = calculate_statistics(organized_data)

        # 5. Separate Input and IP
        print("Step 5: Separate Input and IP Samples")
        input_details, ip_details = separate_input_and_ip(analyzed_data)
        print(f"Found {len(input_details)} input samples and {len(ip_details)} IP samples\n")

        # 6. Apply dilution correction
        print("Step 6: Apply Dilution Correction")
        apply_dilution_correction(input_details, dilution_factor)

        # 7. Get user identifiers
        print("Step 7: Identify Targets and Samples")
        target_ids, primer_ids, sample_ids = get_user_identifiers(df_clean)

        # 8. Calculate delta Cq
        print("\nStep 8: Calculate Delta Cq (IP - Input)")
        ip_minus_input = calculate_delta_cq(ip_details, input_details, target_ids, sample_ids, primer_ids)

        # 9. Calculate IP/Input percentages
        print("\nStep 9: Calculate IP/IN Percentages")
        percent_of_ip = calculate_percent_ip(ip_minus_input)

        # 10. Export results
        print("\nStep 10: Export Results")
        output_directory = get_file_path(
            "Enter output directory path",
            os.path.join(os.path.expanduser("~"), "Desktop", "CHIP_Output")
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        export_results(ip_minus_input, percent_of_ip, output_directory, target_ids)

        print("\n=== Analysis Complete! ===")
        print("If you have multiple input types, aggregate the data into a single CSV file for plotting.\n")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Please check your inputs and try again.")


def calculate_error_bars(
    samples: List[float],
    lower_bounds: List[float],
    upper_bounds: List[float]
) -> np.ndarray:
    """
    Calculate asymmetric error bar sizes.

    Args:
        samples: List of sample values
        lower_bounds: List of lower bound values
        upper_bounds: List of upper bound values

    Returns:
        2D numpy array of error bar sizes in format required by matplotlib
    """
    upper_errors = [upper_bounds[i] - samples[i] for i in range(len(samples))]
    lower_errors = [samples[i] - lower_bounds[i] for i in range(len(samples))]

    return np.array(list(zip(lower_errors, upper_errors))).T


def plot_comparison(
    df: pd.DataFrame,
    comparison_samples: List[str],
    comparison_target: str,
    bar_width: float = BAR_WIDTH_TWO_SAMPLES
) -> None:
    """
    Plot comparison of ChIP data for given samples and target.

    Args:
        df: DataFrame containing the plotting data
        comparison_samples: List of sample identifiers to compare
        comparison_target: Target gene identifier
        bar_width: Width of bars in the plot
    """
    num_samples = len(comparison_samples)

    # Extract data for each sample
    sample_data = []
    sample_lower_bounds = []
    sample_upper_bounds = []

    for sample in comparison_samples:
        sample_df = df[(df['sample'] == sample) & (df['target'] == comparison_target)]
        sample_data.append(list(sample_df.loc[:, 'IP/IN (%)']))
        sample_lower_bounds.append(list(sample_df.loc[:, 'lower_bound']))
        sample_upper_bounds.append(list(sample_df.loc[:, 'upper_bound']))

    # Get primer labels from first sample
    first_sample_df = df[(df['sample'] == comparison_samples[0]) & (df['target'] == comparison_target)]
    x_labels = list(first_sample_df.primer)

    # Calculate bar positions
    bar_positions = []
    base_positions = np.arange(len(x_labels))
    for i in range(num_samples):
        bar_positions.append([pos + (i * bar_width) for pos in base_positions])

    # Calculate error bars for each sample
    asymmetric_errors = []
    for i in range(num_samples):
        error_bars = calculate_error_bars(
            sample_data[i],
            sample_lower_bounds[i],
            sample_upper_bounds[i]
        )
        asymmetric_errors.append(error_bars)

    # Create plot
    for i in range(num_samples):
        plt.bar(bar_positions[i], sample_data[i], width=bar_width, label=f"{comparison_samples[i]}")
        plt.errorbar(
            x=bar_positions[i],
            y=sample_data[i],
            yerr=asymmetric_errors[i],
            c='k',
            fmt=' ',
            elinewidth=ERROR_BAR_WIDTH,
            capsize=ERROR_BAR_CAPSIZE
        )

    # Set x-axis labels to center of grouped bars
    if num_samples == 2:
        plt.xticks([pos + bar_width / 2 for pos in base_positions], x_labels)
    elif num_samples == 4:
        plt.xticks([pos + bar_width * 1.5 for pos in base_positions], x_labels)

    # Labels and title
    sample_names = " and ".join(comparison_samples)
    plt.title(f"Comparison of IP/IN % for {comparison_target} between {sample_names}")
    plt.xlabel("Primers")
    plt.ylabel("IP/IN Percentage")
    plt.legend()

    plt.show()


def chip_plotter() -> None:
    """
    Plot ChIP data comparing 2 samples.
    """
    print("\n=== ChIP Plotter (2 Samples) ===\n")

    try:
        # Read the file
        plotting_csv_path = get_file_path("Enter path to CHIP plotting inputs CSV file")
        df = pd.read_csv(plotting_csv_path)

        while True:
            # Get comparison parameters
            samples_input = input("What 2 samples to compare, separated by comma: ").lower().split(',')
            comparison_samples = [x.strip().center(len(x.strip()) + 2) for x in samples_input]

            target_input = input("What target to compare: ").lower()
            comparison_target = target_input.strip().center(len(target_input.strip()) + 2)

            # Generate plot
            plot_comparison(df, comparison_samples, comparison_target, BAR_WIDTH_TWO_SAMPLES)

            # Ask if user wants another graph
            repeat = input("Another Graph? (y/n): ").lower()
            if repeat != 'y':
                break

    except Exception as e:
        print(f"\nError during plotting: {str(e)}")
        print("Please check your inputs and try again.")


def chip_combined_plotter() -> None:
    """
    Plot ChIP data comparing 4 samples.
    """
    print("\n=== ChIP Combined Plotter (4 Samples) ===\n")

    try:
        # Read the file
        plotting_csv_path = get_file_path("Enter path to CHIP plotting inputs CSV file")
        df = pd.read_csv(plotting_csv_path)

        while True:
            # Get comparison parameters
            samples_input = input("What 4 samples to compare, separated by comma: ").lower().split(',')
            comparison_samples = [x.strip().center(len(x.strip()) + 2) for x in samples_input]

            if len(comparison_samples) != 4:
                print("Error: Please enter exactly 4 samples.")
                continue

            target_input = input("What target to compare: ").lower()
            comparison_target = target_input.strip().center(len(target_input.strip()) + 2)

            # Generate plot
            plot_comparison(df, comparison_samples, comparison_target, BAR_WIDTH_FOUR_SAMPLES)

            # Ask if user wants another graph
            repeat = input("Another Graph? (y/n): ").lower()
            if repeat != 'y':
                break

    except Exception as e:
        print(f"\nError during plotting: {str(e)}")
        print("Please check your inputs and try again.")


def main() -> None:
    """
    Main entry point for the ChIP Analysis Tool.
    """
    print("=" * 50)
    print("ChIP Analysis Tool")
    print("=" * 50)
    print("\nSelect a module:")
    print("a: Analyser - Process raw ChIP data")
    print("b: Plotter - Visualize processed data")
    print("=" * 50)

    program = input("\nWhich module would you like to use? (a/b): ").lower()

    if program == 'a':
        chip_analyser()
    elif program == 'b':
        num_comparisons = input("Number of samples to compare? (2 or 4): ")

        if num_comparisons == '2':
            chip_plotter()
        elif num_comparisons == '4':
            chip_combined_plotter()
        else:
            print("Invalid Input. Please enter 2 or 4.")
    else:
        print("Invalid input. Please enter 'a' or 'b'.")


if __name__ == '__main__':
    main()
