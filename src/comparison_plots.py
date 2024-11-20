# -*- coding: utf-8 -*-
"""
Comparison Plots for RetrofitABM Model
Created on Tue Dec 12 15:27:24 2023
@author: akhatova
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.cm as cm
import ast
import os
from config import BATCH_RUN_DIR


# Global configurations
pd.options.plotting.backend = "matplotlib"
cmap = plt.cm.get_cmap('viridis')  # Use Viridis colormap
font_titles = 14
font_labels = 13
font_legends = 12
font_ticks = 12

# Define global colors and styles
num_colors = 10
colors = cmap(np.linspace(0, 1, num_colors))
line_styles = ['-', '--', ':']
markers = ['o', 's', '^']

# Helper Functions
def generate_xticks(min_val, max_val, n_ticks):
    """
    Generates a range of ticks for the x-axis based on minimum and maximum values.
    
    Args:
    - min_val (float): The minimum value for the range.
    - max_val (float): The maximum value for the range.
    - n_ticks (int): Number of ticks to generate.

    Returns:
    - np.ndarray: Array of tick values.
    """
    tick_range = max_val - min_val
    tick_step = tick_range / (n_ticks - 1) if min_val >= 0 else max_val / (n_ticks - 1)
    return np.arange(min_val, max_val + tick_step, tick_step)


def load_data(file_path):
    """
    Loads data from an Excel file if it exists.

    Args:
    - file_path (str): Path to the Excel file.

    Returns:
    - pd.DataFrame: DataFrame containing the loaded data.
    """
    if os.path.exists(file_path):
        return pd.read_excel(file_path, sheet_name='Sheet1', header=0, index_col=0)
    else:
        print(f"File not found: {file_path}")
        return None


def plot_adoptions(file_ad, save_path=None):
    """
    Plots cumulative adoptions per year for multiple scenarios.

    Args:
    - file_ad (str): Path to the Excel file containing adoption data.
    - save_path (str, optional): Path to save the plot image. Default is None.

    Returns:
    - None: Displays the plot.
    """
    adoptions_df = load_data(file_ad)
    if adoptions_df is None:
        return

    adoptions_df = adoptions_df.reset_index()
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), dpi=300)
    y_values1 = adoptions_df.columns[2:5].tolist()
    y_values2 = adoptions_df.columns[5:8].tolist()

    for i, (ax, y_values) in enumerate(zip(axs, [y_values1, y_values2])):
        for j, column in enumerate(y_values):
            ax.plot(adoptions_df['Year'], adoptions_df[column], label=column, 
                    color=colors[j], linestyle=line_styles[j % len(line_styles)], 
                    marker=markers[j % len(markers)], alpha=0.8, linewidth=2)
        ax.set_xlabel('Year', fontsize=font_labels)
        ax.legend(title='Scenario', fontsize=font_legends)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.text(0.04, 0.5, 'Cumulative number of adoptions', va='center', rotation='vertical', fontsize=font_labels)
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    plt.show()


def plot_adopted_packages(file_pack, save_path=None):
    """
    Plots cumulative adopted packages for multiple scenarios.

    Args:
    - file_pack (str): Path to the Excel file containing adopted package data.
    - save_path (str, optional): Path to save the plot image. Default is None.

    Returns:
    - None: Displays the plot.
    """
    packages_df = load_data(file_pack)
    print(type(packages_df))
    print(packages_df.head())
    if isinstance(packages_df, pd.DataFrame):
        packages_df = packages_df.loc[~(packages_df == 0).all(axis=1)]
    else:
        raise ValueError("packages_df is not a Pandas DataFrame.")
    packages_df = packages_df[~(packages_df == 0).all(axis=1)]  # Remove rows where all values are 0
    
    fig, axs = plt.subplots(2, 3, figsize=(7, 6), dpi=300)
    
    for i in range(6):
        ax = axs[i // 3, i % 3]
        col_index = i % 3
        scenario_label = 'Reference' if i < 3 else 'SOC'
        
        col_name = packages_df.columns[col_index] if scenario_label == 'Reference' else packages_df.columns[col_index + 3]
        y_positions = range(len(packages_df))
        ax.barh(y_positions, packages_df[col_name], color=colors[i], alpha=1.0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(packages_df.index, fontsize=10 if i % 3 == 0 else 0)
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'Scenario: {col_name[:9]}', fontsize=12)
    
    fig.text(0.5, 0.02, 'Adoption rate [% of adopters]', ha='center', va='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    plt.show()


def transform_current_stock(file_stock):
    """
    Transforms the current stock data from an Excel file, applying transformations to each column.

    Args:
    - file_stock (str): Path to the Excel file containing current stock data.

    Returns:
    - dict: Dictionary where each key is a scenario name and each value is a DataFrame containing the transformed data.
    """
    df = load_data(file_stock)
    transformed_dfs = {}

    for column in df.columns:
        df[column] = df[column].apply(ast.literal_eval)
        transformed_df = df[column].apply(lambda x: pd.Series(x))
        transformed_dfs[column] = transformed_df

    return transformed_dfs


def plot_heating_insulation_stock(file_stock, save_path=None):
    """
    Generates area plots of the current heating system and insulation for multiple scenarios in a 2x3 grid.

    Args:
    - file_stock (str): Path to the Excel file containing current stock data.
    - save_path (str, optional): Path to save the plot image. Default is None.

    Returns:
    - None: Displays the plot.
    """
    transformed_dfs = transform_current_stock(file_stock)

    # 6 subplots in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=300, sharey=True)
    axs_flattened = axs.flatten()

    # Enumerate over axes and transformed_dfs items
    for ax, (scenario, df) in zip(axs_flattened, transformed_dfs.items()):
        # Set up the DataFrame for plotting
        df = df.reset_index(drop=True)
        df.index = df.index + 2024  # Set starting year as 2024
        df.index.name = 'Year'

        # Use the Viridis colormap for vibrant colors
        colors = cm.viridis(np.linspace(0, 1, len(df.columns)))

        # Plot each package group as a stacked area plot
        baseline = np.zeros(len(df.index))
        for color, column in zip(colors, df.columns):
            ax.fill_between(df.index, baseline, baseline + df[column], label=column, color=color)
            baseline += df[column]

        # Customizing individual subplot
        ax.set_title(scenario)
        ax.set_xlabel('Year', fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer year ticks
        
        # Set y-axis label only for the first column
        if ax in axs[:, 0]:
            ax.set_ylabel('Number of Agents', fontsize=12)

    # Create a legend below the plot
    handles, labels = axs_flattened[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=True)
    
    # Adjust layout to accommodate the legend below the plot
    plt.subplots_adjust(hspace=0.3, wspace=0.1, bottom=0.08)

    # Save or display the plot
    if save_path:
        fig.savefig(save_path, dpi=300, format='png', bbox_inches='tight')
    plt.show()


# Example plotting function
def plot_final_mix(file_stock, scenario, save_path=None):
    """
    Plot the final mix of options for retrofit packages for either the reference
    scenario or a comparison of a non-reference scenario with the reference.
    """
    df = transform_current_stock(file_stock)
    
    final_values = {}
    for scenario_name, data in df.items():
        final_values[scenario_name] = data.iloc[-1]  # Get the last row for each scenario
    
    final_df = pd.DataFrame(final_values).T 
    
    if scenario == 'reference':
        final_df_path = os.path.join(BATCH_RUN_DIR, "final_mix_reference.xlsx")
        # Save the DataFrame
        final_df.to_excel(final_df_path, index=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.viridis(np.linspace(0, 1, final_df.shape[1]))
        final_df.plot(kind='bar', stacked=True, ax=ax, color=colors)
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Number of Retrofit Package Groups')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'{scenario}')
        plt.xticks(rotation=0)

        # Update legend settings
        legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.7, frameon=True)
        legend.get_frame().set_alpha(0.7)  # Slight transparency for the box
        legend.get_frame().set_boxstyle('round,pad=0.3')  # Small padding
        legend.set_bbox_to_anchor((1.0, 0.0))  # Place within plot, lower right
        legend.set_draggable(True)  # Optional: allow dragging
        final_df.to_excel(os.path.join(BATCH_RUN_DIR,'final_mix_{scenario}.xlsx'), index=True)
        plt.tight_layout()
        
    else:
        df_fin_ref = pd.read_excel(os.path.join(BATCH_RUN_DIR,'final_mix_reference.xlsx'), sheet_name='Sheet1', index_col=0)
        final_df_path = os.path.join(BATCH_RUN_DIR, f"final_mix_{scenario}.xlsx")
        # Save the DataFrame
        final_df.to_excel(final_df_path, index=True)
        diff = final_df - df_fin_ref
        
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.viridis(np.linspace(0, 1, diff.shape[1]))
        diff.plot(kind='bar', stacked=True, ax=ax, color=colors)
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Change in Number of Retrofit Packages')
        ax.set_title(f"{scenario}")        
        plt.xticks(rotation=0)
        ax.set_ylim(-100, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Update legend settings
        legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.7, frameon=True)
        legend.get_frame().set_alpha(0.7)
        legend.get_frame().set_boxstyle('round,pad=0.3')
        legend.set_bbox_to_anchor((1.0, 0.0))
        plt.tight_layout()

    if save_path:
        fig.savefig(os.path.join(BATCH_RUN_DIR, save_path), dpi=300, format='png', bbox_inches='tight')
    plt.show()


def plot_heat_and_co2_reduction(file_heat, file_co2, save_path=None):
    """
    Plots heat demand reduction and CO2 reduction over time for multiple scenarios using the Viridis colormap.

    Parameters:
    - file_heat (str): The path to the Excel file containing heat demand reduction data.
    - file_co2 (str): The path to the Excel file containing CO2 reduction data.
    - save_path (str, optional): Path to save the plot image. Default is None.

    Returns:
    - None: The function directly generates and displays plots.
    """
    # Load data
    heat_df = pd.read_excel(file_heat, sheet_name='Sheet1', header=0, index_col=1)
    co2_df = pd.read_excel(file_co2, sheet_name='Sheet1', header=0, index_col=1)
    
    # Remove the 'Step' column if it exists
    if 'Step' in heat_df.columns:
        heat_df = heat_df.drop(columns='Step')
    if 'Step' in co2_df.columns:
        co2_df = co2_df.drop(columns='Step')
        
    # Determine y-axis limits based on the data
    global_max = max(heat_df.max().max(), co2_df.max().max())
    global_min = min(heat_df.min().min(), co2_df.min().min())
    ylim_lower = global_min - (0.1 * abs(global_min))
    ylim_upper = global_max + (0.1 * abs(global_max))

    # Set up the figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    # Use Viridis colormap for line colors
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, len(heat_df.columns)))

    # Plot heat demand reduction
    for i, (column, color) in enumerate(zip(heat_df.columns, viridis_colors)):
        axs[0].plot(heat_df.index, heat_df[column], label=column, 
                    color=color, linestyle='-', marker='o', alpha=0.8, linewidth=2)    

    axs[0].set_xlabel('Year', fontsize=12)
    axs[0].set_ylabel('Savings in final energy demand for SH [%]', fontsize=12)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(True, linestyle='--', alpha=0.8)
    axs[0].set_ylim(ylim_lower, ylim_upper)

    # Plot CO2 reduction
    for i, (column, color) in enumerate(zip(co2_df.columns, viridis_colors)):
        axs[1].plot(co2_df.index, co2_df[column], label=column, 
                    color=color, linestyle='-', marker='o', alpha=0.8, linewidth=2)    

    axs[1].set_xlabel('Year', fontsize=12)
    axs[1].set_ylabel('Carbon emissions reduction [%]', fontsize=12)
    axs[1].legend(title='Scenario', fontsize=10)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(True, linestyle='--', alpha=0.8)
    axs[1].set_ylim(ylim_lower, ylim_upper)

    plt.tight_layout()
    if save_path:
        fig.savefig(os.path.join(BATCH_RUN_DIR, save_path), dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
    
def save_all_plots(variant_name, file_ad, file_pack, file_stock, file_heat, file_co2):
    """
    Generates and saves all comparison plots as high-resolution PNGs with the variant name.

    Parameters:
    - variant_name: Name of the variant (e.g., 'reference', 'subsidy_hp_0.3', etc.).
    - file_ad, file_pack, file_stock, file_heat, file_co2: paths to the data files.
    """
    # Directory for saving plots
    os.makedirs(BATCH_RUN_DIR, exist_ok=True)

    # Save each plot with a filename that includes the variant name
    plot_adoptions(file_ad, save_path=os.path.join(BATCH_RUN_DIR, f"adoptions_plot_{variant_name}.png"))
    plot_adopted_packages(file_pack, save_path=os.path.join(BATCH_RUN_DIR, f"adopted_packages_plot_{variant_name}.png"))
    plot_heating_insulation_stock(file_stock, save_path=os.path.join(BATCH_RUN_DIR, f"heating_insulation_stock_plot_{variant_name}.png"))
    plot_heat_and_co2_reduction(file_heat, file_co2, save_path=os.path.join(BATCH_RUN_DIR, f"heat_co2_reduction_plot_{variant_name}.png"))
    plot_final_mix(file_stock, scenario=variant_name, save_path=os.path.join(BATCH_RUN_DIR, f"final_mix_plot_{variant_name}.png"))


def plot_final_mix_from_dynamic_files(base_path_template, scenarios, output_directory="final_mix_plots"):
    """
    Plot the final mix of options for multiple scenarios using dynamically generated file paths.

    Parameters:
    - base_path_template (str): Template for the file paths, e.g., "final_mix_plot_{scenario}".
    - scenarios (list): List of scenario names to replace in the template.
    - output_directory (str): Directory to save the final mix plots. Default is 'final_mix_plots'.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    for scenario in scenarios:
        file_path = base_path_template.format(scenario=scenario)
        save_path = os.path.join(output_directory, f"final_mix_{scenario}.png")
        plot_final_mix(file_path, scenario, save_path=save_path)
        print(f"Saved final mix plot for scenario '{scenario}' to {save_path}")



if __name__ == "__main__":

    # Example usage
    # base_path_template = os.path.join(BATCH_RUN_DIR, "current_stock_{scenario}.xlsx")  # Template for file paths
    # scenarios = ['reference', 'subsidy_hp_0.3', 'subsidy_ins', 
    #                   'subsidy_hp_0.3_and_subsidy_ins']  # Add your actual scenario names here
    # plot_final_mix_from_dynamic_files(base_path_template, scenarios)
    
    file_heat=os.path.join(BATCH_RUN_DIR, 'heat_reduction_reference.xlsx') 
    file_co2=os.path.join(BATCH_RUN_DIR, 'co2_reduction_reference.xlsx')
    plot_heat_and_co2_reduction(file_heat, file_co2, "heat_co2_reduction_plot_reference.png")

