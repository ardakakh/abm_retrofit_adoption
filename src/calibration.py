# -*- coding: utf-8 -*-
"""
Calibration script for RetrofitABM model (SOC only).
...
"""

import mesa
from model import RetrofitABM
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from config import CALIBRATION_DIR
import os


# Extract only varying parameters
def get_varying_parameters(params):
    return [(key, value) for key, value in params.items() if isinstance(value, list)]

def extract_and_categorize_results(results_df, varying_params, num_iter, n_runs):
    df1, df2 = pd.DataFrame(), pd.DataFrame()

    for i in range(num_iter * n_runs):
        # Filter for the current RunId
        res = results_df[results_df['RunId'] == i]
        
        # Dynamically get varying parameters from the first two items in varying_params
        param1_name = varying_params[0][0]
        param2_name = varying_params[1][0]
        param1_value = res[param1_name].iloc[0]
        param2_value = res[param2_name].iloc[0]
        
        # Define scenario name using parameters
        scen = f"{param1_name}{param1_value}_{param2_name}{param2_value}"
        
        # Extract Share HP adoption for the last step in this run
        share_hp_2022 = res[['Step', 'Share HP adoption']].groupby('Step').first().iloc[4]
        
        share_hp_2022.index = [scen]  # Label with scenario
        df1 = pd.concat([df1, share_hp_2022], axis=0)
        share_hp_2023 = res[['Step', 'Share HP adoption']].groupby('Step').first().iloc[4]
        share_hp_2023.index = [scen]  # Label with scenario
        df2 = pd.concat([df2, share_hp_2023], axis=0)
    # Rename columns for clarity
    df1.rename(columns={0: 'Share HP adoption 2022'}, inplace=True)
    df2.rename(columns={0: 'Share HP adoption 2023'}, inplace=True)
    
    return df1, df2

def run_calibration(pbc_value):
    params = {
        "soc": True,  # Calibration scenario specifically for SOC
        "n_agents": 400, 
        "plot_distr": False,
        "randomness": False,
        "current_year": 2019,
        "price_scenario": 'hist',
        "mean_pbc": pbc_value, 
        "mean_att": [0.2, 0.25, 0.3, 0.35, 0.4], 
        "int_thr": [0.2, 0.25, 0.3, 0.35, 0.4], 
        "subsidy_ins": True, 
        "subsidy_hp": 0.3, 
        "gas_ban": False,
        "sens_ins": False, 
        "n_contact": 3
    }
    
    # Save `params` to a JSON file
    with open("params.json", "w") as f:
        json.dump(params, f)
        
    num_iter = 1
    results_batch = mesa.batch_run(
        RetrofitABM,
        parameters=params,
        iterations=num_iter,
        number_processes=1,
        data_collection_period=1,
        max_steps=20,
        display_progress=True
    )
    
    results_df = pd.DataFrame(results_batch)
    
    pbc_value = params["mean_pbc"]  # or wherever you get `pbc` from
    
    # Save the batch results with pbc value in filename
    results_filename = os.path.join(CALIBRATION_DIR, f"calibration_results_pbc_{pbc_value}.xlsx")
    results_df.to_excel(results_filename, index=False)
    #print(f"Results saved to {results_filename}")
    
    varying_params = get_varying_parameters(params)
    num_var_params = len(varying_params)
    num_iter = 1  # Adjust based on what you used in the batch run
    num_val_params = 1
    for i in range(num_var_params):
        num_val_params *= len(varying_params[i][1])
    n_runs = num_iter*num_val_params
    # Run the extraction and categorization
    df1, df2 = extract_and_categorize_results(results_df, varying_params, num_iter, n_runs)
    print(df1)
    print(df2)
    # # Compute the average of df1 and df2
    df_avg = (df1["Share HP adoption 2022"] + df2["Share HP adoption 2023"]) / 2
    df_avg = pd.DataFrame(df_avg, columns=["Share HP adoption"])
    
    # Save the average DataFrame to a new Excel file
    df_avg_filename = os.path.join(CALIBRATION_DIR, f"calibration_hpshare_22-23_avg_pbc_{pbc_value}.xlsx")
    df_avg.to_excel(df_avg_filename, index=True)

    plot_share_heat_pumps(df_avg, pbc_value)   

    
def load_and_plot_avg_results(pbc_value):
    """
    Load df_avg from an Excel file and plot.
    """
    # Load the average data
    df_avg_filename = os.path.join(CALIBRATION_DIR, f"calibration_hpshare_22-23_avg_pbc_{pbc_value}.xlsx")
    df_avg = pd.read_excel(df_avg_filename, index_col=0)
    
    # Plot the loaded data
    plot_share_heat_pumps(df_avg, pbc_value)
    
    
def plot_share_heat_pumps(df_avg, pbc_value):
    """
    Plots the share of heat pumps for different scenarios as a heatmap.
    """
    # Extract param1 and param2 from the index
    df_avg = df_avg.reset_index()
    df_avg[['param1', 'param2']] = df_avg['index'].str.extract(r'mean_att([0-9.]+)_int_thr([0-9.]+)')
    #df_avg = pd.DataFrame(df_avg, columns=["Share HP adoption"])
    # Convert param1 and param2 to numeric for sorting
    df_avg['param1'] = df_avg['param1'].astype(float)
    df_avg['param2'] = df_avg['param2'].astype(float)
    # Convert HP share to percentage
    df_avg['Share HP adoption'] *= 100  # Convert to percentage
    # Average values for duplicate combinations of param1 and param2
    df_avg = df_avg.groupby(['param1', 'param2'], as_index=False).mean()
    # Pivot the data
    pivot_table = df_avg.pivot(index='param1', columns='param2', values='Share HP adoption')
    # Example pivot table for Share HP adoption with `pbc` in the filename
    pbc_value = pbc_value
    pivot_table_filename = os.path.join(CALIBRATION_DIR, f"pivot_table_results_pbc_{pbc_value}.xlsx")
    pivot_table.to_excel(pivot_table_filename, index=True)
    
    # # Plot as a heatmap
    # plt.figure(figsize=(10, 6))
    # heatmap = plt.imshow(pivot_table, cmap="viridis", aspect="auto", fmt=".1f%%", origin="lower")
    # plt.colorbar(heatmap, label='Share HP Adoption')
    # plt.xlabel('Intention Threshold (int_thr)')
    # plt.ylabel('Mean Attitude (mean_att)')
    # plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=45)
    # plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    # plt.title("Share of Heat Pumps (pbc={pbc_value})")
    # plt.show()
    
    # Plot as a heatmap
    plt.figure(figsize=(6, 4), dpi=300)
    sns.heatmap(
       pivot_table, 
       annot=True, 
       fmt=".0f",  # Display values with one decimal place
       cmap="viridis", 
       cbar_kws={'label': 'Share HP Adoption (%)'}
   )
    
    # Add labels and title with `pbc` value
    plt.xlabel('Intention Threshold')
    plt.ylabel('Mean Attitude')
    plt.title(f"Share of Heat Pumps (pbc={pbc_value})")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the figure with high quality
    output_path = os.path.join(CALIBRATION_DIR, f"calib_plot_pbc_{pbc_value}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == "__main__":
    
    pbc_values = [0.1, 0.2, 0.3, 0.4]
    for pbc_value in pbc_values:
        # # Run calibration and plot results
        run_calibration(pbc_value)
        # # pbc_value = 0.1
        # results_filename = f"calibration_results_pbc_{pbc_value}.xlsx"
        # results_df = pd.read_excel(results_filename, sheet_name='Sheet1', header=0)
        # varying_params = get_varying_parameters(params)
        # num_var_params = len(varying_params)
        # num_iter = 1
        # num_val_params = 1
        # for i in range(num_var_params):
        #     num_val_params *= len(varying_params[i][1])
        # n_runs = num_iter*num_val_params
        # df1, df2 = extract_and_categorize_results(results_df, varying_params, num_iter, n_runs)
        
        # df_avg_filename = f"calibration_hpshare_22-23_avg_pbc_{pbc_value}.xlsx"
        # df_avg = pd.read_excel(df_avg_filename, sheet_name='Sheet1', header=0, index_col=0)
    
        # plot_share_heat_pumps(df_avg, pbc_value)   