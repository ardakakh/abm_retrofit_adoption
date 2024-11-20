# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:00:21 2023

Comparison script for RetrofitABM model (SOC vs. FIN).
"""

import mesa
from model import RetrofitABM
import pandas as pd
from config import BATCH_RUN_DIR
import os

# Set display options for DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Global constants
n_steps = 20
current_year = 2024
base_params = {
    "soc": [False, True],
    "plot_distr": False,
    "randomness": False,
    "current_year": current_year,
    "price_scenario": ['19', '22', '23'],
    "mean_pbc": 0.1,
    "mean_att": 0.3,
    "int_thr": 0.2,
    "subsidy_ins": False,
    "subsidy_hp": 0,
    "gas_ban": False,
    "sens_ins": False,
    "ins_costs": 1,  
    "sens_hp": False,
    "hp_costs": 1,
    "n_contact": 3
}


# Define parameter variants
variants = {
    "reference": {"subsidy_hp": 0, "subsidy_ins": False, "gas_ban": False},
    "subsidy_hp_0.3": {"subsidy_hp": 0.3, "subsidy_ins": False, "gas_ban": False},
    "subsidy_ins": {"subsidy_hp": 0, "subsidy_ins": True, "gas_ban": False},
    "subsidy_hp_0.3_and_subsidy_ins": {"subsidy_hp": 0.3, "subsidy_ins": True, "gas_ban": False},
    "gas_ban": {"subsidy_hp": 0, "subsidy_ins": False, "gas_ban": True}
}


def determine_scenario(params=base_params):
    """
    Determines the scenario based on input parameters.
    
    Args:
        params (dict): Dictionary of parameters.
        
    Returns:
        str: 'ref' if parameters match the reference scenario; 'other' otherwise.
    """
    ref_params = {
        "soc": [False, True],
        "plot_distr": False,
        "randomness": False,
        "current_year": current_year,
        "price_scenario": ['19', '22', '23'],
        "mean_pbc": 0.1,
        "mean_att": 0.3,
        "int_thr": 0.2,
        "subsidy_ins": False,
        "subsidy_hp": 0,
        "gas_ban": False,
        "sens_ins": False,
        "ins_costs": 1,  
        "sens_hp": False,
        "hp_costs": 1,
        "n_contact": 3
    }
    return 'ref' if params == ref_params else 'other'


def run_batch_simulation(params):
    """Runs batch simulation for the specified parameters."""
    print("Running batch simulation...")
    results_batch = mesa.batch_run(
        RetrofitABM,
        parameters=params,
        iterations=1,
        number_processes=1,
        data_collection_period=1,
        max_steps=n_steps,
        display_progress=False
    )
    return pd.DataFrame(results_batch)

def process_adoption_share(res, column_name):
    """Processes and returns adoption share data."""
    numadopt = res[['Step', 'Num adopters']].groupby('Step').mean().round(decimals=2)
    numadopt.columns = [column_name]
    return numadopt

def process_adopted_packages(res, column_name):
    """Processes and returns adopted packages data."""
    res_adopt = res['Adopted options cumulative'].iloc[-1]
    res_adopt_df = pd.DataFrame(list(res_adopt.values()), index=res_adopt.keys(), columns=[column_name])
    return res_adopt_df

def process_heat_demand_reduction(res, column_name):
    """Processes and returns heat demand reduction data."""
    res_heat = res[['Step', 'Heat demand reduction (share)']].groupby('Step').mean().round(decimals=2)
    res_heat.columns = [column_name]
    return res_heat

def process_co2_reduction(res, column_name):
    """Processes and returns CO2 reduction data."""
    res_co2 = res[['Step', 'Carbon emission reduction']].groupby('Step').mean().round(decimals=2)
    res_co2.columns = [column_name]
    return res_co2


def run_comparison_analysis(res_df, variant_name):
    """Runs comparison analysis for the FIN vs. SOC scenarios and saves results."""
    
    df1, df2, df3, df4, df_group = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    for i in range(6):
        # Get data for each run
        res = res_df[res_df['RunId'] == i]
        soc_str = "SOC" if res['soc'].iloc[0] else "FIN"
        price_scen = str(20)+res['price_scenario'].iloc[0]
        column_name = f"{soc_str}, {price_scen}"

        # Process and accumulate data for each metric
        df1 = pd.concat([df1, process_adoption_share(res, column_name)], axis=1)
        df2 = pd.concat([df2, process_adopted_packages(res, column_name)], axis=1)
        df3 = pd.concat([df3, process_heat_demand_reduction(res, column_name)], axis=1)
        df4 = pd.concat([df4, process_co2_reduction(res, column_name)], axis=1)

        # Cumulative state by heating system type
        cumul_state = res[['Step', "Current HS and INS"]].groupby('Step').first()
        cumul_state.columns = [column_name]
        df_group = pd.concat([df_group, cumul_state], axis=1)

    # Add year information to df1, df3, and df4
    df1.insert(0, 'Year', df1.index + current_year)
    df3.insert(0, 'Year', df3.index + current_year)
    df4.insert(0, 'Year', df4.index + current_year)
    
    # Save results to Excel files
    save_results_to_excel(df1, df2, df3, df4, df_group, variant_name)
    
    
    
def save_results_to_excel(df1, df2, df3, df4, df_group, variant_name):
    """Saves result DataFrames to Excel files with variant-specific filenames."""
    
    os.makedirs(BATCH_RUN_DIR, exist_ok=True)
    df1.to_excel(os.path.join(BATCH_RUN_DIR, f'adoptions_{variant_name}.xlsx'), index=True)
    df2.to_excel(os.path.join(BATCH_RUN_DIR, f'adopted_packages_{variant_name}.xlsx'), index=True)
    df3.to_excel(os.path.join(BATCH_RUN_DIR, f'heat_reduction_{variant_name}.xlsx'), index=True)
    df4.to_excel(os.path.join(BATCH_RUN_DIR, f'co2_reduction_{variant_name}.xlsx'), index=True)

    df_group.to_excel(os.path.join(BATCH_RUN_DIR, f'current_stock_{variant_name}.xlsx'), index=True)
    print(f"Results saved to Excel with prefix '{variant_name}'.")



def main():
    """Main function to run simulations for all parameter variants."""
    for variant_name, variant_params in variants.items():
        # Update base parameters with the variant-specific parameters
        params = base_params.copy()
        params.update(variant_params)
        
        # Run the batch simulation and analysis
        print(f"Running scenario: {variant_name}")
        res_df = run_batch_simulation(params)
        run_comparison_analysis(res_df, variant_name)
        

if __name__ == "__main__":
    main()
