# -*- coding: utf-8 -*-
"""
Sensitivity Analysis for RetrofitABM Model
Created on Wed Dec 20 13:00:09 2023
@author: akhatova
"""

import mesa
from model import RetrofitABM
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from config import SENSITIVITY_DIR

def get_varying_parameters(params):
    """Identify varying parameters from params dictionary."""
    return [(key, value) for key, value in params.items() if isinstance(value, list)]

# Base parameters with fixed values for non-varying parameters
params = {
    "soc": False,
    "plot_distr": False,
    "randomness": False,
    "current_year": 2024,
    "price_scenario": ['19', '22', '23'],
    "mean_pbc": 0.3,
    "mean_att": 0.25,  # Base value, will vary if mean_att is selected for sensitivity
    "int_thr": 0.3,  # Base value, will vary if int_thr is selected for sensitivity
    "subsidy_ins": False,
    "subsidy_hp": 0,
    "gas_ban": False,
    "sens_ins": False,
    "ins_costs": 1,  
    "sens_hp": False,
    "hp_costs": 1,
    "n_contact": 3
}



def set_sensitivity_params(base_params, vary_param, vary_values):
    """
    Adjust parameters for sensitivity analysis on a specific parameter.
    
    Parameters:
    - base_params (dict): The base parameters.
    - vary_param (str): The parameter to vary for sensitivity.
    - vary_values (list): List of values to use for the sensitivity analysis.
    
    Returns:
    - params_list (list): List of parameter dictionaries with varied values for `vary_param`.
    """
    params_list = []
    for value in vary_values:
        params_copy = base_params.copy()
        
        # Set the main vary_param to the value
        if vary_param == "insulation_costs":
            params_copy["sens_ins"] = True
        elif vary_param == "heat_pump_costs":
            params_copy["sens_hp"] = True
        else:
            params_copy[vary_param] = value  # Default case for other parameters
        
        params_list.append(params_copy)
    return params_list



def run_sensitivity_analysis(num_iter=1, sensitivity_params=None):
    """
    Runs sensitivity analysis on the RetrofitABM model with specified parameters,
    ensuring all values of `vary_param` are generated.
    """
    # Define batch parameter configuration
    batch_params = params.copy()
    if sensitivity_params:
        batch_params.update(sensitivity_params)

    # Use Mesa batch run to vary specified parameters over the entire set of values
    results_batch = mesa.batch_run(
        RetrofitABM,
        parameters=batch_params,
        iterations=num_iter,
        number_processes=1,
        data_collection_period=1,
        max_steps=20,
        display_progress=True
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_batch)
    results_df.to_excel(os.path.join(SENSITIVITY_DIR,"sensitivity_batch_results.xlsx"), index=False)
    return results_df



def process_sensitivity_results(res_df, price_scenarios, vary_param, vary_values, num_iter, params):
    """
    Process sensitivity analysis results to create DataFrames for plotting.

    Parameters:
    - res_df (DataFrame): The main results DataFrame from simulations.
    - price_scenarios (list): List of constant price scenario values to test.
    - vary_param (str): Name of the parameter being varied.
    - vary_values (list): Values for the parameter being varied.
    - num_iter (int): Number of iterations for each combination of parameters.
    - params (dict): Dictionary of parameters to determine the decision-making framework (FIN or SOC).

    Returns:
    - df1, df2: DataFrames containing results for Share HP adoption and Savings in Final Energy for Space Heating respectively.
    """
    df1, df2 = pd.DataFrame(), pd.DataFrame()

    # Determine the decision-making framework based on the "soc" parameter
    decision_framework = "SOC" if params.get("soc", False) else "FIN"

    # Ensure types of vary_param values match those in res_df
    res_df[vary_param] = res_df[vary_param].astype(float)  # Ensure the type is consistent
    res_df['price_scenario'] = res_df['price_scenario'].astype(str)  # Convert price_scenario to string if needed

    for price_scenario in price_scenarios:
        for param_value in vary_values:
            for i in range(num_iter):
                # Debugging: print types and unique values in res_df to verify compatibility
                print(f"\nFiltering: price_scenario={price_scenario}, {vary_param}={param_value}, iteration={i}")
                print("res_df[vary_param] unique values:", res_df[vary_param].unique())
                print("res_df['price_scenario'] unique values:", res_df['price_scenario'].unique())

                # Apply filter with type-consistent values
                run_filter = (
                    (res_df['price_scenario'] == str(price_scenario)) & 
                    (res_df[vary_param] == float(param_value)) & 
                    (res_df['iteration'] == i)
                )
                
                res = res_df[run_filter]

                if res.empty:
                    print(f"No data found for {decision_framework}, price scenario {price_scenario}, {vary_param} = {param_value}")
                    continue
                
                scenario_name = f"{decision_framework}_price_scenario{price_scenario}_{vary_param}{param_value}"
                print(f"Processing: {scenario_name}")

                # Extract the final values for Share HP adoption and Savings in Final Energy for Space Heating
                share_hp = res[['Step', 'Share HP adoption']].groupby('Step').first().iloc[-1]
                heat_dem_red = res[['Step', 'Heat demand reduction (share)']].groupby('Step').first().iloc[-1]

                # Append results to DataFrames with scenario names
                df1 = pd.concat([df1, pd.DataFrame([share_hp], index=[scenario_name])], axis=0)
                df2 = pd.concat([df2, pd.DataFrame([heat_dem_red], index=[scenario_name])], axis=0)

    # Rename columns and save results
    df1.rename(columns={0: 'Share_HP_adoption'}, inplace=True)
    df2.rename(columns={0: 'Heat_demand_reduction'}, inplace=True)

    # Use the decision framework in file names
    df1.to_excel(os.path.join(SENSITIVITY_DIR,f'sens_{decision_framework}_{vary_param}_share_hp_adoption.xlsx'))
    df2.to_excel(os.path.join(SENSITIVITY_DIR,f'sens_{decision_framework}_{vary_param}_heat_demand_reduction.xlsx'))

    return df1, df2



def save_sensitivity_plot(fig, filename=None):
    """
    Save the plot with 300 DPI resolution in a folder named "sensitivity plots".
    
    Parameters:
    - fig: The matplotlib figure object to be saved.
    - filename (str): The name of the file to save (default: "sensitivity_plot.png").
    """
    # Define the directory path
    folder_path = "sensitivity plots"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Save the figure with 300 DPI resolution
    fig.savefig(file_path, dpi=300, bbox_inches='tight')

    

def plot_sensitivity_price_scenarios(df, vary_param, vary_values, ylabel="Metric Value", filename=None, params=None):
    """
    Plot sensitivity analysis results for different price scenarios with uniform y-axis scale,
    using Viridis colormap and increased font sizes for readability.

    Parameters:
    - df: DataFrame containing sensitivity analysis results (either Share HP adoption or Savings in Final Energy for Space Heating).
    - vary_param: Parameter varied in the sensitivity analysis.
    - vary_values: List of varied parameter values.
    - ylabel: Label for the y-axis.
    - filename: Name of the file to save the plot.
    - params: Dictionary of base parameters to determine decision-making framework (FIN or SOC).
    """
    # Determine the decision-making framework based on the "soc" parameter
    decision_framework = "SOC" if params and params.get("soc", False) else "FIN"

    # Define the mapping of scenarios for labels
    scenario_labels = {
        '19': f'{decision_framework}, 2019',
        '22': f'{decision_framework}, 2022',
        '23': f'{decision_framework}, 2023',
    }

    # Extract scenarios and filter data based on price scenarios
    price_scenarios = params.get("price_scenario", ['19', '22', '23'])
    fig, axes = plt.subplots(1, len(price_scenarios), figsize=(8, 2), sharey=True)
    
    # Define Viridis colormap for consistent color theme
    viridis = cm.get_cmap('viridis', len(vary_values))
    
    # Define global min and max
    global_min = -0.05
    global_max = 1.05    

    for i, (scenario, ax) in enumerate(zip(price_scenarios, axes)):
        # Filter data for the current price scenario
        scenario_data = df[df.index.str.contains(f'price_scenario{scenario}_')]
        
        # Ensure we have data for the scenario
        if scenario_data.empty:
            print(f"No data found for {decision_framework}, price scenario {scenario}.")
            continue
        
        # Extract the metric values for each vary_param value
        y = scenario_data.iloc[:, 0].values  # Select the first column (for both df1 and df2)
        percent_changes_labels = ['-50%', '-25%', '0%', '25%', '50%', '100%']

        # Plot data with Viridis colors
        ax.plot(percent_changes_labels, y, marker='o', linestyle='-', color=viridis(i / len(price_scenarios)))

        # Set titles and labels with increased font sizes
        xlabel_map = {
            'int_thr': 'Intention Threshold (%)',
            'mean_att': 'Mean Attitude (%)',
            'insulation_costs': 'Insulation Costs (%)',
            'heat_pump_costs': 'Heat Pump Costs (%)',
            'mean_pbc': 'Mean PBC (%)'
        }
        
        xlabel = xlabel_map.get(vary_param, f'Change in {vary_param}')  # Default to 'Parameter (%)' if not found
        ax.set_xlabel(xlabel, fontsize=8)
        
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=8)

        # Apply uniform y-axis limits
        ax.set_ylim(global_min, global_max)
        
        # Add scenario label to the subplot
        ax.set_title(scenario_labels[scenario], fontsize=10)
        
        # Customize ticks and grid
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add a suptitle for the entire plot
    #plt.suptitle(f'Sensitivity Analysis: {vary_param} ({decision_framework})', fontsize=14, fontweight='bold')
    
    # Save the plot using the provided filename
    if filename:
        save_sensitivity_plot(fig, filename=os.path.join(SENSITIVITY_DIR,filename))
    else:
        print("No filename provided. Plot not saved.")

    plt.show()
    plt.close(fig)


def plot_sensitivity_results(vary_param, vary_values, param):
    """Generate and save sensitivity analysis plots for Share HP adoption and Savings in Final Energy for Space Heating."""
    
    decision_framework = "SOC" if params and params.get("soc", False) else "FIN"
    # Load data for Share HP adoption and Savings in Final Energy for Space Heating based on `vary_param`
    df1, df2 = load_sensitivity_results(params, vary_param)
    
    # Debug: Confirm data loading
    if df1 is None or df2 is None:
        print(f"Error: Data for {vary_param} could not be loaded. Check your Excel files.")
        return
    
    # Plot and save for Share HP adoption
    plot_sensitivity_price_scenarios(
        df1, 
        vary_param, 
        vary_values, 
        ylabel="Share HP Adoption", 
        filename=f"sens_{decision_framework}_{vary_param}_share_hp_adoption.png",
        params=params
    )
    
    # Plot and save for Savings in Final Energy for Space Heating
    plot_sensitivity_price_scenarios(
        df2, 
        vary_param, 
        vary_values, 
        ylabel="Savings in Final Energy for SH", 
        filename=f"sens_{decision_framework}_{vary_param}_heat_demand_reduction.png",
        params=params
    )


    
def load_sensitivity_results(params, vary_param):
    """
    Load sensitivity analysis results for Share HP adoption and Savings in Final Energy for Space Heating
    from saved Excel files based on the specified vary_param.
    
    Parameters:
    - vary_param (str): The parameter that was varied in the sensitivity analysis.
    
    Returns:
    - df1, df2: DataFrames containing results for Share HP adoption and Savings in Final Energy for Space Heating.
    """
    decision_framework = "SOC" if params and params.get("soc", False) else "FIN"
    
    # Define file paths based on vary_param
    file_path_hp_adoption = os.path.join(SENSITIVITY_DIR,f'sens_{decision_framework}_{vary_param}_share_hp_adoption.xlsx')
    file_path_heat_demand_reduction = os.path.join(SENSITIVITY_DIR,f'sens_{decision_framework}_{vary_param}_heat_demand_reduction.xlsx')
    
    # Load the data from Excel files
    try:
        df1 = pd.read_excel(file_path_hp_adoption, index_col=0)
        df2 = pd.read_excel(file_path_heat_demand_reduction, index_col=0)
        print(f"Loaded data from '{file_path_hp_adoption}' and '{file_path_heat_demand_reduction}'.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    
    return df1, df2


def main():
    """Main function to execute the sensitivity analysis and plotting."""
    # Step 1: Get user input for parameter to vary
    vary_param = input("Enter the parameter to vary: 'int_thr', 'mean_att', 'mean_pbc', 'ins_costs', 'hp_costs', 'all': ")
    
    # Define base values for each parameter
    parameters = ['int_thr', 'mean_att', 'mean_pbc', 'ins_costs', 'hp_costs']
    base_values = [0.3, 0.25, 0.3, 1, 1]
    
    # Define percent changes for sensitivity analysis
    percent_changes = [-0.5, -0.25, 0, 0.25, 0.5, 1.0]
    
    if vary_param == 'all':
        # If 'all', loop through all parameters and base values
        for param, base_value in zip(parameters, base_values):
            run_sensitivity_analysis_for_param(param, base_value, percent_changes)
    else:
        # If single parameter, find its base value and run analysis
        if vary_param in parameters:
            index = parameters.index(vary_param)
            base_value = base_values[index]
            run_sensitivity_analysis_for_param(vary_param, base_value, percent_changes)
        else:
            print(f"Invalid parameter: {vary_param}. Please choose a valid parameter.")
    

def run_sensitivity_analysis_for_param(param, base_value, percent_changes):
    """Run sensitivity analysis for a single parameter."""
    # Map specific parameters to their internal variable names if needed
    if param == 'ins_costs': 
        params["sens_ins"] = True  # Ensure sens_ins is enabled for this parameter
    elif param == 'hp_costs':
        params["sens_hp"] = True  # Ensure sens_hp is enabled for this parameter
    
    vary_param = param
   
    # Calculate vary values with rounding
    vary_values = [round(base_value * (1 + p), 2) for p in percent_changes]
    print(f"Running sensitivity analysis for {vary_param} with values: {vary_values}")
    
    # Set up sensitivity parameters for the batch run
    sensitivity_params = {
        vary_param: vary_values
    }
    
    # Step 4: Run the sensitivity analysis
    num_iter = 1  # Number of iterations
    res_df = run_sensitivity_analysis(num_iter=num_iter, sensitivity_params=sensitivity_params)

    
    # Process results for plotting
    price_scenarios = params["price_scenario"]
    df1, df2 = process_sensitivity_results(res_df, price_scenarios, vary_param, vary_values, num_iter, params)
    
    # Plot results
    plot_sensitivity_results(vary_param, vary_values, params)
    
    
    
if __name__ == "__main__":
    # Define parameters and their base values
    parameters = ['int_thr', 'mean_att', 'mean_pbc', 'ins_costs', 'hp_costs']
    base_values = [0.3, 0.25, 0.3, 1, 1]
    percent_changes = [-0.5, -0.25, 0, 0.25, 0.5, 1.0]

    # Get user input to specify parameter or run for all
    vary_param = input("Enter the parameter to vary for plotting: 'int_thr', 'mean_att', 'mean_pbc', 'ins_costs', 'hp_costs', 'all': ")
    
    if vary_param == 'all':
        # Plot for all parameters
        for param, base_value in zip(parameters, base_values):
            vary_values = [round(base_value * (1 + p), 2) for p in percent_changes]
            plot_sensitivity_results(param, vary_values, params)
    elif vary_param in parameters:
        # Plot for the selected parameter
        index = parameters.index(vary_param)
        base_value = base_values[index]
        vary_values = [round(base_value * (1 + p), 2) for p in percent_changes]
        plot_sensitivity_results(vary_param, vary_values, params)
    else:
        print("Invalid parameter. Please choose a valid parameter or 'all'.")
