import model
import comparison  # Existing comparison plotting script
import pandas as pd
import calibration
import comparison_plots
import sensitivity
import os
from config import SINGLE_RUN_DIR, BATCH_RUN_DIR

    
def run_single_scenario(scenario_type, price_scenario):
    """
    Runs the model for a single scenario, retrieves data, and saves results.
    
    Args:
        scenario_type (str): 'fin', 'soc'.
        price_scenario (int): The price scenario to use (e.g., 19, 22, or 23).
    """
    if scenario_type not in ['fin', 'soc']:
        raise ValueError("Invalid scenario_type. Choose 'fin' or 'soc'.")

    soc = True if scenario_type == 'soc' else False
    
    # Run the model with the selected scenario and price, and get data
    model_instance = model.run(soc=soc, price_scenario=price_scenario)
    
    # Collect results directly from model_instance
    data = model_instance.get_results()  # Adjust `get_results()` in your model to return DataFrames
    
    # Extract desired data for analysis
    adoptions = data[["Num adopters"]]
    adoptions_share = data[["Share adopters"]]
    packages_final = data["Adopted options cumulative"].iloc[-1]  # Use -1 for the final row
    share_hp = data[["Share HP adoption"]]

    # Save results to Excel files
    save_single_run_results(
        adoptions=adoptions,
        adoptions_share=adoptions_share,
        packages_final=packages_final,
        share_hp=share_hp,
        scenario_type="SOC" if soc else "FIN",
        price_scenario=price_scenario
    )


def save_single_run_results(adoptions, adoptions_share, packages_final, share_hp, scenario_type, price_scenario):
    """
    Saves results from a single scenario to Excel files in the appropriate directory.

    Args:
        adoptions (pd.DataFrame): DataFrame containing the number of adopters.
        adoptions_share (pd.DataFrame): DataFrame containing the share of adopters.
        packages_final (dict): Final adopted options as a dictionary.
        share_hp (pd.DataFrame): DataFrame containing the share of heat pump adoption.
        scenario_type (str): Indicates the scenario type ('FIN' or 'SOC').
    """
    # Define the output directory
  
    os.makedirs(SINGLE_RUN_DIR, exist_ok=True)

    # Define filenames with scenario type
    adoptions_file = os.path.join(SINGLE_RUN_DIR, f'adoptions_{scenario_type}_price{price_scenario}.xlsx')
    adoptions_share_file = os.path.join(SINGLE_RUN_DIR, f'share_adoptions_{scenario_type}_price{price_scenario}.xlsx')
    packages_final_file = os.path.join(SINGLE_RUN_DIR, f'cumulative_packages_{scenario_type}_price{price_scenario}.xlsx')
    share_hp_file = os.path.join(SINGLE_RUN_DIR, f'share_hp_{scenario_type}_price{price_scenario}.xlsx')

    # Save data to Excel files
    adoptions.to_excel(adoptions_file, index=False)
    adoptions_share.to_excel(adoptions_share_file, index=False)

    # Convert cumulative packages to a DataFrame and save as Excel
    cumulative_df = pd.DataFrame(packages_final.items(), columns=["Option", "Value"])
    cumulative_df.to_excel(packages_final_file, index=False)

    # Save heat pump adoption share
    share_hp.to_excel(share_hp_file, index=False)
    

def run_calibration_scenario():
    """
    Runs the calibration scenario by executing the calibration script.
    """
    print("Running calibration scenario...")
    pbc_values = [0.1, 0.2, 0.3, 0.4]
    for pbc_value in pbc_values:
        # # Run calibration and plot results
        calibration.run_calibration(pbc_value) 

    
def run_comparison_with_variants():
    """
    Runs the comparison scenario with multiple parameter variants, including the reference scenario,
    saving each plot with a unique name.
    """
    # Define parameter variants
    variants = {
        "reference": {"subsidy_hp": 0, "subsidy_ins": False, "gas_ban": False},
        "subsidy_hp_0.3": {"subsidy_hp": 0.3, "subsidy_ins": False, "gas_ban": False},
        "subsidy_ins": {"subsidy_hp": 0, "subsidy_ins": True, "gas_ban": False},
        "subsidy_hp_0.3_and_subsidy_ins": {"subsidy_hp": 0.3, "subsidy_ins": True, "gas_ban": False},
        "gas_ban": {"subsidy_hp": 0, "subsidy_ins": False, "gas_ban": True}
    }

    # Run each variant
    for variant_name, params_update in variants.items():
        # Update base parameters with the variant-specific values
        params = comparison.base_params.copy()
        params.update(params_update)
        
        # Run the batch simulation for the current variant
        print(f"Running scenario: {variant_name}")
        res_df = comparison.run_batch_simulation(params)
        
        # Perform analysis and save results with variant name
        comparison.run_comparison_analysis(res_df, variant_name)
        
        # Save plots with the variant name
        comparison_plots.save_all_plots(
            variant_name, 
            file_ad=os.path.join(BATCH_RUN_DIR, f'adoptions_{variant_name}.xlsx'), 
            file_pack=os.path.join(BATCH_RUN_DIR, f'adopted_packages_{variant_name}.xlsx'), 
            file_stock=os.path.join(BATCH_RUN_DIR, f'current_stock_{variant_name}.xlsx'), 
            file_heat=os.path.join(BATCH_RUN_DIR, f'heat_reduction_{variant_name}.xlsx'), 
            file_co2=os.path.join(BATCH_RUN_DIR, f'co2_reduction_{variant_name}.xlsx'))



def run_sensitivity_scenario():
    """
    Runs the sensitivity analysis and generates relevant plots.
    This function uses the sensitivity analysis functions defined in `sensitivity.py`
    to execute the analysis, save results, and generate visualizations.
    """
    print("Running sensitivity analysis scenario...")
    
    # Run the sensitivity analysis and plot results
    sensitivity.main()  # Execute the main function in sensitivity.py that handles analysis and plotting
    
    print("Sensitivity analysis and plots completed.")



def main():
    # Select the scenario type: 'financial', 'social', 'calibration', or 'comparison'
    scenario = input("Select the scenario: 'fin', 'soc', 'calib', 'compare', 'sens': ").strip()

    if scenario == 'calib':
        run_calibration_scenario()
    elif scenario == 'compare':
        run_comparison_with_variants()
    elif scenario == 'sens':
        run_sensitivity_scenario()
    elif scenario in ['fin', 'soc']:
        # For 'fin' or 'soc' scenario types, ask for a price scenario
        try:
            price_scenario = int(input("Enter the price scenario (19, 22, or 23): ").strip())
            if price_scenario not in [19, 22, 23]:
                raise ValueError
            run_single_scenario(scenario, price_scenario)
        except ValueError:
            print("Invalid price scenario. Please enter 19, 22, or 23.")
    else:
        print("Invalid scenario selection. Please choose 'fin', 'soc', 'calib', 'compare' or 'sens'.")



        
if __name__ == "__main__":
    main()
    