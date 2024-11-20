# Configuration file for centralized paths

import os
import sys

# Base directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Specific data file paths
INPUT_BUILDINGS = os.path.join(DATA_DIR, "input_buildings.xlsx")
INPUT_ENERGY_PRICES = os.path.join(DATA_DIR, "input_energy_prices.xlsx")
INPUT_RETROFIT_PACKAGES = os.path.join(DATA_DIR, "input_retrofit_packages.xlsx")

# Example usage of paths
if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("SRC_DIR:", SRC_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("Input Buildings Path:", INPUT_BUILDINGS)

# Directory for the scenario results
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create the directory if it doesn't exist

SINGLE_RUN_DIR = os.path.join(RESULTS_DIR, "single run")
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create the directory if it doesn't exist

BATCH_RUN_DIR = os.path.join(RESULTS_DIR, "batch run")
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Directory for calibration outputs
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")
os.makedirs(CALIBRATION_DIR, exist_ok=True)  # Create the folder if it doesn't exist

# Directory for sensitivity analysis outputs
SENSITIVITY_DIR = os.path.join(RESULTS_DIR, 'sensitivity')
os.makedirs(SENSITIVITY_DIR, exist_ok=True)  # Create the folder if it doesn't exist