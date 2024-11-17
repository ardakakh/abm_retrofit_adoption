# ABM of retrofit adoption
This project implements an agent-based model (ABM) for retrofit adoption analysis using Python and the Mesa framework. The model simulates decision-making processes among homeowners and evaluates policy impacts on energy-efficient retrofitting. 

## Features
- Agent-based modeling using Python and Mesa.
- Sensitivity analysis for key model parameters.
- Data visualization and comparison of simulation results.
- Organized input and output data management.

## Project Structure
```
abm_retrofit_adoption/
├── data/                 # Input data (Excel files, etc.)
├── src/                  # Source code (e.g., model.py, sensitivity.py)
├── results/              # Output data (Excel files, plots)
│   ├── calibration/      # Calibration outputs
│   ├── sensitivity/      # Sensitivity analysis outputs
├── environment.yml       # Conda environment specification
├── README.md             # Project description and instructions
├── config.py             # Centralized paths/configuration
└── requirements.txt      # Pip dependencies (optional)
```

## Setup Instructions

To set up the environment for this project, follow these steps:

### 1. Clone the Repository
git clone https://github.com/yourusername/yourproject.git
cd yourproject

### 2. Set Up the Environment
You can set up the environment using Conda or Pip.

#### Option 1: Using Conda
Create the environment using the environment.yml file:
```conda env create -f environment.yml```

Activate the environment:
```conda activate my_environment```

#### Option 2: Using Pip
If you're using Pip, install the dependencies from requirements.txt:
```pip install -r requirements.txt```

### 3. Verify the Setup
Ensure all dependencies are installed correctly by running:
```python --version
pip list ```

## How to Run the Project

After activating the environment, you can run the scripts as needed:
python src/scenario_manager.py

Scenario manager allows to run:
- single runs (fin or soc models with selected variables)
- comparison runs (between fin and soc for several price scenarios)
- sensitivity analyses
- calibration of the model

## Dependencies
This project requires: 
- Python (>= 3.8)
- Mesa
- Pandas
- NumPy
- Matplotlib
See environment.yml or requirements.txt for full dependency details.

## Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## **Reference**

If you use this code or find it helpful, please cite the following article:

> Your Name, _"Title of Your Article"_, Journal Name, Volume(Issue), Pages, Year. DOI: [10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx)

## Contact
For questions or feedback, please contact:

Email: ardak.akhatova@gmail.com
GitHub: @ardakakh


