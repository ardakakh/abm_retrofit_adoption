
import itertools

import mesa
from agents import Homeowner
import pandas as pd
import scipy.stats as stats
from scipy.stats import weibull_min
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from config import DATA_DIR, RESULTS_DIR



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#pd.options.mode.chained_assignment = None


class RetrofitABM(mesa.Model):
    
    """
    This model ... (description)
    User-settable PARAMETERS are defined under 
    "__init__(
    param1,
    param2,
    ...)" 
    """
    
    # id generator to track run number in batch run data
    id_gen = itertools.count(1)
    
    
    def __init__(
            self,
            uniform_breakdowns=True,
            n_agents = 100,
            n_steps = 20,
            soc = True, 
            price_scenario ='hist',
            share_det_ter=0.2,
            n_consider = 3,
            seed = 4,
            gas_boiler = True, plot_distr=True,
            sens_elprices = False, sens_gasprices = False, 
            sens_ins = False, sens_hp = False,
            ins_costs = 2, hp_costs = 2, 
            base_price_el = 1, base_price_gas = 1, 
            rest_v_ins = True, # there is a rest value of insulation after 20 years 
            randomness = False, 
            current_year = 2019,
            gas_ban = False, ban_year = 2026,
            ###!!!!! subsidy
            subsidy_hp = 0.3, subsidy_ins = True,
            
            # input parameters
            n_options = 38,
            
            # heat pump parameters
            #scop1=2.1, scop2=2.5, scop3=2.8, scop4=3.0, scop5=3.5, scop6=4.0, 
            scop_nonren = 2.3,
            scop_ren = 3.1, scop_r_min = 2.5, scop_r_max = 3.8,
            c_maint_hp_avg = 300,
            
            # gas boiler
            gas_eff_new=0.95, gas_eff_old=0.95, c_maint_gb_avg = 150, 
            co2_factor_ng = 0.203, # kg/kWh
            co2_factor_el = 0.421,
            
            # economic parameters
            discount_rate=0.05, rest_value=0.33, 
            
            # social/tpb parameters
            mean_att=0.25, st_dev_att=0.1, 
            int_thr=0.3, cx_hp_max=0.9, cx_hp_min=0.3, cx_gas_max=0.6, 
            cx_gas_min=0.1,
            mean_pbc = 0.3,  st_dev_pbc = 0.1, lower_pbc=0, upper_pbc=1,
            w_att = 0.47, w_sn = 0.19, w_pbc = 0.34,
            
            # social influence
            mu=0.25, n_contact=3, lower_att=0.05, upper_att=0.95, 
            

            # socio-econ / budget?            
            ): 
        
        super().__init__()
        self.uid = next(self.id_gen)
        self.num_agents = n_agents
        self.n_steps = n_steps
        self.soc = soc
        self.price_scenario = price_scenario
        # Load retrofit options once for all building archetypes
        self.retrofit_options = pd.ExcelFile(os.path.join(DATA_DIR, 'input_retrofit_packages.xlsx'))
        self.randomness = randomness
        
        if self.randomness == False:
            self.rng = np.random.default_rng(seed)
            
        self.uniform_breakdowns = uniform_breakdowns
        if self.uniform_breakdowns:
            self.breakdown_schedule = self._generate_fixed_breakdown_schedule()
    
        self.heating_system_breakdowns = {step: 0 for step in range(1, self.n_steps + 1)}


        self.plot_distr = plot_distr
        self.current_year = current_year
        self.gas_boiler = gas_boiler
        self.rest_v_ins = rest_v_ins
        self.gas_ban = gas_ban
        self.ban_year = ban_year

        # sensitivity analyses
        self.sens_elprices = sens_elprices
        self.sens_gasprices = sens_gasprices
        self.base_price_el = base_price_el
        self.base_price_gas = base_price_gas
        
        self.sens_ins = sens_ins
        self.sens_hp = sens_hp
        self.ins_costs = ins_costs
        self.hp_costs = hp_costs     
        # step counter
        self.step_count = 0
        
        # Load buildings data from Excel file
        buildings = pd.read_excel(os.path.join(DATA_DIR, 'input_buildings.xlsx'))
        
        # Adjust n_buildings according to n_agents and share_det_ter
        self._adjust_buildings(buildings, share_det_ter)
        
        # Create a dictionary mapping building archetypes to their respective retrofit options
        self.retrofit_options_dict = {
            arch: pd.read_excel(self.retrofit_options, sheet_name=arch)
            for arch in buildings["Building archetype"].unique()}
        
        self.num_considering = 0
        self.num_with_intention = 0
        self.num_boiler_breakdown_adoptions = 0
        
        self.agents_considering = []
        self.agents_with_intention = []
        self.agents_boiler_breakdown = []
        
        self.stepwise_adoptions = {
           "intention": [],  # To store adopted packages by intention per step
           "breakdown": []   # To store adopted packages by breakdown per step
       }
        self.adoption_history = []
        
        #print(self.retrofit_options_dict)
        # technology parameters - heat pump
        self.scop_nonren = scop_nonren
        self.scop_ren = scop_ren
        self.scop_r_min = scop_r_min
        self.scop_r_max = scop_r_max
        self.c_maint_hp_avg = c_maint_hp_avg
        
        # technology parameters - gas boiler
        self.gas_eff_old = gas_eff_old
        self.gas_eff_new = gas_eff_new
        self.c_maint_gb_avg = c_maint_gb_avg
        self.co2_factor_ng = co2_factor_ng
        self.co2_factor_el = co2_factor_el
        
        # Price projections - always load from the Excel file
        prices_el = pd.read_excel(os.path.join(DATA_DIR, 'input_energy_prices.xlsx'), sheet_name ='NL_20'+price_scenario)
        prices_gas = pd.read_excel(os.path.join(DATA_DIR, 'input_energy_prices.xlsx'), sheet_name='NL_20'+price_scenario)
        
        self.price_proj_el = prices_el.iloc[0].values
        self.price_proj_gas = prices_gas.iloc[1].values
        
        # For sensitivity analysis
        if sens_elprices:
            self._adjust_price_projection(self.price_proj_el, base_price_el)
        if sens_gasprices:
            self._adjust_price_projection(self.price_proj_gas, base_price_gas)
        
        self.n_consider = n_consider
        # subsidy available
        self.subsidy_hp = subsidy_hp
        self.subsidy_ins = subsidy_ins
        self.discount_rate = discount_rate
        self.rest_value = rest_value
        
        #tpb parameters
        self.int_thr = int_thr
        self.cx_hp_max = cx_hp_max
        self.cx_hp_min = cx_hp_min
        self.cx_gas_max = cx_gas_max
        self.cx_gas_min = cx_gas_min
        
        # opinion dynamics parameters
        self.mu = mu #higher mu - more responsive to others' opinions
        self.n_contact = n_contact
        self.lower_att = lower_att 
        self.mean_att = mean_att
        self.st_dev_att = st_dev_att
        self.upper_att = upper_att
        self.n_options = n_options
        
        self.w_att = w_att
        self.w_sn = w_sn
        self.w_pbc = w_pbc
        
        self.mean_pbc = mean_pbc
        self.st_dev_pbc = st_dev_pbc
        self.lower_pbc = lower_pbc
        self.upper_pbc = upper_pbc
        
        # Generate attitude and PBC distributions
        self.att_distr = self._generate_attitude_distribution()
        self.opi_distr = -1 + 2 * self.att_distr  # Transform to opinion distribution
        self.pbc_distr = self._generate_pbc_distribution()
        
        # Calculate total heat demand and emissions before retrofitting
        self._calculate_totals(buildings)
        
        # Generate distributions for building age and heating systems
        self.age_distr65, self.age_distr75 = self._generate_building_age_distributions()
        self.noncond_distr, self.cond_distr1, self.cond_distr2, self.hp_life_distr = self._generate_heating_system_distributions()
        
        # Plot distributions if requested
        if self.plot_distr:
            self.plot_heating_system_distributions()
            self.plot_heating_system_densities()
            self.plot_attitude_and_pbc_distributions()
        
        self.schedule = mesa.time.StagedActivation(self, shuffle=False)

        # for batch run
        self.running = True
        
        self._create_agents(buildings)
        
        # datacollectors
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Num adopters": lambda m: sum(a.adoption_status for a 
                                              in m.schedule.agents), # m.schedule.get_agent_count() 
                "Share adopters": lambda m: sum(a.adoption_status for a 
                                              in m.schedule.agents)/self.num_agents,
                # dictionary counting options adopted
                "Adopted options cumulative": lambda m: adoptions_cumul(m), 
                "Adopted options by step": lambda m: adoptions_by_step(m),
                "Current HS and INS": lambda m: adoptions_by_group(m),
                "Heat demand reduction (share)": lambda m: sum(a.heat_red/m.tot_heat_before 
                                                               for a in m.schedule.agents),
                #"Useful energy after": useful_en_after, 
                "Carbon emission reduction": lambda m: sum(a.co2_red  
                                                    for a in m.schedule.agents)/m.tot_emis_before,
                "Share HP adoption": lambda m: share_hp_adoption(m),
                
                # "Agents considering": lambda m: len([a for a in m.schedule.agents 
                #                                 if a.considering == True]),
                # "Considering: price trigger": lambda m: len([a for a in m.schedule.agents 
                #                                               if (a.price_trigger() == True) and
                #                                             (a.boiler_breaks == False) and 
                #                                             (a.intention_status==False)]),
                # "Considering: boiler breakdown": lambda m: len([a for a in 
                #                                     m.schedule.agents if (a.boiler_breaks == True) and 
                #                                     (a.intention_status==False)]),
                # "Considering: boiler age": lambda m: len([a for a in m.schedule.agents 
                #                                                 if a.years_left < self.n_consider]),
                # "Agents intending": lambda m: len([a for a in m.schedule.agents 
                #                                                    if a.intention_status == True]),
                "Agents decided": lambda m: len([a for a in m.schedule.agents 
                                                 if a.step_adopted == m.step_count]),
                "Two times adopter": lambda m: len([a for a in m.schedule.agents 
                                                 if a.adoption_status==2]),
                "First adoption": lambda m: [a.first_option for a in m.schedule.agents],
                
                "Second adoption": lambda m: [a.second_option for a in m.schedule.agents],
                    
                },
            
           
            agent_reporters={"Opinion": "opi",
                            "Step considered": "step_considered",
                            "Step adopted": "step_adopted",
                            #"Options NPV":"options_npv",
                            "Options utilities": "options_util",
                            "Adopted option": "adopted_option_name",
                            "Boiler breakdown": "boiler_breaks",
                            "Building archetype": "building_arc",
                            "Building age": "building_age",
                            "Intention": "int",
                            "Attitude": "att",
                            "Considering": "considering",
                            "Intending": "intention_status",  
                            #"Adopted": "adopted_this_step", 
                            })
 
    def _calculate_totals(self, buildings):
        """
        Calculate total heat demand and emissions before retrofitting.
        """
        self.tot_heat_before = round(
            sum(buildings['n_buildings'] * buildings['Specific heating demand before [kWh/m2]'] * buildings['Reference floor area [m2]']),
            0
        )
        self.tot_emis_before = self.tot_heat_before * self.co2_factor_ng
         
         # self.tot_heat_before = round(sum(buildings['n_buildings']*buildings[
         #     'Specific heating demand before [kWh/m2]']*buildings['Reference floor area [m2]']),0)
         
         # self.tot_emis_before = self.tot_heat_before*self.co2_factor_ng
    
    
    def _generate_fixed_breakdown_schedule(self):
        """
        Generate a fixed breakdown schedule to ensure 6% of agents replace boilers each step.
        """
        total_agents = self.num_agents
        breakdown_percent = 6
        breakdowns_per_step = int(total_agents * breakdown_percent / 100)
        
        # Create a breakdown schedule with fixed numbers of breakdowns per step
        breakdown_schedule = []
        rng = np.random.default_rng(4)  # Seeded random generator
    
        # Distribute breakdowns evenly across steps
        for step in range(1, self.n_steps + 1):
            breakdown_schedule.extend([step] * breakdowns_per_step)
    
        # Handle any remaining agents
        remaining_agents = total_agents - len(breakdown_schedule)
        if remaining_agents > 0:
            extra_steps = rng.choice(range(1, self.n_steps + 1), size=remaining_agents, replace=False)
            breakdown_schedule.extend(extra_steps)
    
        rng.shuffle(breakdown_schedule)  # Shuffle to randomize agent assignment
        return breakdown_schedule


    def _generate_building_age_distributions(self):
        """
        Generate uniform distributions for building ages.
        """
        rng = np.random.default_rng(4)
        lower_age65, upper_age65 = (self.current_year - 1974), (self.current_year - 1965)
        age_distr65 = rng.uniform(lower_age65, upper_age65, self.num_agents).astype(int)
        rng = np.random.default_rng(4)
        lower_age75, upper_age75 = (self.current_year - 1991), (self.current_year - 1975)
        age_distr75 = rng.uniform(lower_age75, upper_age75, self.num_agents).astype(int)

        return age_distr65, age_distr75

    def _generate_heating_system_distributions(self):
        """
        Generate Weibull distributions for heating system lifespans and convert to integers.
        """
        noncond_distr = self._generate_weibull_distribution(k=25.0, lambd=25.0, size=self.num_agents)
        cond_distr1 = self._generate_weibull_distribution(k=15.0, lambd=15.0, size=self.num_agents)
        cond_distr2 = self._generate_weibull_distribution(k=16.0, lambd=16.0, size=self.num_agents)
        hp_life_distr = self._generate_weibull_distribution(k=20.0, lambd=20.0, size=self.num_agents)
    
        return noncond_distr, cond_distr1, cond_distr2, hp_life_distr
    
    def _generate_weibull_distribution(self, k, lambd, size):
        """
        Generate Weibull distribution and convert to integer.
        """
        rng = np.random.default_rng(4)
        return (rng.weibull(k, size) * lambd).astype(int)

    def plot_heating_system_distributions(self):
        # Define the data and titles
        distributions = [
            (self.noncond_distr, "Non-Condensing Gas Boiler Lifespan"),
            (self.cond_distr1, "Condensing Gas Boiler Lifespan 1"),
            (self.cond_distr2, "Condensing Gas Boiler Lifespan 2"),
            (self.hp_life_distr, "Heat Pump Lifespan")
        ]
        
        # Define the Viridis colormap
        viridis = cm.get_cmap('viridis', len(distributions))
        
        # Plot each distribution in a separate figure
        for i, (data, title) in enumerate(distributions):
            plt.figure(figsize=(4, 3), dpi=300)
            plt.hist(data, bins=20, alpha=0.7, color=viridis(i / len(distributions)))
            plt.title(title, fontsize=12)
            plt.xlabel("Lifespan (years)", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.grid(alpha=0.5, linestyle="--")
            plt.tight_layout()
            
            # Save the plot
            filename = f"{title.replace(' ', '_').lower()}.png"
            save_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')            
            plt.show()
            
    def plot_heating_system_densities(self):
        # Define the data and titles
        distributions = [
            (self.noncond_distr, "Non-Condensing Gas Boiler Lifespan"),
            (self.cond_distr1, "Condensing Gas Boiler Lifespan 1"),
            (self.cond_distr2, "Condensing Gas Boiler Lifespan 2"),
            (self.hp_life_distr, "Heat Pump Lifespan")
        ]
        
        # Define the Viridis colormap
        viridis = cm.get_cmap('viridis', len(distributions))
        
        for i, (data, title) in enumerate(distributions):
            # Fit the data to a Weibull distribution
            shape, loc, scale = weibull_min.fit(data, floc=0)  # Fix location to 0 for meaningful parameters
            
            # Generate x values for the Weibull PDF
            x = np.linspace(min(data), max(data), 1000)
            pdf = weibull_min.pdf(x, shape, loc, scale)
            
            # Plot histogram and Weibull PDF
            plt.figure(figsize=(4, 3), dpi=300)
            plt.hist(data, bins=20, density=True, alpha=0.7, color=viridis(i / len(distributions)), label="Empirical Density")
            plt.plot(x, pdf, color="black", linestyle="--", linewidth=2, label="Weibull PDF")
            
            # Add title, labels, and legend
            plt.title(title, fontsize=12)
            plt.xlabel("Service Life (years)", fontsize=10)
            plt.ylabel("Density", fontsize=10)
            plt.grid(alpha=0.5, linestyle="--")
            plt.legend(fontsize=10)
            
            # Add Weibull parameters to the plot
            plt.text(0.95, 0.85, f"$k$ = {shape:.2f}\n$\\lambda$ = {scale:.2f}", 
                     transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"{title.replace(' ', '_').lower()}_weibull.png"
            save_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        
    def plot_attitude_and_pbc_distributions(self):
        """
        Plot histograms of attitude and PBC distributions with overlaid Beta PDFs.
        """
        # Define the data and titles
        distributions = [
            (self.att_distr, "Attitude Distribution", self.mean_att, self.st_dev_att),
            (self.pbc_distr, "PBC Distribution", self.mean_pbc, self.st_dev_pbc)
        ]
        
        # Define the Viridis colormap
        viridis = cm.get_cmap('viridis', len(distributions))
        
        for i, (data, title, mean, st_dev) in enumerate(distributions):
            # Fit the data to a Beta distribution
            sigma_sq = st_dev ** 2
            alpha = mean * ((mean * (1 - mean) / sigma_sq) - 1)
            beta = (1 - mean) * ((mean * (1 - mean) / sigma_sq) - 1)
            
            # Generate x values for the Beta PDF
            x = np.linspace(0, 1, 1000)
            pdf = stats.beta.pdf(x, alpha, beta)
            
            # Plot histogram and Beta PDF
            plt.figure(figsize=(4, 3), dpi=300)
            plt.hist(data, bins=20, density=True, alpha=0.7, color=viridis(i / len(distributions)), label="Empirical Density")
            plt.plot(x, pdf, color="black", linestyle="--", linewidth=2, label="Beta PDF")
            
            # Add title, labels, and legend
            plt.title(title, fontsize=12)
            plt.xlabel("Value", fontsize=10)
            plt.ylabel("Density", fontsize=10)
            plt.grid(alpha=0.5, linestyle="--")
            plt.legend(fontsize=10)
            
            # Add Beta parameters to the plot
            plt.text(0.95, 0.85, f"$\\alpha$ = {alpha:.2f}\n$\\beta$ = {beta:.2f}", 
                     transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"{title.replace(' ', '_').lower()}_beta.png"
            save_path = os.path.join(RESULTS_DIR, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    def _generate_attitude_distribution(self):
        """
        Generate the attitude distribution using a truncated normal distribution.
        """
        rng = np.random.default_rng(4)
        sigma_sq_att = round(self.st_dev_att ** 2, 2)
        alpha = self.mean_att * ((self.mean_att * (1 - self.mean_att) / sigma_sq_att) - 1)
        beta = (1 - self.mean_att) * ((self.mean_att * (1 - self.mean_att) / sigma_sq_att) - 1)
        return rng.beta(alpha, beta, self.num_agents).round(2)

    def _generate_pbc_distribution(self):
        """
        Generate the perceived behavioral control (PBC) distribution.
        """
        rng = np.random.default_rng(4)
        sigma_sq_pbc = self.st_dev_pbc ** 2
        alpha = self.mean_pbc * ((self.mean_pbc * (1 - self.mean_pbc) / sigma_sq_pbc) - 1)
        beta = (1 - self.mean_pbc) * ((self.mean_pbc * (1 - self.mean_pbc) / sigma_sq_pbc) - 1)
        return rng.beta(alpha, beta, self.num_agents).round(2)
        
        
        # #print('\n Total emissions before: ', self.tot_emis_before)
        # #print('\n TOtal final heat demand before: ', self.tot_heat_before)

    
    def _adjust_price_projection(self, price_proj, base_price):
        """
        Adjust the price projections based on sensitivity analysis.
        """
        price_proj[0] = base_price
        for i in range(19):
            price_proj[i+1] = price_proj[0] * (1 + (i + 1) / 20)
            price_proj[i+21] = price_proj[20] * (1 + (i + 1) / 20)

    
    def _define_stages(self):
       """
       Define stages for agent behavior based on whether the scenario is social or financial.
       """
       if self.soc:
           return ["consider", "fin_eval", "intention", "decision", "opinion_dynamics"]
       return ["consider", "fin_eval", "decision"]
   
    def _adjust_buildings(self, buildings, share_det_ter):
     """
     Adjust the number of buildings for each archetype based on the total number of agents and the ratio of detached to terraced houses.
     This method dynamically adjusts the 'n_buildings' column to ensure that it sums to the total number of agents.
     """
     total_agents = self.num_agents
     archetypes = buildings['Building archetype'].unique()  # Unique building archetypes
     num_archetypes = len(archetypes)  # Number of unique archetypes

     # If the sum of 'n_buildings' in the buildings file doesn't match the number of agents, adjust
     if sum(buildings['n_buildings']) != total_agents:
         # Allocate the buildings dynamically based on share_det_ter (detached + semi-detached to terraced houses)
         num_det_ter = int(share_det_ter * total_agents)  # Number of detached + semi-detached
         num_terraced = total_agents - num_det_ter  # Number of terraced houses

         # Create the new distribution for 'n_buildings'
         new_values = []

         # We will assume the first half of archetypes are detached + semi-detached, and the second half are terraced houses
         det_ter_archetypes = archetypes[:num_archetypes // 2]  # First half is detached + semi-detached
         terraced_archetypes = archetypes[num_archetypes // 2:]  # Second half is terraced

         # Assign the detached + semi-detached count equally across the corresponding archetypes
         for archetype in det_ter_archetypes:
             new_values.append(int(num_det_ter / len(det_ter_archetypes)))

         # Assign the terraced house count equally across the corresponding archetypes
         for archetype in terraced_archetypes:
             new_values.append(int(num_terraced / len(terraced_archetypes)))

         # Update the 'n_buildings' column in the buildings DataFrame
         buildings['n_buildings'] = new_values

         # Save the adjusted buildings data back to the Excel file (optional) 
         buildings.to_excel(os.path.join(DATA_DIR, "input_buildings.xlsx"), index=False)
   
    def _create_agents(self, buildings):
        """
        Create agents for the simulation based on building archetypes.
        """
        num_arch = len(buildings.index)
        agent_id = 0

        for i in range(num_arch):
            num_buildings_current_arch = buildings['n_buildings'][i]
            building_archetype = buildings.iloc[i]["Building archetype"]
            retrofit_options_for_arch = self.retrofit_options_dict[building_archetype]
            
            # Loop over the number of buildings for the current archetype
            for _ in range(num_buildings_current_arch):
                # Sample parameters for the agent
                
                h = Homeowner(agent_id, self, buildings.iloc[i],
                              retrofit_options_for_arch,  # Pass only relevant retrofit options
                              age_distr65=self.age_distr65[agent_id],
                age_distr75=self.age_distr75[agent_id],
                noncond_life=self.noncond_distr[agent_id],
                cond_life1=self.cond_distr1[agent_id],
                cond_life2=self.cond_distr2[agent_id],
                hp_life=self.hp_life_distr[agent_id],
                attitude=self.att_distr[agent_id],
                pbc=self.pbc_distr[agent_id]
                              )
                self.schedule.add(h)
                agent_id += 1
    
    def step(self):
        #tell all the agents in the model to run their step function
        #print('*********MODEL STEP STARTS********************')
        self.step_count += 1
        
        # Reset tracking lists at the beginning of each step
        self.stepwise_adoptions["intention"] = []
        self.stepwise_adoptions["breakdown"] = []
        
        if self.gas_ban == True:
            if self.step_count>=(self.ban_year-self.current_year)+1:
                self.gas_boiler = False
        print('step_count: ', self.step_count)
        
        self.schedule.step()
        
        #Display tracking information
        # print("Number of agents considering:", self.num_considering)
        # print("Agents considering:", self.agents_considering)
        # print("Number of agents with intention:", self.num_with_intention)
        # print("Agents with intention:", self.agents_with_intention)
        # print("Number of agents adopting due to boiler breakdown:", self.num_boiler_breakdown_adoptions)
        # print("Agents adopting due to boiler breakdown:", self.agents_boiler_breakdown)
        
        # Reset tracking variables for the next step
        self.num_considering = 0
        self.num_with_intention = 0
        self.num_boiler_breakdown_adoptions = 0
        self.agents_considering = []
        self.agents_with_intention = []
        self.agents_boiler_breakdown = []        
         
        
        
        for agent in self.schedule.agents:
            agent.building_age +=1 
        self.current_year += 1
        self.datacollector.collect(self)
        # Print share of heat pump adoption for the current step
        share_hp = share_hp_adoption(self)
        
        # Record adoption data for this step
        self.adoption_history.append({
            "step": self.schedule.steps,
            "intention_adoptions": self.stepwise_adoptions["intention"].copy(),
            "breakdown_adoptions": self.stepwise_adoptions["breakdown"].copy()
        })
        
        #print(f"Step {self.step_count}: Share of heat pump adoption = {share_hp:.2f}")
        #print('*********MODEL STEP ENDS********************')
        
        

    def run_model(self):
        """
        Run the model for a specified number of steps.
        """
        for _ in range(self.n_steps):
            self.step()
            
    def get_results(self):
        """
        Retrieve the data collected by DataCollector after the model run.
        """
        return self.datacollector.get_model_vars_dataframe()
         

# def share_adoption(m):
#     """
#     A method to calculate the total share of adopters
#     """
#     agent_adoptions = [agent.adoption_status for agent in m.schedule.agents]
#     return sum(agent_adoptions)/n_agents

def adoptions_by_step(m):
    """
    A method to calculate the number of agents adopting each retrofitting package
    at a specific time step.
    """
    
    adopted_options = [agent.adopted_option_name for agent in m.schedule.agents
                       if agent.step_adopted == m.step_count]
    package_order = m.schedule.agents[0].retrofit_options.index
    options_dict = {package: 0 for package in package_order}
    
    # Count the occurrence of each option and calculate the share
    for option in adopted_options:
        if option in options_dict:
            options_dict[option] += 1
            
    # # Calculate the share for each option
    # for option in options_dict:
    #     options_dict[option] /= m.num_agents

    return options_dict


def adoptions_by_group(m):
    """
    A method to calculate the current stock of heating system and insulation. 
    The existing heating system + insulation state can be one of the following
    four: gas boiler uninsulated, gas boiler insulated, heat pump uninsulated, 
    heat pump insulated. 
    """
    
    # Initialize counters for each heating system and insulation combination
    options_counts = {
        'gas boiler, uninsulated': 0,
        'gas boiler, insulated': 0,
        'heat pump, uninsulated': 0,
        'heat pump, insulated': 0
    }
    
    # Iterate over all agents and update the counters based on their attributes
    for agent in m.schedule.agents:
        insulation_status = 'uninsulated' if agent.current_ins=='none' else 'insulated'
        hs_status = 'gas boiler' if (agent.current_hs == 'gas boiler' or agent.current_hs == 'gas boiler new') else 'heat pump'
        key = f"{hs_status}, {insulation_status}"
        options_counts[key] += 1
        #options_counts['gas boiler, uninsulated'] -= 1

    # # Calculate the total number of agents to normalize the counts into shares
    # total_agents = len(m.schedule.agents)

    # # Convert counts to shares
    # options_shares = {key: count / total_agents for key, count in options_counts.items()}

    return options_counts


def adoptions_cumul(m):
    """
    A method to calculate the number of agents adopting each retrofitting package
    at a specific time step.
    """
    # Assuming each agent has a 'step_adopted' attribute indicating when they adopted the option
    adopted_packs_first = [agent.adopted_option_name for agent in m.schedule.agents
                         if agent.adoption_status == 1]
    adopted_packs_second = [agent.adopted_option_name for agent in m.schedule.agents
                         if agent.adoption_status == 2]
    adopted = adopted_packs_first + adopted_packs_second
    # print(adopted)
    # print(len(adopted))
    
    package_order = m.schedule.agents[0].retrofit_options.index
    options_dict = {package: 0 for package in package_order}
    
    # Count the occurrence of each option
    for option in adopted:
        if option in options_dict:
            options_dict[option] += 1
            
    return options_dict

def share_hp_adoption(m):
    """
    A method to calculate the share of heat pump adoption among the 
    total gas boiler replacements in the current step.
    """
    current_step = m.step_count
    hp_adoptions = 0
    total_adoptions = 0

    # Iterate over agents to count adoptions
    for agent in m.schedule.agents:
        # Debug: Print each agent's adopted option for the current step
        #print(f"Agent {agent.unique_id}: adopted_option_name = {agent.adopted_option_name}, step_adopted = {agent.step_adopted}")

        # Check if the agent adopted in the current step
        if agent.step_adopted == current_step:
            total_adoptions += 1
            if agent.adopted_option_name.lower().startswith(("heat pump", "hp")):
                hp_adoptions += 1

    # Calculate and print the share of HP adoptions
    if total_adoptions > 0:
        share_hp = hp_adoptions / total_adoptions
    else:
        share_hp = 0  # To avoid division by zero

    print(f"Step {current_step}: HP adoptions = {hp_adoptions}, Total adoptions = {total_adoptions}, Share HP adoption = {share_hp}")

    return share_hp



# Global variable to store model results
results = None


def run(soc=True, price_scenario=19, num_steps=20):
    """
    Function to run the model for a specific scenario and return results.
    Args:
        soc (bool): Social or financial scenario.
        price_scenario (int): Price scenario to use (e.g., 19, 22, 23).
    
    Returns:
        pandas.DataFrame: Results from the model run.
    """
    global results
    model = RetrofitABM(soc=soc, price_scenario=str(price_scenario))
    
    for _ in range(num_steps):
        model.step()
    return model


def get_results(self):
    """
    Retrieve the results of the last model run from DataCollector.
    """
    return self.datacollector.get_model_vars_dataframe()


if __name__ == "__main__":
    # Run the model for a default scenario if script is run directly
    model = run(soc=True, price_scenario=19)
    

# def print_adoption_per_step(model):
#     for step_data in model.adoption_history:
#         print(f"Step {step_data['step']}:")
#         print("  Intention-based adoptions:", step_data["intention_adoptions"])
#         print("  Breakdown-based adoptions:", step_data["breakdown_adoptions"])
        
        
# def plot_adoptions_per_step(model):
#     steps = [data["step"] for data in model.adoption_history]
#     intention_counts = [len(data["intention_adoptions"]) for data in model.adoption_history]
#     breakdown_counts = [len(data["breakdown_adoptions"]) for data in model.adoption_history]
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps, intention_counts, label="Intention-based Adoptions", marker="o")
#     plt.plot(steps, breakdown_counts, label="Breakdown-based Adoptions", marker="x")
    
#     plt.xlabel("Simulation Step")
#     plt.ylabel("Number of Adoptions")
#     plt.title("Number of Adopted Packages per Step (Intention vs Breakdown)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
# print_adoption_per_step(model)  # Print detailed adoptions per step
# plot_adoptions_per_step(model)  # Plot number of adoptions per step


# def print_heating_system_breakdowns(self):
#     """
#     Print the number of heating systems breaking down per step.
#     """
#     print("Heating System Breakdowns:")
#     for step, count in self.heating_system_breakdowns.items():
#         print(f"Step {step}: {count} breakdowns")
        
        
        
# # Generate and analyze breakdown schedule
# breakdown_schedule = model.breakdown_schedule
# distribution = Counter(breakdown_schedule)

# # Plot breakdown distribution
# plt.bar(distribution.keys(), distribution.values(), color='blue', alpha=0.7)
# plt.xlabel("Simulation Step")
# plt.ylabel("Number of Breakdowns")
# plt.title("Breakdown Distribution (6% Per Step)")
# plt.grid(axis='y')
# plt.show()

# # Print the distribution for inspection
# print(sorted(distribution.items()))