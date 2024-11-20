""" ABM of net-zero integrated retrofitting:
    Agents
"""

import mesa
import numpy as np
import pandas as pd
import random
import math
from collections import Counter


class Homeowner(mesa.Agent):
    
    """ A Homeowner agent is an owner-occuppier of a single family 
    building (defined in Building Class). Agents are heterogeneous 
    according to: 
        - owned building (4 archetypes) 
        - disposable income
        - attitude towards renovation (opinion)
        - weight factors for financial and environmental evaluation
    """
    
    def __init__(self, unique_id, model, building, retrofit_options, 
                 age_distr65, age_distr75, noncond_life, cond_life1, cond_life2, 
                 hp_life, attitude, pbc): 

        # initialize the parent class with required parameters
        super().__init__(unique_id, model)
        
        #print(self.unique_id)
        #### indicators (with initial values)
        self.state = 'none' # Possible states: None, "considering", "evaluating", "intending"
        self.adoption_status = 0  # Tracks if agent has adopted at any point
        self.step_adopted = -1  # Tracks the specific step when the agent adopted
        
        self.adopted_option = 'none'
        self.adopted_option_name = 'none'  
                
        self.current_hs = 'gas boiler'
        self.current_ins = 'none'
        #### KPIs
        self.num_adoptions = 0
        self.en_use_after = 0
        self.heat_red = 0
        self.co2_before = 0
        self.co2_red = 0
        self.co2_after = 0
        self.cost_inv = 0
        self.cost_env = 0
        self.cost_hp = 0
        self.walls = False
        self.floor = False
        self.windows = False
        self.roof = False 
        
        if self.model.randomness == False:
            random.seed(4)
            
        #### Techno-econ parameter
        ### agent's building (assigned at model initialization, from buildings.xlsx)
        self.building = building
        self.building_arc = self.building['Building archetype']
        #print(self.building_arc)
        self.building_area = self.building['Reference floor area [m2]']        
        if self.building['age_begin']==1975:
            self.building_age = age_distr75 
        else: 
            self.building_age = age_distr65
        #print(self.building_age, self.building_arc)
        ### suitable retrofit options for an agent's building
        
        if self.model.gas_boiler==True:
            self.retrofit_options = retrofit_options
            #self.rp_gb = self.retrofit_options['Name'].iloc[-1]           
        else: 
            self.retrofit_options = retrofit_options.iloc[:-19]
        self.retrofit_options = retrofit_options

        if self.retrofit_options.index.name != 'Name':
            self.retrofit_options.set_index('Name', inplace=True)
                    
        # These are the values sampled from the model's distributions
        self.age_distr65 = age_distr65
        self.age_distr75 = age_distr75
        self.noncond_life = noncond_life
        self.cond_life1 = cond_life1
        self.cond_life2 = cond_life2
        self.hp_life = hp_life
        
        #TPB parameters - later: initialize for a population
        self.att = attitude
        self.att = min(max(self.att, 0.01), 0.99)  # Constrain self.att between 0.01 and 0.99
        self.sn = 0
        self.int = 0
        
        self.pbc = pbc
        #print(self.unique_id, self.pbc)
        self.w_att = self.model.w_att
        self.w_sn = self.model.w_sn
        self.w_pbc = self.model.w_pbc
        
        self.w_pp = 0.33
        self.w_cx = 0.33
        self.w_es = 1-self.w_pp-self.w_cx
        
        self.int_thr = self.model.int_thr
        self.num_options = self.model.n_options
        
        # agent's opinion  parameters
        
        self.opi = round(2 * self.att - 1, 2)
        self.unc = max(round(2 * (1 - abs(self.opi)), 2), 0.01)  # Minimum threshold for uncertainty
        
        self.years_left = (self.building_age - self.noncond_life) % self.cond_life1
        # Data collection for first and second adoption decisions
        self.first_option = 0  # Stores the first adoption decision (for data collection)
        self.second_option = 0  # Stores the second adoption decision (for data collection)
        
        
            
    def _check_heating_system_breakdown(self, agent):
        """
        Determine if the agent's heating system breaks this step.
        """
        if self.model.uniform_breakdowns:
            # Use precomputed schedule
            is_breaking_down = self.model.step_count == self.model.breakdown_schedule[agent.unique_id]
        else:
            is_breaking_down = self.years_left==0
            
        # Track the breakdown event
        if is_breaking_down:
            #print(f"Step {self.model.step_count}, Agent {agent.unique_id}: years_left = {self.years_left}, schedule_step = {self.model.breakdown_schedule[agent.unique_id]}")
            self.model.heating_system_breakdowns[self.model.step_count] += 1
            
        return is_breaking_down
        
    
        
    def write2r_o(self, var, lst, options):
        """
        Writes a list (lst) to the retrofit options DataFrame as a new column (var).
        If the column already exists, it will be updated.
        """
        if var not in options.columns:
            # Use .loc to explicitly set the column values
            options.loc[:, var] = lst
        else:
            # Use .loc for setting values after dropping the column
            options = options.drop(var, axis=1)
            options.loc[:, var] = lst
            
            
    #Helper Functions
    
    def ret_packages(self, k):
        """
        Returns the name of the k-th retrofit package option.
        """
        self.package_names = []
        for i in range(self.model.n_options):
            self.package_names.append(self.retrofit_options.index[i])
        return self.package_names[k]      
    
    
    def calc_scop(self):
        """
        Calculates the Seasonal Coefficient of Performance (SCOP) for heating systems
        based on the 'Specific heating demand after [kWh/m2]' of retrofit options.
    
        This method performs the following steps:
        1. Retrieves the first 19 retrofit options' specific heating demand after retrofit.
        2. Sorts these options based on their specific heating demand to assign SCOP values.
        3. Creates a SCOP list, with values linearly interpolated between `scop_r_max` and `scop_r_min`.
        4. Reverts the sorted SCOP values to their original order and adds the SCOP to the DataFrame.
    
        Returns:
            pd.DataFrame: A DataFrame with 'Specific heating demand after [kWh/m2]' and corresponding SCOP values.
        """
        options = self.retrofit_options[:19]['Specific heating demand after [kWh/m2]']
        # Save the original order by creating a new column with the current index
        df = options.copy().to_frame(name='Specific heating demand after [kWh/m2]')
        df.reset_index(drop=False, inplace=True)
        df['original_order'] = df.index
        df_sorted=df.sort_values(by='Specific heating demand after [kWh/m2]')
        scops_list = np.round(np.linspace(self.model.scop_r_max, self.model.scop_r_min, 
                                            19),1)
        # print('scops_list', scops_list)
        # print('df_sorted', df_sorted)
        
        df_sorted['SCOP'] = scops_list
        # To "unsort" and revert to the original order, sort by the 'original_order' column
        df_unsorted = df_sorted.sort_values(by='original_order').drop(columns='original_order')
        df_unsorted = df_unsorted.rename(columns={'index': 'Name'})
        df_final = df_unsorted.set_index('Name')
   
        return(df_final)
    
    
    def boiler_replacement(self):
        ''' helper function for self.consider() tells whether boiler should be 
        replaced soon
        1 - first boilers that were installed upon construction were non-condensing,
        i.e. conventional gas boilers. Hence, we deduce the age of that boiler
        from building age.
        2 - then see when the next (condensing) boiler should be replaced by
        dividing it to expected lifetime of a boiler  '''
        
        if self.model.uniform_breakdowns and self._check_heating_system_breakdown(self):
            self.boiler_breaks = True 
            years_left = 0            
        else:
            self.boiler_breaks = False 
            if self.model.step_count > 1:
                years_left = self.years_left-1
            else:
                years_left = self.years_left
           # print('Years left: ', self.years_left)
        return years_left
    
    
    def price_trigger(self):
        ''' identify if the prices of gas and electricity increase or decrease 
        by more than 50% in the current step (compared to the previous step) '''
        trigger = False
        
        if self.model.step_count >1:
            price_now = self.model.price_proj_gas[self.model.step_count-1]
            price_before = self.model.price_proj_gas[self.model.step_count-2]
            if self.current_hs == 'gas boiler':
                x_total = self.cond_life1
            elif self.current_hs == 'gas boiler new':
                x_total = self.cond_life2
            else: 
                x_total = self.hp_life
            
            delta_p = round(((price_now-price_before)/price_before)*100, 0)
            x_thr = round(x_total*math.exp(-delta_p/180), 2)
            
            if (x_total-self.years_left) >= x_thr:
                trigger = True
            
        return trigger
    
    
    def complexity(self):
        """
        Assigns complexity values to each retrofit option.
    
        This method does the following:
        1. Creates a predefined list of complexity values (`cx_list`).
        2. Associates each complexity value with its corresponding retrofit option.
        3. Creates a DataFrame with the complexity values, dropping the 'Cost envelope [EUR]' column.
    
        Returns:
            pd.DataFrame: A DataFrame containing complexity values for each retrofit option.
        """
        cx_list = [0.6, 0.95, 0.9, 0.85, 0.9, 0.8, 0.85, 0.8, 0.8, 0.75, 0.75,
                   0.95, 0.9, 0.85, 0.9, 0.8, 0.85, 0.8, 0.6,
                   0.2, 0.7, 0.65, 0.6, 0.65, 0.55, 0.6, 0.55, 0.5, 0.45, 0.45, 
                   0.7, 0.65, 0.6, 0.65, 0.55, 0.6, 0.55, 0.3]
        
        options = self.retrofit_options.loc[:,'Cost envelope [EUR]'].copy()
        options = options.to_frame(name='Cost envelope [EUR]')
        options['complexity']=cx_list
        options = options.drop('Cost envelope [EUR]', axis=1)
        return options
    
    
    # Data Preparation Functions 
    
    def datafr_el_p(self):
        """
        Creates a DataFrame of electricity prices over a 20-year period
        for each retrofit option.
        """
        # analysis time frame (20 yearsfrom current step)
        current_step = self.model.step_count #step1 = year 1 = 2024 --> iloc[0]
        el_prices_temp = pd.DataFrame(self.model.price_proj_el, columns = [self.ret_packages(0)])
        curr_value = el_prices_temp['heat pump (HP) + none'][current_step-1]
        el_prices_data = [curr_value] * 20 # num_steps
        el_prices_df = pd.DataFrame(el_prices_data, columns=['heat pump (HP) + none'])
        
        for i in range(self.num_options-1):
            el_prices_df[self.ret_packages(i+1)] = el_prices_df[self.ret_packages(0)]
        
        return el_prices_df
    
    def datafr_gas_p(self):
        """
        Creates a DataFrame of gas prices over a 20-year period
        for each retrofit option.
        """
        current_step = self.model.step_count #step1 = year 1 = 2024 --> iloc[0]
        gas_prices_temp = pd.DataFrame(self.model.price_proj_gas, columns = [self.ret_packages(0)])
        curr_value = gas_prices_temp['heat pump (HP) + none'][current_step-1]
        gas_prices_data = [curr_value] * 20 # num_steps
        gas_prices_df = pd.DataFrame(gas_prices_data, columns=['heat pump (HP) + none'])
        
        for i in range(self.num_options-1):
            gas_prices_df[self.ret_packages(i+1)] = gas_prices_df[self.ret_packages(0)]
            
        return gas_prices_df
    
###############################################################################
##############CORE DECISION-MAKING METHODS#####################################
###############################################################################

    def step(self):
        """
        Execute all stages in sequence based on the scenario.
        """
        
        self.consider()

        #print(f"Agent {self.unique_id} state after consider: {self.state}")
        
        if self.model.soc == False:
            if self.state == "considering":
                #print(f"Agent {self.unique_id} is in 'considering' state.")
                self.fin_eval()
            if self.state == "evaluating":
                #print(f"Agent {self.unique_id} is in 'evaluating' state.")
                self.decision()
            #if self.state == "completed":
                #print(f"Agent {self.unique_id} is in 'completed' state.")
                
        else:
            if self.state == "considering":
                #print(f"Agent {self.unique_id} is in 'considering' state.")
                self.fin_eval()
            if self.state == "evaluating" and self.years_left != 0:
                #print(f"Agent {self.unique_id} is in 'evaluating' state.")
                self.intention()
            if self.state == "intending":
                #print(f"Agent {self.unique_id} is in 'intending' state.")
                self.decision()
            elif self.boiler_breaks or self.years_left==0:
                self.fin_eval()
                #print(f"Agent {self.unique_id} has no intention")
                # Handle agents who do not have intentions but must replace their heating system
                self._handle_no_intention_adoption()
            if "completed" in self.state:
                #print(f"Agent {self.unique_id} is in 'completed' state.")
                self.opinion_dynamics()
                

    def consider(self):

        """ 
        Agent starts thinking about renovation based on the following triggers:
        - Boiler replacement due to breakdown or nearing end of life
        - Price triggers in a social model 
        """
    
        # Check for boiler replacement or other triggers
        self.years_left = self.boiler_replacement()
        
        if (self.model.soc==False) and (self.years_left==0):
            self.state = "considering"
        elif (self.model.soc and (self.years_left <= self.model.n_consider or self.price_trigger())):
            self.state = "considering"
        else:
            self.state = 'none'                        
            
        # Track agents in the considering state
        if self.state == "considering":
            self.model.num_considering += 1
            self.model.agents_considering.append(self.unique_id)
        
    def fin_eval(self):        
        """ Agents who search for information (i.e. have low TC of info search) 
        evaluate various retrofitting options financially. 
        NPV is used to evaluate the financial attractiveness of the 
        retrofitting options. 
        """

        #number of options 
        if self.model.gas_boiler==False:
            self.num_options = int(self.model.n_options/2)
        # reduction in (useful) energy need for space heating via retrofit
        e_before = self.energy_before_ret()
        e_after = self.energy_after_ret()
        e_saving = e_before - e_after
        e_saving = e_saving.astype(int)
        #print('e_saving: ', e_saving)
        
        # write energy savings to our dataframe
        self.write2r_o('Heating demand reduction [kWh]', e_saving, self.retrofit_options)
        #print(self.retrofit_options['Heating demand reduction [kWh]'])
        npv_diff, payback_years = self.calc_npv_diff()
        
        # write NPV to our dataframe
        npv_list = npv_diff.values.tolist()

        if self.model.gas_boiler==False:
            npv_list.extend([np.nan] * 19)
            
        self.write2r_o('NPV [EUR]', npv_list, self.retrofit_options)
        self.retrofit_options['NPV [EUR]'] = npv_list
        
        if self.model.soc:
            self.add_metrics(e_saving, payback_years)
        
        self.state = "evaluating"
        
        # if self.unique_id == 13:
        #     print(f"Agent {self.unique_id} is {self.state}")


    def intention(self):
        # theory of planned behaviour (TPB)
        
        self.att = (self.opi+1)/2 # min-max normalization 
        # if self.unique_id == 13:
        #     print(f"Agent {self.unique_id} is {self.att}")
        # number of adoptions in the neighborhood
        num_adoptions = sum([1 for agent in self.model.schedule.agents
                          if agent.step_adopted == self.model.step_count])
        
        self.sn = num_adoptions/self.model.num_agents
        # pp_list = self.retrofit_options['Payback period [years]'].values.tolist()
        # if self.model.gas_boiler == False:
        #     pp_list = pp_list[:19]

        self.int = self.att*self.w_att + self.sn*self.w_sn + self.pbc*self.w_pbc

        # if self.int >= self.int_thr:
        #     self.state = "intending"
        # else:
        #     self.state = None 

        if self.int >= self.int_thr:
            self.state = "intending"
        else:
            self.state = 'none'
            
    def decision(self):
        """
        The decision method evaluates retrofit options and makes decisions
        based on whether the model is financial (soc=False) or social (soc=True).
        """
        if self.model.soc == False:
            self._financial_decision()
        else:
            self._social_decision()
            
        
        
        # After decision-making, update adoption status and finalize options
        if self.step_adopted == self.model.step_count:
            self._finalize_adoption()
            
            # print('step_adopted', self.step_adopted)
                
            
    def _financial_decision(self):
        """
        Logic for making a decision in the financial model (soc=False).
        """
        
        self.step_adopted = self.model.step_count
        self.state = "completed"
        # Increment adoption status here
        self.adoption_status += 1
        #print(f"Agent {self.unique_id} adopted {self.adoption_status} times")
        # if self.adoption_status == 2:
        #     print(f"Agent {self.unique_id} adopted {self.adoption_status} times")
            
        if self.adoption_status == 2:
            options_filt = self._filter_adopted_options()       
        else:
            options_filt = self.retrofit_options

        # Filter the options according to the gas ban, if applicable
        options_filt = self._apply_gas_ban(options_filt)
              
        self.adopted_option = options_filt.loc[options_filt['NPV [EUR]'].idxmax()]
        # Set the adopted option name for tracking
        self.adopted_option_name = self.adopted_option.name

        self.cost_inv = self.adopted_option['Full costs [EUR]']
        self.cost_env = self.adopted_option['Cost envelope [EUR]']
        self.cost_hp = self.adopted_option['Cost heating system [EUR]']
 
                    
    def _social_decision(self):
        """
        Decision-making logic for the social scenario (soc=True).
        Agents in the 'intending' state will finalize their adoption based on utility.
        """
        #SOCIAL DECISION-MAKING
        
        self.step_adopted = self.model.step_count
        self.state = "completed (intending)"
        # Increment adoption status here
        self.adoption_status += 1
        #print(f"Agent {self.unique_id} adopted {self.adoption_status} times")
        # if self.adoption_status == 2:
        #     print(f"Agent {self.unique_id} adopted {self.adoption_status} times")
        
        # For second-time adopters, track the first adoption details
        if self.adoption_status == 2:
            options_filt = self._filter_adopted_options()
        else:
            options_filt = self.retrofit_options
        #print(options_filt)
        # Filter the options according to the gas ban, if applicable
        options_filt = self._apply_gas_ban(options_filt)
        # Calculate utilities for the available options
        self._calculate_utilities(options_filt)
        # Select the best option based on the utility directly within options_filt
        self.adopted_option = self._select_best_option(options_filt)
        self.adopted_option_name = self.adopted_option.name
        
        
        
        
    
    def add_metrics(self, e_saving, payback_years):
        """
        Adds and normalizes payback period, complexity, energy-saving index, and 
        normalized complexity index to retrofit options, handling NaN values if gas boiler options are excluded.
        
        Parameters:
        -----------
        e_saving : list
            List of energy savings for each retrofit option.
            
        payback_years : DataFrame or array
            Payback period data for each retrofit option.
        """
        # Payback period (discounted)
        pp_list = payback_years.values.ravel().tolist()
        if self.model.gas_boiler == False and len(pp_list) < 38:
            pp_list.extend([np.nan] * 19)
        self.write2r_o('Payback period [years]', pp_list, self.retrofit_options)
    
        # Complexity list
        cx_list = self.complexity()['complexity'].to_list()
        if self.model.gas_boiler == False and len(cx_list) < 38:
            cx_list[19:] = [np.nan] * 19
        self.write2r_o('complexity', cx_list, self.retrofit_options)
    
        # Energy-saving index, normalized
        es_index_list = [(e - min(e_saving)) / (max(e_saving) - min(e_saving)) if max(e_saving) > min(e_saving) else 0 for e in e_saving]
        es_index_list = [round(x, 2) for x in es_index_list]
        if self.model.gas_boiler == False and len(es_index_list) < 38:
            es_index_list.extend([np.nan] * 19)
        self.write2r_o('es_index', es_index_list, self.retrofit_options)
    
        # Normalized complexity index
        min_inv, max_inv = round(min(self.calc_inv()), 0), round(max(self.calc_inv()), 0)
        cs_index_list = [(c - min_inv) / (max_inv - min_inv) if max_inv > min_inv else 0 for c in self.calc_inv()]
        #cs_index_list = [(c - min(self.calc_inv())) / (max(self.calc_inv()) - min(self.calc_inv())) if max(self.calc_inv()) > min(self.calc_inv()) else 0 for c in self.calc_inv()]
        cs_index_list = [round(x, 2) for x in cs_index_list]
        if self.model.gas_boiler == False and len(cs_index_list) < 38:
            cs_index_list.extend([np.nan] * 19)
        self.write2r_o('cs_index', cs_index_list, self.retrofit_options)
    
    
        
    def _calculate_utilities(self, options_filt):
        """
        Calculates normalized NPV and utility scores for retrofit options, with adjustments for non-gas boiler options.
        
        This function normalizes NPV values and calculates utility for each retrofit option
        based on NPV, complexity, and energy-saving indices, incorporating the agent's attitude.
        
        Returns:
        --------
        None
        """
        # Define the number of options based on the presence of gas boiler options
        self.num_options = len(options_filt)
    
        util_list = []
        npv_norm_list = []
    
        # Calculate normalized NPV and utility for each option
        for i in range(self.num_options):
            npv_norm = round(
                (options_filt['NPV [EUR]'].iloc[i] - options_filt['NPV [EUR]'].min()) / 
                (options_filt['NPV [EUR]'].max() - options_filt['NPV [EUR]'].min()), 2
            )
            npv_norm_list.append(npv_norm)
    
            # Calculate utility using NPV, complexity, and energy-saving index
            cx = 1 - options_filt['complexity'].iloc[i]
            es = options_filt['es_index'].iloc[i]
            utility_score = npv_norm * self.w_pp + cx * self.w_cx + es * self.w_es * self.att
            util_list.append(round(utility_score, 2))
    
        # Store results in the model's data frame
        self.write2r_o('util_list', util_list, options_filt)
        self.write2r_o('npv_norm', npv_norm_list, options_filt)
        
        
        return util_list
    
    def _select_best_option(self, options_filt):
        """
        Selects the best retrofit option based on utility scores already within `options_filt`.
        """
        # Ensure the 'util_list' column exists in options_filt
        if 'util_list' in options_filt.columns:
            # Find the row with the highest utility score within options_filt
            best_option = options_filt.loc[options_filt['util_list'].idxmax()]
            
            # Return the best option directly
            return best_option
        else:
            print("Error: 'util_list' column not found in options_filt.")
            return None

        
    def _finalize_adoption(self):
        """
        Finalizes the adoption process, updating agent attributes.
        Updates the state, records adoption details, and calculates related parameters.
        """
        
        # # Ensure the agent has selected an option before finalizing
        # if self.adopted_option is None:
        #     #print(f"Agent {self.unique_id} has no adopted option at step {self.model.step_count}.")
        #     return  # Exit early if no option is selected
    
        

        self.current_ins = self.adopted_option_name.split(' + ')[1]
        
        if self.adopted_option_name.startswith("gas") or self.adopted_option_name.startswith("GB"):
            self.current_hs = "gas boiler new"
        else:
            self.current_hs = "heat pump"
            
        self.heat_red = self.adopted_option['Heating demand reduction [kWh]']
        #print(self.adopted_option)
        self.co2_before = self.energy_before_ret()*self.model.co2_factor_ng
        
        # update years_left
        if self.adopted_option_name.startswith("gas") or self.adopted_option_name.startswith("GB"):
            self.years_left = self.cond_life2 # new gas condensing boiler lifetime
            self.co2_after = self.energy_after_ret()[self.adopted_option_name]*self.model.co2_factor_ng
            
        else:
            self.years_left = self.hp_life # new gas condensing boiler lifetime
            self.co2_after = self.energy_after_ret()[self.adopted_option_name]*self.model.co2_factor_el
            
        #print(self.retrofit_options['NPV [EUR]'])
        #print(f"Agent {self.unique_id} adopted {self.adopted_option_name}, resetting years_left to {self.years_left}")
        #print(self.retrofit_options[['util_list', 'complexity','es_index', 'cs_index']])

        self.co2_red = round(self.co2_before-self.co2_after,0)
        
        # If this is the agent's second adoption, record it for data collection
        if self.adoption_status == 2:
            self.second_option = {
                'adopted_option': self.adopted_option_name,
                'step_adopted': self.step_adopted,
                'agent_id': self.unique_id,
                'insulation': 'uninsulated' if self.current_ins == 'none' else 'insulated',
                'heating system': 'heat pump' if self.current_hs == "heat pump" else 'gas boiler'
            }
        elif self.adoption_status == 1:  # Only track first adoption during the first adoption
            self.first_option = {
                'adopted_option': self.adopted_option_name,
                'step_adopted': self.step_adopted,
                'agent_id': self.unique_id,
                'insulation': 'uninsulated' if self.current_ins == 'none' else 'insulated',
                'heating system': 'heat pump' if self.current_hs == "heat pump" else 'gas boiler'
            }
        #print(self.state)
        # Track adoption details
        if self.state == "completed (intending)":
            self.model.stepwise_adoptions["intention"].append(self.adopted_option_name)
            # Track agents with intention
            self.model.num_with_intention += 1
            self.model.agents_with_intention.append(self.unique_id)
            
        elif self.state == "completed (breakdown)":
            self.model.stepwise_adoptions["breakdown"].append(self.adopted_option_name)
            # Track agents who adopt because of boiler breakdown
            self.model.num_boiler_breakdown_adoptions += 1
            self.model.agents_boiler_breakdown.append(self.unique_id)
        
    
    def _handle_no_intention_adoption(self):
        
        """
        Handles agents who do not have the intention to retrofit but need to replace their boiler
        due to breakdown or other reasons (e.g., no intention but system failure).
        """
        #print('handle no intention is on')
               
        if (self.years_left == 0):
            if self.current_hs.startswith("heat") or (self.model.gas_boiler==False):
                self.adopted_option = self.retrofit_options.loc['heat pump (HP) + none']
            else:
                self.adopted_option = self.retrofit_options.loc['gas boiler (GB) + none']
            
            self.adopted_option_name = self.adopted_option.name
            self.step_adopted = self.model.step_count
            self.state = "completed (breakdown)"
            # Increment adoption status here
            self.adoption_status += 1     

            self._finalize_adoption()

              
          
    # Utility Methods for Decision-Making
           
    def _filter_adopted_options(self):
        """
        Filters out retrofit options that have already been adopted (e.g., walls, floor, windows, roof) and prevents
        switching from heat pump to gas boiler if the current heating system is a heat pump.
        Returns the filtered retrofit options as options_filt.
        """
        # Set up attributes based on adopted elements
        self.walls = 'walls' in self.adopted_option_name
        self.floor = 'walls' in self.adopted_option_name
        self.windows = 'double' in self.adopted_option_name
        self.roof = 'roof' in self.adopted_option_name
        
        if 'full' in self.adopted_option_name:
            self.walls = True
            self.floor = True
            self.windows = True
            self.roof = True
        element = [self.walls, self.floor, self.windows, self.roof]
        element_str  = ['walls', 'floor', 'windows', 'roof']
        #print(element)
        # Filter element_str to get the names where the condition is True
        to_drop = [name for is_true, name in zip(element, element_str) if is_true]
        #print(to_drop)
        # Create a mask for any index containing any of the substrings in to_drop
        mask = self.retrofit_options.index.to_series().apply(lambda idx: any(sub in idx for sub in to_drop))
        indices_to_drop = self.retrofit_options.index[mask]                        
        options_filt = self.retrofit_options.drop(index=indices_to_drop)
        
        # no switch from heat pump to gas boiler
        
        if self.current_hs.startswith("heat"):
            prefixes_to_drop = ["gas", "GB"]
            mask_prefixes = self.retrofit_options.index.to_series().apply(
                lambda idx: any(idx.startswith(prefix) for prefix in prefixes_to_drop))
            indices_to_drop_pref = self.retrofit_options.index[mask_prefixes]    
            options_filt = self.retrofit_options.drop(index=indices_to_drop_pref)
        
    
        return options_filt
        
        
    def _apply_gas_ban(self, options):
        """
        Filters out gas boiler options if a gas ban is in effect.
        
        Parameters:
        options (DataFrame): The DataFrame of retrofit options to filter.
        
        Returns:
        DataFrame: A filtered DataFrame with gas boiler options removed if the gas ban is in effect.
        """
        # Check if gas ban is active and if the current simulation step is beyond the ban year
        if self.model.gas_ban and self.model.gas_boiler==False: #self.model.step_count >= (self.model.ban_year - self.model.current_year + 1):
            # Remove options that start with 'gas' or 'GB' (gas boiler options)
            return options[~options.index.str.startswith('gas') & ~options.index.str.startswith('GB')]
        
        # Return the original options if no gas ban is in place
        return options
 

    
    def calc_npv_diff(self):
        # Calculate the NPV of a retrofitting option
        c_inv = self.calc_inv()  # Store the result to avoid recalculating
        c_inv = c_inv.dropna()
        CFi = self.oper_cost_after()
        #print("Cash Flows After Retrofitting (CFi):", CFi.head())
        exclude_columns = ['heat pump (HP) + none', 'gas boiler (GB) + none']
        columns_to_modify = [col for col in CFi.columns if col not in exclude_columns]
    
        if self.model.rest_v_ins:
            c_ins = self.retrofit_options['Cost envelope with insulation subsidy [EUR]'] if self.model.subsidy_ins else self.retrofit_options['Cost envelope [EUR]']
            residual_value = self.model.rest_value * c_ins
            # Apply the assignment only to the selected columns
            CFi.loc[CFi.index[0] + 19, columns_to_modify] = residual_value
    
        # Write investment costs as a DataFrame to avoid recalculating
        c_inv_df = pd.DataFrame([c_inv.values], columns=CFi.columns)        
        c_inv_df = -c_inv_df  # Negate the values in c_inv_df
    
        # Add inv cost to the dataframe with index 0
        CFi = pd.concat([c_inv_df, CFi], ignore_index=True)
    
        # Discounted cash flows
        discount_factors_i = 1 / (1 + self.model.discount_rate) ** np.arange(len(CFi))
        DCFi = CFi.mul(discount_factors_i, axis=0)
        
        NPVi = DCFi.cumsum()

        # NPV for gas boiler reference
        CFref = self.oper_cost_before()
        
        c_inv_gb = [-3000] * len(CFref.columns)
        c_inv_gb = dict(zip(CFref.columns, c_inv_gb))
        c_inv_gb = pd.DataFrame(c_inv_gb, index=[0])
        
        CFref = pd.concat([c_inv_gb, CFref], ignore_index=True)
        #print("Initial Cash Flow before retrofit (CFref):\n", CFref.head())
        
        discount_factors_ref = 1 / (1 + self.model.discount_rate) ** np.arange(len(CFref))
        #print("Discount factors for CFref:\n", discount_factors_ref)
        DCFref = CFref.mul(discount_factors_ref, axis=0)
        
        NPVref = DCFref.cumsum().round()
        #print("Cumulative NPV for reference (NPVref):\n", NPVref.head())
        
        npv_diff = NPVi - NPVref
        
        # Payback period calculation
        payback_years = DCFi.apply(lambda col: col[col >= 0].index[0] if any(col >= 0) else 21)
    
        return npv_diff.iloc[-1], payback_years

    
    def calc_inv(self):
        if self.model.gas_boiler == True:
            # retrofit cost for agent's dwelling 
            c_ins = self.retrofit_options['Cost envelope [EUR]'].copy()
            #print('c_ins before subsidy', c_ins)
            c_hs = self.retrofit_options['Cost heating system [EUR]'].copy()
            #print('c_hs before subsidy', c_hs)
        else: 
            # retrofit cost for agent's dwelling 
            c_ins = self.retrofit_options[:19]['Cost envelope [EUR]'].copy()
            #print('c_ins before subsidy', c_ins)
            c_hs = self.retrofit_options[:19]['Cost heating system [EUR]'].copy()
            #print('c_hs before subsidy', c_hs)
        # if self.unique_id==1:
        #     print('c_ins, c_hs: ', c_ins, c_hs)           
        
        #SENSITIVITY 
        if self.model.sens_ins == True:                
            c_ins = c_ins*self.model.ins_costs
            
        #COST INCREASE ON HEAT PUMPS ONLY
        if self.model.sens_hp == True:       
            if self.model.gas_boiler == True:
                c_hs = self.retrofit_options[:19]['Cost heating system [EUR]']*self.model.hp_costs
                c_hs = pd.concat([c_hs, self.retrofit_options[19:]['Cost heating system [EUR]']], ignore_index=False)
            else:
                c_hs = c_hs*self.model.hp_costs
        
        
        # SUBSIDY
        if self.model.subsidy_ins == True:                
            c_ins = self.retrofit_options['Cost envelope with insulation subsidy [EUR]']
        #print('c_ins after subsidy', c_ins)
        
        if self.model.subsidy_hp > 0:
            c_hs[:19] = \
            (1-self.model.subsidy_hp)*self.retrofit_options['Cost heating system [EUR]'][:19]
        #print('c_hs after subsidy', c_hs)
        c_inv = c_ins+c_hs
        
        # write (discounted) Energy costs to our dataframe
        self.write2r_o('Full costs [EUR]', c_inv.round(0), self.retrofit_options)
        return c_inv  

    def oper_cost_before(self):
        # operational costs before retrofit
        return -self.energy_before_ret()*(self.datafr_gas_p())            
    
    
    def oper_cost_after(self):
       # operational costs after retrofit
       c_n = -self.energy_after_ret().mul(self.datafr_el_p(), axis=0)
       # relevant gas prices
       a=self.datafr_gas_p()[self.datafr_gas_p().columns[19:]]
       # if self.model.step_count==1: 
       #     if self.building_arc == 'TC75-91':
       #         print('before', c_n)
       if self.model.gas_boiler==True:
           c_n[c_n.columns[19:]] = -self.energy_after_ret().iloc[19:]*a
           
       # if self.model.step_count==1: 
       #     if self.building_arc == 'TC75-91':
       #         print('after', c_n)
       # check
       # if self.building_arc == 'TC75-91':
       #     print(self.oper_cost_before())
       return c_n
   
    
    def energy_before_ret(self):
        e_0=self.building['Specific heating demand before [kWh/m2]'] \
            *self.building_area/self.model.gas_eff_old
        return e_0
    
    def energy_after_ret(self):
        # final energy demand after retrofit (electric)
        if self.model.gas_boiler==True:
            n = int(self.model.n_options/2)
            ##print('ro_hp', ro_hp)
            #print(self.calc_scop())
            e_n1 = self.retrofit_options[:n]['Specific heating demand after [kWh/m2]'] \
                * self.building_area/self.calc_scop()['SCOP']
            #e_n.drop(columns=18, inplace=True)
            #print('e_n', e_n1)
            e_n2 = self.retrofit_options[n:]['Specific heating demand after [kWh/m2]']\
                        *self.building_area/self.model.gas_eff_new
            e_n = pd.concat([e_n1, e_n2])
            #print(e_n)
        else:
            e_n = self.retrofit_options[:19]['Specific heating demand after [kWh/m2]'] \
                * self.building_area/self.calc_scop()['SCOP']
                
        # if self.model.step_count==1:
        #     print(e_n)
        return(e_n)
       
  
    def _handle_equal_top_values(self, options_filt, top_two_values):
        """
        Handles the case where the top two utility values are equal.
        """
        mask_ins = top_two_values.index.str.contains('ins.')
        mask_glaz = top_two_values.index.str.contains('glazing')
        adopted_name = top_two_values.index[0]
    
        if any(mask_ins):
            adopted_name = top_two_values.index[mask_ins][0]
        elif any(mask_glaz):
            adopted_name = top_two_values.index[mask_glaz][0]
    
        self.adopted_option = options_filt.loc[adopted_name] if self.adoption_status >= 2 else self.retrofit_options.loc[adopted_name]

    
    def opinion_dynamics(self):
        """Interact with another random homeowner in the neighborhood. This 
        changes the attitude of the current homeowner acording to the opinion 
        dynamics model by Deffuant et al (2002) called 'Relative Agreement' """
        # defining neighbours 
        
        others = [x for x in self.model.schedule.agents if x.unique_id != 
                  self.unique_id]
        #self.neighbour = [agent for agent in others if agent.building_id==self.building_id]
        # Create a list with other homeowners (excl. currently selected agent)
        list_others = [x for x in self.model.schedule.agents if x.unique_id != 
                       self.unique_id]
        # Select other random agents
        rng = np.random.default_rng(4)
        others = rng.choice(list_others, self.model.n_contact, replace=False) 
        #print('Agent {} with opinion {} interacts with Agent {} of opinion {}'
        #      .format(self.unique_id, self.opinion, other.unique_id, other.opinion))
        for i in others:
            other = i
            #print(self.opi, self.unc)
            overlap = min(self.opi+self.unc, other.opi+other.unc) - \
                max(self.opi-self.unc, other.opi-other.unc)
            # Check for overlap
            if overlap <= 0:
                continue  # Skip to the next agent if there is no overlap
            # Ensure uncertainty is non-zero, finite, and defined
            
            if self.unc != 0 and not np.isnan(self.unc) and not np.isinf(self.unc):
                rel_agr = overlap / self.unc - 1
             # Proceed if there's sufficient overlap for interaction
            if overlap >= self.unc:
                # current agent influences the other one
                opi0 = other.opi
                unc0 = other.unc
                other.opi = round(opi0 + self.model.mu*rel_agr*(self.opi - opi0), 2)
                other.unc = round(unc0 + self.model.mu*rel_agr*(self.unc - unc0), 2)
                
                # Normalize and set factors
                other.opi_norm = round((other.opi+1)/2, 2)
                other.factors = np.array([1-other.opi_norm, other.opi_norm])
                
                
        
            
            

            
    
    
            
  
        
