# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:32:32 2024

@author: akhatova
"""


import mesa
#import model
from model import RetrofitABM

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import itertools
from matplotlib.cm import get_cmap


pd.options.plotting.backend = "matplotlib"

params = {"soc": True, 
          "plot_distr": False,
          "randomness": True,
          "current_year": 2024, 
          "price_scenario": '19',
          "mean_pbc": 0.2, "mean_att": 0.1, "int_thr": 0.37,    
          "subsidy_ins": True, "subsidy_hp": 0.3, 
          "gas_ban": False,
          "sens_ins": False, "sens_hp": False,
          "n_contact": 3
          } 

current_year = params["current_year"]

num_iter = 25

results_batch = mesa.batch_run(
    RetrofitABM,
    parameters = params,
    iterations = num_iter,
    number_processes = 1,
    data_collection_period = 1,
    max_steps = 20,
    display_progress = True
)

res_df = pd.DataFrame(results_batch)

# Create an empty DataFrame with predefined indices and columns

df1 = pd.DataFrame()

for i in range(num_iter):
    res = res_df[res_df['RunId']==i]
    
    #calculate heat demand 
    res_heat = res[['Step','Heat demand reduction (share)']].groupby('Step').\
        mean().round(decimals=2)
    df1 = pd.concat([df1, res_heat], axis=1)

df1.insert(loc=0, column='Year', value=df1.index+current_year)
df1=df1.set_index('Year', drop=True)
df1 = df1.reset_index()
melted_df = df1.melt(id_vars=['Year'], var_name='Variable', value_name='Heat demand reduction (share)')

#df1 = df1.reset_index()
df_grouped = (
    melted_df[['Year', 'Heat demand reduction (share)']].groupby(['Year']).agg(['mean', 'std', 'count'])).reset_index()
df_grouped.columns = [f'{stat}' if metric == 'Heat demand reduction (share)' else metric for metric, stat in df_grouped.columns]

# Calculate a confidence interval as well.
df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
df_grouped.to_excel('uncertainty_2019_soc.xlsx', index=True)

num_colors = 10
cmap = get_cmap('tab10')
colors = cmap(np.linspace(0, 1, num_colors))

fig, axs = plt.subplots(1, 1, figsize=(6, 5), dpi=500)

x = df_grouped['Year']
y = df_grouped['mean']
ci_lower = df_grouped['ci_lower']  # You need to have these columns in your DataFrame
ci_upper = df_grouped['ci_upper']

# First plot
axs.plot(x, y, label='Mean Heat Demand Reduction')
axs.fill_between(x, ci_lower, ci_upper, color='skyblue', alpha=0.2, label='Confidence Interval')
axs.set_ylabel('Heat demand reduction [%]')
axs.set_ylim(ymin=0)
axs.grid(True, linestyle='--', alpha=0.7)
axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axs.legend()

# Adjusting the format of x-axis labels
fig.autofmt_xdate(rotation=0)

plt.show()



