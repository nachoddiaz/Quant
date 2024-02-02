# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:54:20 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd
import importlib

import random_variables
importlib.reload(random_variables)
inputs = random_variables.sim_inputs()

#inputs
ric = 'EURUSD=X'

directory = 'C:\\Users\\Nacho\\.spyder-py3\\2024_python_for_trading\\2024-1-data\\'
path = directory + ric + '.csv'
raw_data = pd.read_csv(path)
t = pd.DataFrame()
t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True, format='%Y-%m-%d')
t['close'] =raw_data['Close']
t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1)
t['return close'] = t['close']/t['close_previous'] -1
t = t.dropna()
t= t.reset_index(drop=True)


inputs.random_var_type = ric + ' | real time'
inputs.decimals = 5

#Constructor with only the aeguments that arent iniziatizated
sim = random_variables.simulator(inputs)

#Generation of de random vector
sim.x = t['return close'].values
#Generation of the Jarque-Bera Stat
sim.inputs.size = len(sim.x)
sim.str_title = sim.inputs.random_var_type
sim.compute_stats()
#Ploting results
sim.plot()
