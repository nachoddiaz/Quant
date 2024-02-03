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
import os

import random_variables
import market_data
importlib.reload(market_data)

#inputs
ric = '^MXX'

#Compute
dist = market_data.distribution_manager(ric)
dist.load_timeseries()
dist.plot_timeseries()
dist.compute_stats()
dist.plot_histogram()


directory = 'C:\\Users\\Nacho\\.spyder-py3\\2024_python_for_trading\\2024-1-data\\'

rics = []
is_normals = []
for file_name in os.listdir(directory):
    #print('file_name = ' + file_name)
    #returns the first element after split
    ric = file_name.split('.')[0]
    #get data_frame
    path = directory + ric + '.csv'
    dist = market_data.distribution_manager(ric)
    dist.load_timeseries()
    #dist.plot_timeseries()
    dist.compute_stats() 
    #dist.plot_histogram()
    #generate lists
    rics.append(ric)
    is_normals.append(dist.is_normal)
    
df = pd.DataFrame()
df['ric'] = rics
df['is_normal'] = is_normals
df.sort_values(by='is_normal', ascending = False)









