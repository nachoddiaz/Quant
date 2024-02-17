# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 17:53:52 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st 
import importlib


import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)

rics = ['^MXX','^SPX','XLK','XLF','XLV','XLP','XLY','XLE','XLI']

#Sync timeseries returns
df = pd.DataFrame()
#Like a mapping in solidity
dic_timeseries = {}
timestamps=[]
# Get intersection of all timestamps
for ric in rics:
    t = market_data.load_timeseries(ric)
    dic_timeseries[ric] = t
    #To make all arrays the same lenght
    if len(timestamps) == 0:
        timestamps = list(t['date'].values)
    temp_timestamps = list(t['date'].values)
    timestamps = list(set(timestamps) & set(temp_timestamps))
    
    
#sync timeseries
for ric in dic_timeseries:
    t = dic_timeseries[ric]
    t = t[t['date'].isin(timestamps)]
    t = t.sort_values(by='date', ascending=True)
    t = t.dropna()
    t = t.reset_index(drop=True)
    dic_timeseries[ric] = t
    if df.shape[1] == 0:
        df['date'] = timestamps
    df[ric] = t['return']
    


# compute variance-covariance matrix


#To only have the returns column
mtx= df.drop(columns=['date'])
mtx_var_cov = np.cov(mtx, rowvar=False)


#compute correlation matrix
mtx_correl = np.corrcoef(mtx, rowvar=False)

