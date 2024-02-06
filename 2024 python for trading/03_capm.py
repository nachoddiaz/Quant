# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:36:17 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd
import importlib
import os

import market_data
importlib.reload(market_data)

benchmark = '^MXX'      #x
security = 'EWW'    #y

#def sync_timeseries (security, benchmark):

timeseries_x = market_data.load_timeseries(benchmark)
timeseries_y = market_data.load_timeseries(security) 
timestamps_x = list(timeseries_x['date'].values)
timestamps_y = list(timeseries_y['date'].values)

#To synchronize the two lists, we calculate their intersection
timestamps = list(set(timestamps_x) & set(timestamps_y))

#This step filters timeseries_x, selecting only rows whose 'date'
#column contains values ​​that are also present in timestamps.
timeseries_x = timeseries_x[timeseries_x['date'].isin(timestamps)]
timeseries_x = timeseries_x.dropna()
timeseries_x= timeseries_x.reset_index(drop=True)

timeseries_y = timeseries_y[timeseries_y['date'].isin(timestamps)]
timeseries_y = timeseries_y.dropna()
timeseries_y= timeseries_y.reset_index(drop=True)

timeseries = pd.DataFrame()
timeseries['date'] = timeseries_x['date']
timeseries['close_x'] = timeseries_x['close']
timeseries['close_y'] = timeseries_y['close']
timeseries['return_x'] = timeseries_x['return']
timeseries['return_y'] = timeseries_y['return']

#plot timeseries
plt.figure(figsize=(12,5))
plt.title('time series of close prices')
plt.xlabel('Time')
plt.ylabel('Price')
ax = plt.gca()
ax1 = timeseries.plot(kind='line', x='date', y='close_x', grid=True, ax=ax, color='blue', \
       label=benchmark, title='Timeseries of close prices for '+ benchmark)
ax2 = timeseries.plot(kind='line', x='date', y='close_y', grid=True, ax=ax, color='red', \
       label=security, secondary_y = True, title='Timeseries of close prices for '+ security)   
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.show()

#timeseries_x
#timeseries_y

