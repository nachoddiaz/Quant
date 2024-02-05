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

benchmark = '^SPX'      #x
security = 'BTC-USD'    #y

timeseries_x = market_data.load_timeseries(security)
timeseries_y = market_data.load_timeseries(benchmark) 
timestamp_x = timeseries_x['date'].values
timestamp_y = timeseries_y['date'].values