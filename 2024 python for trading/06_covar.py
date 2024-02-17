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


# compute variance-covariance matrix


#compute correlation matrix

