# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:30:35 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd
import importlib
import os

import capm
importlib.reload(capm)


#inputs
benchmark = '^SPX'
position_security = 'BTC-USD'
position_delta_usd = 1 # in M USDC
hedge_securites = ['^SPX','^VIX']

hedger = capm.hedger(position_security, position_delta_usd, hedge_securites, benchmark)
hedger.compute_betas()
hedger.compute_optimal_hedge()
print('Our position of ' + str(position_delta_usd) + ' million USD of '  + str(position_security))
print('Has been hedged with ' + str(hedger.hedge_weights[0]) + ' million USD of '+ str(hedge_securites[0])+ ' and ' + str(hedger.hedge_weights[1]) + ' Million USD of ' + str(hedge_securites[1]) )