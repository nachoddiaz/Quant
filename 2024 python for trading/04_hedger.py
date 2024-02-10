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
position_ric = 'NVDA'
position_delta_usd = 10 # in M USDC
hedge_rics = ['AAPL','MSFT']

hedger = capm.hedger(position_ric, position_delta_usd, hedge_rics, benchmark)
hedger.compute_betas()