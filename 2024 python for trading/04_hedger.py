# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:30:35 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats  as st
import scipy.optimize as op
import importlib
import os

import capm
importlib.reload(capm)

#inputs
benchmark = '^SPX'
position_security = 'NVDA'
position_delta_usd = 10 # in M USDC
hedge_universe = ['SPY','AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'NFLX', 'NVDA', 'XLF', 'XLK']
regularisation = 0.01

df = capm.dataframe_correl_beta(benchmark, position_security, hedge_universe)

hedge_securities = [hedge_universe[0], hedge_universe[1]]
hedger = capm.hedger(position_security, position_delta_usd, hedge_securities, benchmark)
hedger.compute_betas()
hedger.compute_hedge_weights_optimize(regularisation)

#variables
hedge_weights = hedger.hedge_weights
hedge_delta_usd = hedger.hedge_delta_usd
hedge_beta_usd = hedger.hedge_beta_usd

portfolio_delta_usd = hedger.hedge_delta_usd + position_delta_usd
portfolio_beta_usd = hedger.hedge_beta_usd + hedger.position_beta_usd











