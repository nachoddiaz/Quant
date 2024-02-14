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
hedge_securites = ['AAPL', 'MSFT']
regularisation = 0.01

hedger = capm.hedger(position_security, position_delta_usd, hedge_securites, benchmark)
hedger.compute_betas()
hedger.compute_hedge_weights_optimize(regularisation)

#variables
hedge_weights = hedger.hedge_weights
hedge_delta_usd = hedger.hedge_delta_usd
hedge_beta_usd = hedger.hedge_beta_usd

portfolio_delta_usd = hedger.hedge_delta_usd + position_delta_usd
portfolio_beta_usd = hedger.hedge_beta_usd + hedger.position_beta_usd











