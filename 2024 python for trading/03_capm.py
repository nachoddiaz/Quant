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

import capm
importlib.reload(capm)

benchmark = '^SPX'      #x
security = 'AAPL'    #y

model = capm.model(benchmark, security)
model.sync_timeseries()
model.plot_timeseries()
model.compute_linear_regression()
model.plot_linear_regression()







