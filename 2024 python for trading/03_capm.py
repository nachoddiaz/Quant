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


benchmark = 'USDT-USD'      #x
security = 'DAI-USD'    #y

import capm
importlib.reload(capm)

capm = capm.capm(benchmark, security)
capm.sync_timeseries()
capm.plot_timeseries()
capm.compute_linear_regression()
capm.plot_linear_regression()







