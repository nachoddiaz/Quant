# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:24:34 2024

@author: Nacho
"""

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

position_security = 'AAPL'    #y
factors = ['^SPX', 'IVW', 'IVE', 'SIZE', 'MTUM', 'QUAL', 'USMV' ]    #x

df = capm.dataframe_factors(position_security, factors)







