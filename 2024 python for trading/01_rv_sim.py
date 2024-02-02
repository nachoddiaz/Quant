# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd
import importlib

import random_variables
importlib.reload(random_variables)

#inputs
inputs = random_variables.sim_inputs()
inputs.df = 5 #df in student t and chi-squared
inputs.scale = 5 #scale in esponential = tasa λ = 1/coef
inputs.mean = 5 #mean in Normal
inputs.std = 10 #standerd deviaion in Normal
inputs.size=10**6 
inputs.random_var_type = 'Normal'
#Destributions: Standard_normal Normal Student-t Uniform Exponential Chi-squared
inputs.decimals = 6

#Constructor with only the aeguments that arent iniziatizated
sim = random_variables.simulator(inputs)

#Generation of de random vector
sim.generate_rv()
#Generation of the Jarque-Bera Stat
sim.compute_stats()
#Ploting results
sim.plot()


# #####################################
# #Loop of Jarque-Bera normality test #
# #####################################
# str_title=random_var_type
# n = 0
# is_normal = True
# while is_normal and n<100:
#     x = np.random.standard_normal(size)
#     mu = st.tmean(x) 
#     sigma = st.tstd(x)
#     skewness = st.skew(x)
#     kurt = st.kurtosis(x)
#     jb_stat= (size/6)*(skewness**2 + 1/4*kurt**2)
#     p_value = 1- st.chi2.cdf(jb_stat, df=2)
#     is_normal = (p_value > 0.05)
#     print('n='+ str(n) +' is normal= ' + str(is_normal))
#     n += 1

# str_title += '\n' + 'Mean = ' + str(np.round(mu ,decimals)) \
#     +' | ' + 'Volatility = ' + str(np.round(sigma ,decimals)) \
#     +'\n' + 'Skewness = ' + str(np.round(skewness ,decimals)) \
#     +' | ' + 'Kurtosis = ' + str(np.round(kurt ,decimals)) \
#     +'\n' + 'JB stat = ' + str(np.round(jb_stat ,decimals)) \
#     +' | ' + 'P Value = ' + str(np.round(p_value ,decimals)) \
#     +' \n ' + 'Is Normal = ' + str(is_normal)

# #plot
# plt.figure()
# plt.hist(x, bins=100, density=True)
# plt.title(str_title)
# plt.show()


# #fake homework

# import scipy.stats as st
# def test_jb(x):
#     skewness = st.skew(x)
#     kurt = st.kurtosis(x)
#     jb_stat= (size/6)*(skewness**2 + 1/4*kurt**2)
#     p_value = 1- st.chi2.cdf(jb_stat, df=2)
#     is_normal = (p_value > 0.0375)
#     return is_normal





















