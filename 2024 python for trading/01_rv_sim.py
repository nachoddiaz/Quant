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
coef = 5 #df in student t, scale in esponential = tasa λ = 1/coef
size=10**6 
random_var_type = 'Normal'
#Destributions: Normal Student t Uniform Exponential Chi-squared
decimals = 6

#Constructor with only the aeguments that arent iniziatizated
sim = random_variables.simulator(coef, random_var_type)

sim.generate_rv()
x=sim.x
str_title=sim.str_title

sim.jb_Stat()
is_normal = sim.is_normal
  



str_title += '\n' + 'Mean = ' + str(np.round(mu ,decimals)) \
    +' | ' + 'Volatility = ' + str(np.round(sigma ,decimals)) \
    +'\n' + 'Skewness = ' + str(np.round(skewness ,decimals)) \
    +' | ' + 'Kurtosis = ' + str(np.round(kurt ,decimals)) \
    +'\n' + 'JB stat = ' + str(np.round(jb_stat ,decimals)) \
    +' | ' + 'P Value = ' + str(np.round(p_value ,decimals)) \
    +' \n ' + 'Is Normal = ' + str(is_normal)
#plot
#plot
plt.figure()
plt.hist(x, bins=100, density=True)
plt.title(str_title)
plt.show()


#####################################
#Loop of Jarque-Bera normality test #
#####################################
str_title=random_var_type
n = 0
is_normal = True
while is_normal and n<100:
    x = np.random.standard_normal(size)
    mu = st.tmean(x) 
    sigma = st.tstd(x)
    skewness = st.skew(x)
    kurt = st.kurtosis(x)
    jb_stat= (size/6)*(skewness**2 + 1/4*kurt**2)
    p_value = 1- st.chi2.cdf(jb_stat, df=2)
    is_normal = (p_value > 0.05)
    print('n='+ str(n) +' is normal= ' + str(is_normal))
    n += 1

str_title += '\n' + 'Mean = ' + str(np.round(mu ,decimals)) \
    +' | ' + 'Volatility = ' + str(np.round(sigma ,decimals)) \
    +'\n' + 'Skewness = ' + str(np.round(skewness ,decimals)) \
    +' | ' + 'Kurtosis = ' + str(np.round(kurt ,decimals)) \
    +'\n' + 'JB stat = ' + str(np.round(jb_stat ,decimals)) \
    +' | ' + 'P Value = ' + str(np.round(p_value ,decimals)) \
    +' \n ' + 'Is Normal = ' + str(is_normal)

#plot
plt.figure()
plt.hist(x, bins=100, density=True)
plt.title(str_title)
plt.show()


#fake homework

import scipy.stats as st
def test_jb(x):
    skewness = st.skew(x)
    kurt = st.kurtosis(x)
    jb_stat= (size/6)*(skewness**2 + 1/4*kurt**2)
    p_value = 1- st.chi2.cdf(jb_stat, df=2)
    is_normal = (p_value > 0.0375)
    return is_normal





















