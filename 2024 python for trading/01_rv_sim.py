# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skew, kurtosis, chi2

#inputs
coef = 2000 #df in student t, scale in esponential = tasa λ = 1/coef
size=10**6 
random_var_type = 'Chi-squared'
#Destributions: Normal Student t Uniform Exponential Chi-squared
decimals = 6

#code
str_title = random_var_type
if random_var_type == 'Normal': 
    x = np.random.standard_normal(size)
elif random_var_type == 'Student t':
    x = np.random.standard_t(df=coef, size=size)
    str_title += ' df=' + str(coef)      
elif random_var_type == 'Uniform':
    x = np.random.uniform(size = size)
elif random_var_type == 'Exponential':
    x = np.random.exponential(scale=coef, size=size)
    str_title= 'Exponential'
    str_title += ' scale=' + str(coef)  
elif random_var_type == 'Chi-squared':
    x = np.random.chisquare(df=coef, size=size)
    str_title += ' df=' + str(coef)  
    
mu = np.mean(x) 
sigma = np.std(x)
skew = skew(x)
kurt = kurtosis(x)

str_title += '\n' + 'Mean = ' + str(np.round(mu ,decimals)) \
    +'\n' + 'Volatility = ' + str(np.round(sigma ,decimals)) \
    +'\n' + 'Skewness = ' + str(np.round(skew ,decimals)) \
    +'\n' + 'Kurtosis = ' + str(np.round(kurt ,decimals))

#plot
plt.figure()
plt.hist(x, bins=100, density=True)
plt.title(str_title)
plt.show()
