# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skew, kurtosis, chi2, tmean, tstd

#inputs
coef = 5 #df in student t, scale in esponential = tasa λ = 1/coef
size=10**6 
random_var_type = 'Student t'
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
#skew is positive if the queue in the right is graeter than the left one
#this is important to know the expected profitability
skewness = skew(x)
#kurt measures the probability of black swans / home runs or 
# the probability of falling into queues
kurt = kurtosis(x)

#Normality Test: Jarque-Bera Test
#If data comes from a nornmal distribution, the JB test 
#is distributed asymptotically as a chi square distribution
#With that test we are going to know how a distribution is related to a normal
jb_stat= (size/6)*(skewness**2 + 1/4*kurt**2)
#cdf gives the distribution function, the integral in x
p_value = 1- chi2.cdf(jb_stat, df=2)
#p-value 
is_normal = (p_value > 0.05)


str_title += '\n' + 'Mean = ' + str(np.round(mu ,decimals)) \
    +' | ' + 'Volatility = ' + str(np.round(sigma ,decimals)) \
    +'\n' + 'Skewness = ' + str(np.round(skewness ,decimals)) \
    +' | ' + 'Kurtosis = ' + str(np.round(kurt ,decimals)) \
    +'\n' + 'JB stat = ' + str(np.round(jb_stat ,decimals)) \
    +' | ' + 'P Value = ' + str(np.round(p_value ,decimals)) \
    +' \n ' + 'Is Normal = ' + str(is_normal)
#plot
 

#Cálculo de un supuesto donde falla el test



n = 0
is_normal = True
while is_normal and n<100:
    x = np.random.standard_normal(size)
    mu = tmean(x) 
    sigma = tstd(x)
    skewness = skew(x)
    kurt = kurtosis(x)
    jb_stat= (size/6)*(skewness**2 + 1/4*kurt**2)
    p_value = 1- chi2.cdf(jb_stat, df=2)
    is_normal = (p_value > 0.05)
    print('n='+ str(n) +' is normal= ' + str(is_normal))
    n += 1






















