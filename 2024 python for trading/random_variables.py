# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 23:23:45 2024

@author: Nacho
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd


class simulator():
    
    #Constructor
    def __init__(self, coef, rv_type, size=10**6, decimals=5):
        self.coef = coef
        self.random_var_type = rv_type
        self.size = size
        self.decimals = decimals
        self.x = None
        self.str_title = None
        self.is_normal = None

    def generate_rv(self):
        self.str_title = self.random_var_type
        if self.random_var_type == 'Normal': 
            self.x = np.random.standard_normal(self.size)
        elif self.random_var_type == 'Student t':
            self.x = np.random.standard_t(df=self.coef, size=self.size)
            self.str_title += ' df=' + str(self.coef)      
        elif self.random_var_type == 'Uniform':
            self.x = np.random.uniform(size = self.size)
        elif self.random_var_type == 'Exponential':
            self.x = np.random.exponential(scale=self.coef, size=self.size)
            self.str_title= 'Exponential'
            self.str_title += ' scale=' + str(self.coef)  
        elif self.random_var_type == 'Chi-squared':
            self.x = np.random.chisquare(df=self.coef, size=self.size)
            self.str_title += ' df=' + str(self.coef)
        #return x, str_title
        
    def jb_Stat(self):
        mu = np.mean(self.x) 
        sigma = np.std(self.x)
        skewness = st.skew(self.x)
        kurt = st.kurtosis(self.x)
        jb_stat= (self.size/6)*(skewness**2 + 1/4*kurt**2)
        p_value = 1- st.chi2.cdf(jb_stat, df=2)
        self.is_normal = (p_value > 0.05)
        #return is_normal

    def plot (x,str_title) :
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