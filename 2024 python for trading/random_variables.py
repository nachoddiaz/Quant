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
        self.mean = None
        self.volatility = None
        self.skewness = None
        self.kurt = None
        self.jb_stat = None
        self.p_value = None

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
        self.mean = st.tmean(self.x) 
        self.volatility = st.tstd(self.x)
        self.skewness = st.skew(self.x)
        self.kurt = st.kurtosis(self.x)
        self.jb_stat= (self.size/6)*(self.skewness**2 + 1/4*self.kurt**2)
        self.p_value = 1- st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05)
        #return is_normal

    def plot (self) :
        self.str_title += '\n' + 'Mean = ' + str(np.round(self.mean ,self.decimals)) \
            +' | ' + 'Volatility = ' + str(np.round(self.volatility ,self.decimals)) \
            +'\n' + 'Skewness = ' + str(np.round(self.skewness ,self.decimals)) \
            +' | ' + 'Kurtosis = ' + str(np.round(self.kurt ,self.decimals)) \
            +'\n' + 'JB stat = ' + str(np.round(self.jb_stat ,self.decimals)) \
            +' | ' + 'P Value = ' + str(np.round(self.p_value ,self.decimals)) \
            +' \n ' + 'Is Normal = ' + str(self.is_normal)
        
        plt.figure()
        plt.hist(self.x, bins=100) #density=True
        plt.title(self.str_title)
        plt.show()
        str_title = ''