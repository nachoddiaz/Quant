# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 23:23:45 2024

@author: Nacho
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd


class sim_inputs:
    def __init__(self):
        self.df = None
        self.scale = None
        self.mean = None
        self.std = None
        self.size = None
        self.random_var_type = None
        self.decimals = None

class simulator:
    
    #Constructor
    def __init__(self, inputs):
        self.inputs = inputs
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
        self.str_title = self.inputs.random_var_type
        if self.inputs.random_var_type == 'Standard_normal': 
            self.x = np.random.standard_normal(self.inputs.size)
        if self.inputs.random_var_type == 'Normal': 
            self.x = np.random.normal(self.inputs.mean, self.inputs.std, self.inputs.size)
        elif self.inputs.random_var_type == 'Student-t':
            self.x = np.random.standard_t(self.inputs.df, self.inputs.size)
            self.str_title += ' df=' + str(self.inputs.df)      
        elif self.inputs.random_var_type == 'Uniform':
            self.x = np.random.uniform(size = self.inputs.size)
        elif self.inputs.random_var_type == 'Exponential':
            self.x = np.random.exponential(self.inputs.scale, self.inputs.size)
            self.str_title= 'Exponential'
            self.str_title += ' scale=' + str(self.inputs.scale)  
        elif self.inputs.random_var_type == 'Chi-squared':
            self.x = np.random.chisquare(df=self.inputs.df, size=self.inputs.size)
            self.str_title += ' df=' + str(self.inputs.df)
      
        
    def compute_stats(self):
        self.mean = st.tmean(self.x) 
        self.volatility = st.tstd(self.x)
        self.skewness = st.skew(self.x)
        self.kurt = st.kurtosis(self.x)
        self.sharpe = (self.mean-(0.02**(1/365))-1)/self.volatility
        self.jb_stat= (self.inputs.size/6)*(self.skewness**2 + 1/4*self.kurt**2)
        self.p_value = 1- st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05)


    def plot (self) :
        self.str_title += '\n' + 'Mean = ' + str(np.round(self.mean ,self.inputs.decimals)) \
            +' | ' + 'Volatility = ' + str(np.round(self.volatility ,self.inputs.decimals)) \
            +'\n' + 'Skewness = ' + str(np.round(self.skewness ,self.inputs.decimals)) \
            +' | ' + 'Kurtosis = ' + str(np.round(self.kurt ,self.inputs.decimals)) \
            +'\n' + 'JB stat = ' + str(np.round(self.jb_stat ,self.inputs.decimals)) \
            +' | ' + 'P Value = ' + str(np.round(self.p_value ,self.inputs.decimals)) \
            +' \n ' + 'Is Normal = ' + str(self.is_normal) \
            +' | ' + 'Sharpe Ratio = ' + str(np.round(self.sharpe ,self.inputs.decimals)) \

        
        plt.figure()
        plt.hist(self.x, bins=100) #density=True
        plt.title(self.str_title)
        plt.show()
        str_title = ''