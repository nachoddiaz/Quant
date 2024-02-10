# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:04:50 2024

@author: Nacho
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st #import skew, kurtosis, chi2, tmean, tstd


def load_timeseries(ric):
    directory = 'C:\\Users\\Nacho\\.spyder-py3\\2024_python_for_trading\\2024-1-data\\'
    path = directory + ric + '.csv'
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True, format='mixed')
    t['close'] =raw_data['Close']
    t= t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return'] = t['close']/t['close_previous'] -1
    t = t.dropna()
    t= t.reset_index(drop=True)
    return t


class distribution_manager:
    
    #Constructor
    def __init__(self, ric, decimals = 5):
        self.ric = ric
        self.decimals = decimals
        self.timeSeries = None
        self.x = None
        self.str_title = None
        self.is_normal = None
        self.mean_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.var_95 = None
        self.skewness = None
        self.kurt = None
        self.jb_stat = None
        self.p_value = None
 

    def load_timeseries(self):
        self.timeSeries = load_timeseries(self.ric)
        self.x = self.timeSeries['return'].values
        self.size = len(self.x)
        self.str_title = self.ric + " | real data"
        
    def plot_timeseries (self):
        plt.figure()
        self.timeSeries.plot(kind='line', x='date', y='close', grid=True, color='blue', \
               label=self.ric, title='Timeseries of close prices for '+ self.ric)
        plt.show()
      
        
    def compute_stats(self, factor = 252):
        #self.
        self.mean_annual = st.tmean(self.x) * factor
        self.volatility_annual = st.tstd(self.x) * np.sqrt(factor)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual if self.volatility_annual > 0 else 0.0
        # if (self.volatility_annual > 0):
        #     self.sharpe_ratio = self.mean_annual / self.volatility_annual 
        # else: 
        #     0.0
        self.var_95 = np.percentile(self.x, 5)
        self.skewness = st.skew(self.x)
        self.kurt = st.kurtosis(self.x)
        self.jb_stat= (self.size/6)*(self.skewness**2 + 1/4*self.kurt**2)
        self.p_value = 1- st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05)
        
    #def normality_test(self):
        


    def plot_histogram (self) :
        self.str_title += '\n' + 'Mean_annual = ' + str(np.round(self.mean_annual ,self.decimals)) \
            +' | ' + 'Volatility_annual = ' + str(np.round(self.volatility_annual ,self.decimals)) \
            +'\n' + 'Skewness = ' + str(np.round(self.skewness ,self.decimals)) \
            +' | ' + 'Kurtosis = ' + str(np.round(self.kurt ,self.decimals)) \
            +' | ' + 'VaR_95 = ' + str(np.round(self.var_95 ,self.decimals)) \
            +'\n' + 'JB stat = ' + str(np.round(self.jb_stat ,self.decimals)) \
            +' | ' + 'P Value = ' + str(np.round(self.p_value ,self.decimals)) \
            +' \n ' + 'Is Normal = ' + str(self.is_normal) \
            +' | ' + 'Sharpe Ratio = ' + str(np.round(self.sharpe_ratio ,self.decimals))

        
        plt.figure()
        plt.hist(self.x, bins=100) #density=True
        plt.title(self.str_title)
        self.str_title = self.ric + " | real data"
        plt.show()
        
        
        
        
        
        
        
        