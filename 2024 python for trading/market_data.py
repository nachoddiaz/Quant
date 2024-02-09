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
        
        
class capm:
    
    def __init__(self, benchmark, security, decimals = 4):
        self.benchmark = benchmark
        self.security = security
        self.decimals = decimals
        self.timestamps_x = None
        self.timestamps_y = None
        self.timestamps = None
        self.timeseries_x = None
        self.timeseries_y = None
        self.timeseries = None
        self.beta = None
        self.alpha = None
        self.r_value = None
        self.r_squared = None
        self.p_value = None
        self.corr = None
        self.null_hyp = None
        self.line = None
        self.x = None
        self.y = None
        

    def sync_timeseries(self):
        self.timeseries_x = load_timeseries(self.benchmark)
        self.timeseries_y = load_timeseries(self.security) 
        self.timestamps_x = list(self.timeseries_x['date'].values)
        self.timestamps_y = list(self.timeseries_y['date'].values)

        #To synchronize the two lists, we calculate their intersection
        self.timestamps = list(set(self.timestamps_x) & set(self.timestamps_y))

        #This step filters timeseries_x, selecting only rows whose 'date'
        #column contains values ​​that are also present in timestamps.
        self.timeseries_x = self.timeseries_x[self.timeseries_x['date'].isin(self.timestamps)]
        self.timeseries_x = self.timeseries_x.dropna()
        self.timeseries_x= self.timeseries_x.reset_index(drop=True)

        self.timeseries_y = self.timeseries_y[self.timeseries_y['date'].isin(self.timestamps)]
        self.timeseries_y = self.timeseries_y.dropna()
        self.timeseries_y= self.timeseries_y.reset_index(drop=True)

        self.timeseries = pd.DataFrame()
        self.timeseries['date'] = self.timeseries_x['date']
        self.timeseries['close_x'] = self.timeseries_x['close']
        self.timeseries['close_y'] = self.timeseries_y['close']
        self.timeseries['return_x'] = self.timeseries_x['return']
        self.timeseries['return_y'] = self.timeseries_y['return']
        
    
    def plot_timeseries(self):
        plt.figure(figsize=(12,5))
        plt.title('time series of close prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        ax = plt.gca()
        ax1 = self.timeseries.plot(kind='line', x='date', y='close_x', grid=True, ax=ax, color='blue', \
               label=self.benchmark, title='Timeseries of close prices for '+ self.benchmark)
        ax2 = self.timeseries.plot(kind='line', x='date', y='close_y', grid=True, ax=ax, color='red', \
               label=self.security, secondary_y = True, title='Timeseries of close prices for '+ self.security)   
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
    
    def compute_linear_regression(self):
        self.x=self.timeseries['return_x'].values
        self.y=self.timeseries['return_y'].values
        self.beta, self.alpha, self.r_value, self.p_value, std_err = st.linregress(self.x, self.y) #, alternative='two-sided'
        self.corr = np.round(self.r_value,self.decimals)
        self.r_squared = np.round(self.r_value**2,self.decimals)
        self.alpha = np.round(self.alpha, self.decimals)
        self.beta = np.round(self.beta, self.decimals)
        self.p_value = np.round(self.p_value, self.decimals)
        self.null_hyp = self.p_value > 0.05
        self.line = self.alpha + self.beta*self.x
        
    def plot_linear_regression(self):
        self.x=self.timeseries['return_x'].values
        self.y=self.timeseries['return_y'].values
        str_self = 'Linear regression | security ' + self.security\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis: ' + str(self.null_hyp) + '\n'\
            + 'correl (r-value) ' + str(self.corr)\
            + ' | r-squared ' + str(self.r_squared)
        str_title = 'Scatterplot of returns' + '\n' + str(str_self)
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.x,self.y)
        plt.plot(self.x, self.line, color='red')
        plt.ylabel(self.security)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
    
        
        
        
        
        
        
        
        
        