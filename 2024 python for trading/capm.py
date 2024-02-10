# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:30:01 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st 
import scipy.optimize  as op 
import importlib
import os

import market_data
importlib.reload(market_data)

def compute_betas (benchmark, security):
    m = model(benchmark, security)
    m.sync_timeseries()
    m.compute_linear_regression()
    return m.beta

     


class model:
    
    def __init__(self, benchmark, security, decimals = 6):
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
        self.timeseries_x = market_data.load_timeseries(self.benchmark)
        self.timeseries_y = market_data.load_timeseries(self.security) 
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
        
        
class hedger:
    
    def __init__ (self, position_security, position_delta_usd, hedge_securities, benchmark):
        self.position_security= position_security
        self.position_delta_usd = position_delta_usd
        self.hedge_securities = hedge_securities
        self.benchmark = benchmark
        self.position_beta = None
        self.position_beta_usd = None
        self.hedge_betas = []
        self.hedge_weights = []
        self.hedge_delta = None
        self.hedge_beta_usd = None
        
    def compute_betas(self):
        self.position_beta = compute_betas(self.benchmark, self.position_security)
        self.position_beta_usd = self.position_beta * self.position_delta_usd
        for security in self.hedge_securities:
            beta = compute_betas(self.benchmark, security)
            self.hedge_betas.append(beta)
    
    def compute_optimal_hedge(self):
        dimensions = len(self.hedge_securities)
        if dimensions != 2:
            print('Cannot compute the exact solution cause dimensions = ' + str(dimensions))
            return
        deltas = np.ones([dimensions])
        target = -np.array([self.position_delta_usd, self.position_beta_usd]) 
        #First we put the 2 arrays as columns and then into rows
        mtx = np.transpose(np.column_stack((deltas, self.hedge_betas)))          
        self.hedge_weights = np.linalg.solve(mtx,target)
        #Those are tests
        self.hedge_delta = np.sum(self.hedge_weights)
        self.hedge_beta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item()

             
        
        

        
        