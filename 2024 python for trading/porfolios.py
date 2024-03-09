# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:40:57 2024

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
import capm
importlib.reload(capm)

def portfolio_var(x, mtx_var_cov):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_cov,x))
    return variance


class manager: 
    
    def __init__(self,rics,notional, number_rics, decimals = 6,factor = 252):
        self.number_rics = number_rics
        self.rics = rics
        self.factor = factor
        self.decimals = decimals
        self.notional = notional
        self.mtx_var_cov = None
        self.mtx_correl = None
        self.portfolio_type = None
        self.returns = None
        self.volatilities = None
        self.df_timeseries = None
        
    def compute_covariance(self):
        df = capm.model.sync_returns(self.rics)
        mtx= df.drop(columns=['date'])
        self.mtx_var_cov = np.cov(mtx, rowvar=False) *252
        self.mtx_correl = np.corrcoef(mtx, rowvar=False)
        returns = []
        volatilities = []
        for ric in self.rics:
            r = np.round(np.mean(df[ric]) * self.factor, self.decimals)
            v = np.round(np.std(df[ric]) * np.sqrt(self.factor), self.decimals)
            returns.append(r)
            volatilities.append(v)
        self.returns = np.array(returns)
        self.volatilities = np.array(volatilities)
        df_m = pd.DataFrame()
        df_m['rics'] = self.rics
        df_m['returns'] = self.returns 
        df_m['volatilities'] = self.volatilities
        self.dataframe_metrics = df_m
        self.df_timeseries = df
        
    
    def compute_portfolio(self, portfolio_type=None, target_return=None): 
        #inputs
        x0 = [1/ len(self.rics)] * len(self.rics)
        L2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] #unitary in norm L2
        L1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] #unitary in norm L1
        #diapo 7 https://fractalvelvet.files.wordpress.com/2023/10/optimisation_problems-3.pdf
        markowitz = [{'type': 'eq', 'fun': lambda x: self.returns.dot(x) - target_return}]
        
        non_negative = [(0, None) for i in range(len(self.rics))]
        
        #compute porfolio depending on the type
        if portfolio_type == 'min_var_L1':
            optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
                                         args=(self.mtx_var_cov),\
                                         constraints=L1_norm)
            weights = np.array(optimal_result.x)
        elif portfolio_type == 'min_var_L2':
            optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
                                         args=(self.mtx_var_cov),\
                                         constraints=L2_norm)
            weights = np.array(optimal_result.x)
            
        elif portfolio_type == 'long_only':
            optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
                                         args=(self.mtx_var_cov),\
                                         constraints=(L1_norm), \
                                         bounds = non_negative)
            weights = np.array(optimal_result.x)
            
        elif portfolio_type == 'markowitz':
            epsylon = 10**-4
            if target_return == None:
                target_return = np.mean(self.returns)
            elif target_return < np.min(self.returns):
                target_return = np.min(self.returns) + epsylon
            elif target_return > np.max(self.returns):
                target_return = np.max(self.returns) - epsylon
            
            optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
                                          args=(self.mtx_var_cov),\
                                    #to concatenate l1 and mark lists
                                          constraints=(L1_norm + markowitz), \
                                          bounds = non_negative)    
            weights = np.array(optimal_result.x)

        else :
            portfolio_type = 'equi_weight'
            weights = np.array(x0)
        
        
        optimal_portfolio = output(self.rics, self.notional)
        optimal_portfolio.type = portfolio_type
        optimal_portfolio.weights = weights / sum(abs(weights))
        optimal_portfolio.allocation = self.notional * optimal_portfolio.weights
        optimal_portfolio.target_return = target_return
        optimal_portfolio.return_annual = np.round(self.returns.dot(weights), self.decimals)
        optimal_portfolio.volatility_annual = np.dot(weights.T, \
                                                     np.dot(self.mtx_var_cov, weights))
        optimal_portfolio.sharpe_ratio = \
            optimal_portfolio.return_annual /  optimal_portfolio.volatility_annual \
            if optimal_portfolio.volatility_annual > 0 else 0.0

        # extend dataframe of metrics with optimal weights and allocations
        # It is good practice to create a new DataFrame if we want to introduce
        # new columns or rows
        df_al = self.dataframe_metrics.copy()
        df_al['weights'] = optimal_portfolio.weights
        df_al['allocation'] = optimal_portfolio.allocation
        optimal_portfolio.dataframe_allocation = df_al
        
        # extend dataframe of timeseries with porfolio returns
        df_ts = self.df_timeseries.copy()
        rics = list(df_al['rics'])
        port_rets = df_ts[rics[0]].values * 0.0
        for ric in rics:
            df = df_al.loc[df_al['rics'] == ric]
            w = df['weights'].item()
            port_rets += self.df_timeseries[ric].values * w
        df_ts['portfolio'] = port_rets
        optimal_portfolio.df_timeseries = df_ts
        
        optimal_portfolio.compute_stats()
            
        return optimal_portfolio        
        
        
class output: 
    
    def __init__(self,rics,notional, decimals = 6,factor = 252):
        self.rics = rics
        self.notional = notional
        self.decimals = decimals
        self.factor = factor
        self.type = None
        self.weights = None
        self.allocation = None
        self.target_return = None
        self.return_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.var_95 = None
        self.skewness = None
        self.kurt = None
        self.jb_stat= None
        self.p_value = None
        self.is_normal = None
        self.dataframe_allocation = None
        self.df_timeseries = None
        self.str_title = None
        self.x = None
        
        pass
    
    def compute_stats(self, factor = 252):
        self.x = self.df_timeseries['portfolio'].values
        self.var_95 = np.percentile(self.x, 5)
        self.skewness = st.skew(self.x)
        self.kurt = st.kurtosis(self.x)
        self.jb_stat= (self.factor/6)*(self.skewness**2 + 1/4*self.kurt**2)
        self.p_value = 1- st.chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05)


    def plot_histogram(self):
        self.str_title = 'Portfolio = ' + self.type
        if self.target_return != None:
            self.str_title += ' | target return = ' + str (np.round(self.target_return ,self.decimals))
        self.str_title += '\n' + 'Mean_annual = ' + str(np.round(self.return_annual ,self.decimals)) \
             +' | ' + 'Volatility_annual = ' + str(np.round(self.volatility_annual ,self.decimals)) \
             +' | ' + 'Sharpe Ratio = ' + str(np.round(self.sharpe_ratio ,self.decimals))\
             +'\n' + 'Skewness = ' + str(np.round(self.skewness ,self.decimals)) \
             +' | ' + 'Kurtosis = ' + str(np.round(self.kurt ,self.decimals)) \
             +' | ' + 'VaR_95 = ' + str(np.round(self.var_95 ,self.decimals)) \
             +'\n' + 'JB stat = ' + str(np.round(self.jb_stat ,self.decimals)) \
             +' | ' + 'P Value = ' + str(np.round(self.p_value ,self.decimals)) \
             +' \n ' + 'Is Normal = ' + str(self.is_normal)
        plt.figure()
        plt.hist(self.x, bins=100) #density=True
        plt.title(self.str_title)
        plt.show()
        
        
def compute_eff_front(rics, notional, number_rics, target_return, include_min_var):
    #special-portfolios
    label1 = 'min_var_L1'
    label2 = 'min_var_L2'
    label3 = 'equi_weight'
    label4 = 'long_only'
    label5 = 'markowitz-none'
    label6 = 'markowitz-target'
    
    #cov_mtx
    prt_mng= manager(rics, notional, number_rics)
    prt_mng.compute_covariance()
    
    #compute vectors of returns and volatilities por Markowitz portfolios
    min_returns = np.min(prt_mng.returns)
    max_returns = np.max(prt_mng.returns)
    returns = min_returns + np.linspace(0.05, 0.95,100) * (max_returns- min_returns)
    volatilities = np.zeros([len(returns), 1])
    i = 0
    for ret in returns:
        port_markowitz = prt_mng.compute_portfolio('markowitz', ret)
        volatilities[i] = port_markowitz.volatility_annual
        i += 1
        
    #Computing special portfolios
    port1 = prt_mng.compute_portfolio(label1)
    port2 = prt_mng.compute_portfolio(label2)
    port3 = prt_mng.compute_portfolio(label3)
    port4 = prt_mng.compute_portfolio(label4)
    port5 = prt_mng.compute_portfolio('markowitz')
    port6 = prt_mng.compute_portfolio('markowitz', target_return)
    
    
    #create scatterplots of special porfolios
    x1 = port1.volatility_annual
    y1 = port1.return_annual
    x2 = port2.volatility_annual
    y2 = port2.return_annual
    x3 = port3.volatility_annual
    y3 = port3.return_annual
    x4 = port4.volatility_annual
    y4 = port4.return_annual
    x5 = port5.volatility_annual
    y5 = port5.return_annual
    x6 = port6.volatility_annual
    y6 = port6.return_annual

    
    #plot Efficient Frontier
    plt.figure()
    plt.title('Efficient Frontier for a portfolio including ' + rics[0])
    plt.scatter(volatilities,returns)
    if include_min_var:
        plt.plot(x1, y1,"oy", label = label1)
        plt.plot(x2, y2,"or", label = label2)
    plt.plot(x3, y3, "^y", label=label3) # yellow triangle
    plt.plot(x4, y4, "^r", label=label4) # red triangle
    plt.plot(x5, y5, "sy", label=label5) # yellow square
    plt.plot(x6, y6, "sr", label=label6) # red square
    plt.ylabel('portfolio return')
    plt.xlabel('portfolio volatility')
    plt.grid()
    if include_min_var:
        plt.legend(loc='best')
    else:
        plt.legend(loc='lower right',  borderaxespad=0.)
    plt.show()
    
    dict_portfolios = {label1: port1,
                       label2: port2,
                       label3: port3,
                       label4: port4,
                       label5: port5,
                       label6: port6}
    
    return dict_portfolios
    
    
    
    
    
    
    

    

