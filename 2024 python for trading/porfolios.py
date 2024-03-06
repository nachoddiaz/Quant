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
        
    
    def compute_portfolio(self, portfolio_type=None, target_return=None): 
        #inputs
        x0 = [self.notional/ len(self.rics)] * len(self.rics)
        L2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] #unitary in norm L2
        L1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] #unitary in norm L1
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
            if
            elif
            elif
            optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
                                          args=(self.mtx_var_cov),\
                                          constraints=(L1_norm + markowitz) \
                                              )
            weights = np.array(optimal_result.x)

        else :
            weights = np.array(x0)
         
        optimal_portfolio = output(self.rics, self.notional)
        optimal_portfolio.type = self.portfolio_type
        optimal_portfolio.weights = self.notional * weights / sum(abs(weights))
        
        return optimal_portfolio
        
class output: 
    
    def __init__(self,rics,notional):
        self.rics = rics
        self.notional = notional
        self.type = None
        self.weights = None
        pass
    

