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
    
    def __init__(self,rics,notional):
        self.rics = rics
        self.notional = notional
        self.mtx_var_cov = None
        self.mtx_correl = None
        
    def compute_covariance(self):
        df = capm.model.sync_returns(self.rics)
        mtx= df.drop(columns=['date'])
        self.mtx_var_cov = np.cov(mtx, rowvar=False) *252
        self.mtx_correl = np.corrcoef(mtx, rowvar=False)
    
    
    def compute_porfolio(self, portfolio_type):           
        x0 = [self.notional/len(self.rics)] * len(self.rics)
        L2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] #unitary in norm L2
        L1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] #unitary in norm L2
        optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
                    args=(self.mtx_var_cov),\
                    constraints=L1_norm)
        optimize_vector = optimal_result.x 
        # Normalize with L1 norm
        optimize_vector = self.notional * optimize_vector / sum(abs(optimal_result.x))
         
        
# class output: 
    
#     def __init__(self):
        
