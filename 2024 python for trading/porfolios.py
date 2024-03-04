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

class manager: 
    
    def __init__(self,rics,notional):
        self.rics = rics
        self.notional = notional
        self.mtx_var_cov = None
        self.mtx_correl = None
        self.mtx = None
        
    def mtx(self):
        df = capm.model.sync_returns(self.rics)
        mtx= df.drop(columns=['date'])
        return self.mtx

    def var_matrix(self):
        mtx_var_cov = np.cov(self.mtx, rowvar=False) *252
        return mtx_var_cov
    
    def corr_matrix(rics):
        df = capm.model.sync_returns(rics)
        mtx= df.drop(columns=['date'])
        mtx_correl = np.corrcoef(mtx, rowvar=False)
        return mtx_correl
    
    
   # def min_var_porfolio(self):
        
class output: 
    
    def __init__(self):
        
