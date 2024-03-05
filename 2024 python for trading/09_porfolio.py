# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:40:15 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st 
import importlib
import random
import scipy.optimize as op


import porfolios
importlib.reload(porfolios)
import capm
importlib.reload(capm)


#inputs
universe = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI','^FCHI','^VIX',\
            'XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU',\
            'SPY','EWW',\
            'IVW','IVE','QUAL','MTUM','SIZE','USMV',\
            'AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
            'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
            'LLY','JNJ','PG','MRK','ABBV','PFE',\
            'BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD',\
            'EURUSD=X','GBPUSD=X','CHFUSD=X','SEKUSD=X','NOKUSD=X','JPYUSD=X','MXNUSD=X'\
            ]
    
rics = random.sample(universe, 5)
notional = 15


#compute corr & var matrix
# df = capm.model.sync_returns(rics)
# mtx= df.drop(columns=['date'])
# mtx_var_cov = np.cov(mtx, rowvar=False) *252
# mtx_correl = np.corrcoef(mtx, rowvar=False)



prt_mng = porfolios.manager(rics, notional)

tres = prt_mng.compute_covariance()


#mtx_correl = porfolios.manager.corr_matrix



#compute desired portfolio
port_min_var_L1 = prt_mng.compute_portfolio('min_var_L1')
port_min_var_L2 = prt_mng.compute_portfolio('min_var_L2')
port_eq_weigth = prt_mng.compute_portfolio('eq_weigth')
port_vol_weigth = prt_mng.compute_portfolio('volatility_weigth')



