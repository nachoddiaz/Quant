# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:56:41 2024

@author: Nacho
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:44:52 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st 
import importlib
import random
import scipy.optimize as op


import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)

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
notional = 10

# rics = ['^MXX','^SPX','XLK','XLF','XLV','XLP','XLY','XLE','XLI']

# rics = ['^MXX','^SPX','^IXIC', '^STOXX', '^GDAXI', '^FCHI','^VIX', \
#         'BTC-USD','ETH-USD','USDC-USD','SOL-USD','USDT-USD','DAI-USD']
    
# rics = ['BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD']

    
df = capm.model.sync_returns(rics)

mtx= df.drop(columns=['date'])
mtx_var_cov = np.cov(mtx, rowvar=False) *252
mtx_correl = np.corrcoef(mtx, rowvar=False)

#min variance portfolio with eigenverctors
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_cov)
min_var_vector = eigenvectors[:,0]


#min variance portfolio with optimize
#def portfolio_var:
    
    
x0 = [notional/len(rics)] * len(rics)
optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
            args=(len(rics),\
                  notional,\
                  position_beta_usd))
hedge_weights = optimal_result.x
hedge_delta_usd = np.sum(hedge_weights)
hedge_beta_usd = np.transpose(len(rics)).dot(hedge_weights).item()











