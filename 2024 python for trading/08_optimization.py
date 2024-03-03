# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:56:41 2024

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
notional = 15

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
min_var_vector = notional * min_var_vector / sum(abs(min_var_vector))

#min variance portfolio with optimize
def portfolio_var(x, mtx_var_cov):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_cov,x))
    return variance
       
x0 = [notional/len(rics)] * len(rics)
L2_norm = [{"type": "eq", "fun": lambda x: sum(x**2) - 1}] #unitary in norm L2
L1_norm = [{"type": "eq", "fun": lambda x: sum(abs(x)) - 1}] #unitary in norm L2
optimal_result = op.minimize(fun=portfolio_var, x0=x0,\
            args=(mtx_var_cov),\
            constraints=L1_norm)
optimize_vector = optimal_result.x 
# Normalize with L1 norm
optimize_vector = notional * optimize_vector / sum(abs(optimal_result.x))
 


df_weigths = pd.DataFrame()
df_weigths['rics'] = rics
df_weigths['min_var_vector']= min_var_vector
wei = sum(abs(min_var_vector))
df_weigths['optimize_vector']= optimize_vector














