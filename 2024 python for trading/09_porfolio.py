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
strategies_str = ['min_var_L1', 'min_var_L2', 'eq_weigth', \
              'long_only', 'markowitz']

  
weights = []
returns = []
volatilities = []
sharpe = []

number_rics = 5
rics = random.sample(universe, number_rics)
notional = 100


prt_mng = porfolios.manager(rics, notional, number_rics)
prt_mng.compute_covariance()
output = porfolios.output(rics, notional)


#compute desired portfolio
port_min_var_L1 = prt_mng.compute_portfolio('min_var_L1')
port_min_var_L2 = prt_mng.compute_portfolio('min_var_L2')
port_eq_weigth = prt_mng.compute_portfolio('eq_weigth')
port_long_only = prt_mng.compute_portfolio('long_only')
#A target return is needed in Markowirz,
# if isnt given, it uses the mean return of the rics
port_markowitz = prt_mng.compute_portfolio('markowitz', target_return=0.15)

strategies = [port_min_var_L1, port_min_var_L2, port_eq_weigth, \
              port_long_only, port_markowitz]

for portafolio in strategies:
    returns.append(portafolio.return_annual)
    volatilities.append(portafolio.volatility_annual)
    sharpe.append(portafolio.sharpe_ratio)

# returns.append(port_min_var_L1.return_annual)
# returns.append(port_min_var_L2.return_annual)
# returns.append(port_eq_weigth.return_annual)
# returns.append(port_long_only.return_annual)
# returns.append(port_markowitz.return_annual)

# volatilities.append(port_min_var_L1.volatility_annual)
# volatilities.append(port_min_var_L2.volatility_annual)
# volatilities.append(port_eq_weigth.volatility_annual)
# volatilities.append(port_long_only.volatility_annual)
# volatilities.append(port_markowitz.volatility_annual)

# sharpe.append(port_min_var_L1.sharpe_ratio)
# sharpe.append(port_min_var_L2.sharpe_ratio)
# sharpe.append(port_eq_weigth.sharpe_ratio)
# sharpe.append(port_long_only.sharpe_ratio)
# sharpe.append(port_markowitz.sharpe_ratio)


df_weights = pd.DataFrame()
df_weights['rics'] = rics
for strategie in strategies_str:
    opt_port = prt_mng.compute_portfolio(portfolio_type=strategie)
    df_weights[strategie] = opt_port.weights
    
df = pd.DataFrame()
df['strategies'] = strategies_str
df['returns'] = returns
df['volatility'] = volatilities
df['sharpe ratio'] = sharpe


