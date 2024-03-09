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

import market_data
importlib.reload(market_data)
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
    

returns = []
volatilities = []
sharpe = []

number_rics = 5
rics = random.sample(universe, number_rics)
notional = 1


prt_mng = porfolios.manager(rics, notional, number_rics)
prt_mng.compute_covariance()
out = porfolios.output(rics, notional)


#compute desired portfolio
port_min_var_L1 = prt_mng.compute_portfolio('min_var_L1')
port_min_var_L2 = prt_mng.compute_portfolio('min_var_L2')
port_eq_weigth = prt_mng.compute_portfolio('eq_weigth')
port_long_only = prt_mng.compute_portfolio('long_only')
#A target return is needed in Markowirz,
# if isnt given, it uses the mean return of the rics
port_markowitz = prt_mng.compute_portfolio('markowitz', target_return=0.15)

#DataFrame to compare weights of rics in different strategies
df_weights = pd.DataFrame()
df_weights['rics'] = rics
for strategie in strategies_str:
    opt_port = prt_mng.compute_portfolio(portfolio_type=strategie)
    df_weights[strategie] = opt_port.weights
    
strategies = [port_min_var_L1, port_min_var_L2, port_eq_weigth, \
              port_long_only, port_markowitz]

#DataFrame to compare reults of different strategies
for portafolio in strategies:
    returns.append(portafolio.return_annual)
    volatilities.append(portafolio.volatility_annual)
    sharpe.append(portafolio.sharpe_ratio)
    
    
df = pd.DataFrame()
df['strategies'] = strategies_str
df['returns'] = returns
df['volatility'] = volatilities
df['sharpe ratio'] = sharpe



port_min_var_L1.plot_histogram()
port_min_var_L2.plot_histogram()
port_eq_weigth.plot_histogram()
port_long_only.plot_histogram()
port_markowitz.plot_histogram()
