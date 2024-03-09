# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:14:17 2024

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

number_rics =10
#rics = random.sample(universe, number_rics)
rics = ['XLC', 'XLU','QUAL','BTC-USD','XLY','NOKUSD=X' ,'XLB' ,'^GDAXI','GOOG','CHFUSD=X']
notional = 1
print(rics)


#efficient frontier
target_return = 0.075
include_min_var = True
portfolio_dic = porfolios.compute_eff_front(rics, notional, number_rics, target_return, include_min_var)
