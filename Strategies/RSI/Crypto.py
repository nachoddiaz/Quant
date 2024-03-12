# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:02:44 2024

@author: Nacho
"""
import pandas as pd
import ccxt
import mplfinance as mpf
import pandas_ta as ta
import importlib

import support_functions
importlib.reload(support_functions)

symbol = 'BTC/EUR'
days = 700

manager = support_functions.manager(symbol, days)
manager.rsi_strategy()
manager.plot_strategie()

manager.implement_strategie()

#DataFrame for the price data
returns = manager.returns

print('The return for this RSI strategy is: '+ str(returns))








