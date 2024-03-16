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

symbol = 'ADA/EUR'
days = 800
operational_days = 252
interval = '1d'

manager = support_functions.manager(symbol, days, interval)
manager.rsi_strategy()
manager.plot_strategie()

manager.implement_strategie()

manager.compute_stats(operational_days)
annualized_return = manager.annualized_return
volatility_annual = manager.volatility_annual

#print(manager.ticker['first'])
print(manager.ticker['last'])

print('The mean daily return for this RSI strategy is: '+ str(manager.mean) + ' %')
print('The annualized return for this RSI strategy is: '+ str(annualized_return) + ' %')
print('The volatility for this RSI strategy is: '+ str(volatility_annual) + ' %')








