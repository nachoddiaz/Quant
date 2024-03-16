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

import RSI_FIFO
importlib.reload(RSI_FIFO)

import RSI_LIFO
importlib.reload(RSI_LIFO) 

symbol = 'ADA/EUR'
days = 800
operational_days = 365
interval = '1d'

datos_preparados = support_functions.manager(symbol, days, interval)

datos_preparados = datos_preparados.prepare_data()

rsi_fifo = RSI_FIFO.manager(datos_preparados, symbol)
rsi_lifo = RSI_LIFO.manager(datos_preparados, symbol)

rsi_fifo.rsi_strategy()
rsi_fifo.plot_strategie()
rsi_fifo.implement_strategie()
rsi_fifo.compute_stats(operational_days)


rsi_lifo.rsi_strategy()
rsi_lifo.plot_strategie()
rsi_lifo.implement_strategie()
rsi_lifo.compute_stats(operational_days)




annualized_return = rsi_fifo.annualized_return
volatility_annual = rsi_fifo.volatility_annual

print('The mean daily return for a FIFO RSI strategy is: '+ str(rsi_fifo.mean) + ' %')
print('The annualized return for this FIFO RSI strategy is: '+ str(annualized_return) + ' %')
print('The volatility for this FIFO RSI strategy is: '+ str(volatility_annual) + ' %')








