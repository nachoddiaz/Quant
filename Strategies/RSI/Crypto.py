# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:02:44 2024

@author: Nacho
"""

import importlib

import support_functions
importlib.reload(support_functions)

import RSI_FIFO
importlib.reload(RSI_FIFO)


symbol = 'ADA/EUR'
days = 800
operational_days = 365
interval = '1d'

support = support_functions.manager(symbol, days, interval)

datos_preparados = support.prepare_data()

rsi_fifo = RSI_FIFO.manager(datos_preparados, symbol)

rsi_fifo.rsi_strategy()
rsi_fifo.plot_strategie()
rsi_fifo.implement_strategie()
rsi_fifo.compute_stats(operational_days)

annualized_return = rsi_fifo.annualized_return
volatility_annual = rsi_fifo.volatility_annual

print('The mean daily return for a FIFO RSI strategy is: '+ str(rsi_fifo.mean) + ' %')
print('The annualized return for this FIFO RSI strategy is: '+ str(annualized_return) + ' %')
print('The volatility for this FIFO RSI strategy is: '+ str(volatility_annual) + ' %')


plot_result = support.plot_result(rsi_fifo.info)





