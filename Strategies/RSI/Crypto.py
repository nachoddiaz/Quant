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

manager = support_functions.manager(symbol)
manager.rsi_strategy()
manager.plot_strategie()

#DataFrame for the price data









