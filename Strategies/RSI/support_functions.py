# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:05:05 2024

@author: Nacho
"""

import pandas as pd
import ccxt
import mplfinance as mpf
import pandas_ta as ta
import numpy as np
import scipy.stats  as st 
import importlib

import RSI_FIFO
importlib.reload(RSI_FIFO)

        
  
class manager:
    
    def __init__(self,symbol,days, interval):
        self.symbol = symbol
        self.days = days
        self.interval = interval
    
    def prepare_data(self):
        exchange = ccxt.binance()
        exchange.load_markets()
        symbols_path = './symbols_binance.xlsx'
        df_symbols = pd.read_excel(symbols_path)
        df_symbols = df_symbols['Symbol'].values

        if self.symbol in df_symbols:
             # Get symbol price
             ohlcv = exchange.fetch_ohlcv(self.symbol, self.interval, limit=self.days)
        else: print('symbol doesnt exists')
        
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        
        return self.df
    
        
    def plot_result(self, info):
        self.info = info
        
        
        
        