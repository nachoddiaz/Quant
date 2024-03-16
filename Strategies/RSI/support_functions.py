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
  
        
  
class manager:
    
    def __init__(self,symbol,days, interval):
        self.df_done = None
        self.apds_done = None
        self.symbol = symbol
        self.days = days
        self.interval = interval
        self.ohlcv = None
        self.df_symbols = None
        self.df_position = None
        self.returns = None
        self.purchases = []
        self.sales = []
        self.daily_return = None
    
    def prepare_data(self):
        exchange = ccxt.binance()
        exchange.load_markets()
        symbols_path = './symbols_binance.xlsx'
        self.df_symbols = pd.read_excel(symbols_path)
        self.df_symbols = self.df_symbols['Symbol'].values

        if self.symbol in self.df_symbols:
             # Get symbol price
             self.ohlcv = exchange.fetch_ohlcv(self.symbol, self.interval, limit=self.days)
        else: print('symbol doesnt exists')
        
        self.df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        
        return self.df
       