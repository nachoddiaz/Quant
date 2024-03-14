# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:05:05 2024

@author: Nacho
"""

import pandas as pd
import ccxt
import mplfinance as mpf
import pandas_ta as ta
  
        
  
class manager:
    
    def __init__(self,symbol,days):
        self.df_done = None
        self.apds_done = None
        self.symbol = symbol
        self.days = days
        self.ohlcv = None
        self.df_symbols = None
        self.df_position = None
        self.returns = None
        self.purchases = []
        self.sales = []
        
    
    def rsi_strategy(self):
        exchange = ccxt.binance()
        exchange.load_markets()
        symbols_path = './symbols_binance.xlsx'
        self.df_symbols = pd.read_excel(symbols_path)
        self.df_symbols = self.df_symbols['Symbol'].values

        if self.symbol in self.df_symbols:
             # Get symbol price
             self.ohlcv = exchange.fetch_ohlcv(self.symbol, '1d', limit=self.days)
        else: print('symbol doesnt exists')
        
        
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        #DataFrame for RSI indicator
        df['RSI'] = ta.rsi(df['close'], length=14)
        df = df.dropna()

        # Identifica los cruces hacia arriba y hacia abajo del RSI con las líneas de 70 y 30
        crossing_up_70 = (df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)
        crossing_down_70 = (df['RSI'] < 70) & (df['RSI'].shift(1) >= 70)
        crossing_up_30 = (df['RSI'] > 30) & (df['RSI'].shift(1) <= 30)
        crossing_down_30 = (df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)

        crossing_up_70_date = df.index[crossing_up_70]
        crossing_down_70_date = df.index[crossing_down_70]
        crossing_up_30_date = df.index[crossing_up_30]
        crossing_down_30_date = df.index[crossing_down_30]


        crossing_up_70_rsi = df['RSI'][crossing_up_70]
        crossing_down_70_rsi = df['RSI'][crossing_down_70]
        crossing_up_30_rsi = df['RSI'][crossing_up_30]
        crossing_down_30_rsi = df['RSI'][crossing_down_30]

        crossing_up_70_data = pd.Series(crossing_up_70_rsi.values, index=crossing_up_70_date)
        crossing_down_70_data = pd.Series(crossing_down_70_rsi.values, index=crossing_down_70_date)
        crossing_up_30_data = pd.Series(crossing_up_30_rsi.values, index=crossing_up_30_date)
        crossing_down_30_data = pd.Series(crossing_down_30_rsi.values, index=crossing_down_30_date)

        df['crossing_up_70_data'] = crossing_up_70_data
        df['crossing_down_70_data'] = crossing_down_70_data
        df['crossing_up_30_data'] = crossing_up_30_data
        df['crossing_down_30_data'] = crossing_down_30_data


        # Graphic panel for RSI bellow candlestick graphic)
        apds = [mpf.make_addplot(df['RSI'], panel=2, color='blue', ylabel='RSI', ylim=(0, 100)),
                # Add horizontal lines in 70 and 30 levels of RSI
                mpf.make_addplot([70]*len(df.index), panel=2, type='line', color='red', linestyle='--', width=0.7),
                mpf.make_addplot([30]*len(df.index), panel=2, type='line', color='red', linestyle='--', width=0.7),
        # Crea gráficos de dispersión para los cruces usando los datos preparados
                mpf.make_addplot(df['crossing_up_70_data'], scatter=True, markersize=50, marker='v', color='red', panel=2),
                mpf.make_addplot(df['crossing_down_70_data'], scatter=True, markersize=40, marker='.', color='yellow', panel=2),
                mpf.make_addplot(df['crossing_up_30_data'], scatter=True, markersize=40, marker='.', color='yellow', panel=2),
                mpf.make_addplot(df['crossing_down_30_data'], scatter=True, markersize=50, marker='^', color='green', panel=2)]

        self.df_done = df
        self.apds_done = apds
    
    
    def plot_strategie(self):
        mpf.plot(self.df_done, type='candle', style='charles',
                 title='Gráfico de Velas '+self.symbol+' con RSI y Cruces en Escala Logarítmica',
                 ylabel='Precio (USDT)',
                 volume=True,
                 ylabel_lower='Volumen',
                 addplot=self.apds_done,
                 figratio=(12, 8),
                 mav=(10, 20),
                 yscale='log')
        
        
    def implement_strategie(self):
        
        self.purchases = pd.DataFrame(columns=['price'])
        self.sales = pd.DataFrame(columns=['price'])
        self.df_position = self.df_done[['close','crossing_up_70_data']]
        self.df_position = self.df_position.dropna()
        self.df_position2 = self.df_done[['close','crossing_down_30_data']]
        self.df_position2 = self.df_position2.dropna()
        self.df_position= pd.concat([self.df_position, self.df_position2], axis=0)
        self.df_position = self.df_position.sort_values('timestamp')
        self.df_position = self.df_position.reset_index()

        
        for i in range(len(self.df_position)):
            #If the crossing_down_30_data column is empty, it means that they
            # are crossing_up_70_data values ​​and vice versa
            if pd.isna(self.df_position['crossing_down_30_data'].iloc[i]):
                self.sales.loc[i, 'date'] = self.df_position.iloc[i]['timestamp']
                self.sales.loc[i,'price'] = self.df_position.iloc[i]['close']
            elif pd.isna(self.df_position['crossing_up_70_data'].iloc[i]):
                self.purchases.loc[i, 'date'] = self.df_position.iloc[i]['timestamp']
                self.purchases.loc[i,'price'] = self.df_position.iloc[i]['close']
        
        self.diff_nan = len(self.sales) - len(self.purchases)
            
        # # #To make the same number of purchases and sales using FIFO
        # if self.diff_nan <= 0:
        #     self.purchases = self.purchases[:self.diff_nan]
        # else:
        #     self.sales = self.sales[:-self.diff_nan]
            
            
        
            
    def compute_stats(self, Operational_days):
        
        #first_day = 
           
        #Relative Return
        self.returns = [(v - c) for c, v in zip(self.purchases['price'], self.sales['price'])]
        # self.returns = sum(self.returns)
        
        #Absolut Return
        self.abs_return = sum(self.sales['price']) - sum(self.purchases['price'])

        