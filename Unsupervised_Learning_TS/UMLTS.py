from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.api as sm
import yfinance as yf
#import pandas_ta
import warnings
warnings.filterwarnings("ignore")

sp500 = pd.read_html('https://es.wikipedia.org/wiki/Anexo:Compa%C3%B1%C3%ADas_del_S%26P_500')[0]

sp500['Símbolo'] = sp500['Símbolo'].str.replace('.', '-')

symbols_list = sp500['Símbolo'].unique().tolist()

end_date = '2024-01-20'
start_date = pd.to_datetime(end_date)-pd.DateOffset(356*8)

df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date)

df