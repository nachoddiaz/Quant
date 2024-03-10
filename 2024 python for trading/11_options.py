# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:01:49 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st 
import scipy.optimize  as op 
import importlib
import os

import market_data
importlib.reload(market_data)
import options
importlib.reload(options)



inputs = options.inputs()
inputs.price = 69000
inputs.time = 0
inputs.maturity = 1
inputs.time_to_maturity = inputs.maturity - inputs.time
inputs.strike = 60000
inputs.interest_rate = 0.0453
inputs.volatility = 0.5556
inputs.type = 'call'

option_mng = options.manager(inputs)

option_mng.compute_black_scholes_price()
option_mng.compute_montecarlo_price()
option_mng.plot_histogram()

BSPrice = option_mng.black_scholes_price
MCPrice = option_mng.montecarlo_price