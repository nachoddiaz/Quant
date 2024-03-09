# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:02:01 2024

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


class inputs:
    
    def __init__(self):
        self.price = None
        self.time = None
        self.maturity = None
        self.strike = None
        self.interest_rate = None
        self.volatility = None
        self.type = None #Call or Put

    
class manager:
    
    def __init__(self, inputs):
        self.price = inputs.price
        self.time = inputs.time
        self.maturity = inputs.maturity
        self.time_to_maturity = inputs.maturity -inputs.time
        self.strike = inputs.strike
        self.interest_rate = inputs.interest_rate
        self.volatility = inputs.volatility
        self.type = inputs.type
        self.black_scholes_price = None
        self.montecarlo_price = None
        self.montecarlo_confidence_interval = 95
        
        
    def compute_black_scholes_price(self):
        d1 = 1 / (self.volatility * np.sqrt(self.time_to_maturity)) * \
            (np.log(self.price/self.strike) + \
            (self.interest_rate + 0.5*self.volatility**2)* self.time_to_maturity)
        
        d2 = d1 - self.volatility * self.time_to_maturity
        if self.type == 'call':
            price = self.price * st.norm.cdf(d1) - \
            st.norm.cdf(d2) * self.strike * np.exp(-self.interest_rate*self.time_to_maturity)
        
        elif self.type == 'put':
            price = st.norm.cdf(-d2) * self.strike * np.exp(-self.interest_rate*self.time_to_maturity)\
            - self.price * st.norm.cdf(-d1)
            
        self.black_scholes_price = price
        
        
    def compute_montecarlo_price(self):
        self.montecarlo_price = self.price*\
            np.exp((self.interest_rate-0.5*self.volatility**2)*self.time_to_maturity+\
                   self.volatility*)
        
        
        
        
        
        
        
        

    
    