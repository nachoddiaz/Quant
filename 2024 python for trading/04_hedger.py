# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 16:30:35 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats  as st
import scipy.optimize as op
import importlib
import os

# import capm
# importlib.reload(capm)

def cost_function(x, roots, coeffs):
    f = 0
    for n in range(len(x)):
        f += coeffs[n]*(x[n] - roots[n])**2
    return f

dimensions = 5
roots = np.random.randint(low=-20, high=20, size=dimensions)
coeffs = np.ones(dimensions)

x = np.zeros(dimensions)

optimal_result = op.minimize(fun=cost_function, x0=x, args=(roots,coeffs))

print('optimal result:')
print(optimal_result)


# #inputs
# benchmark = '^SPX'
# position_security = 'BTC-USD'
# position_delta_usd = 1 # in M USDC
# hedge_securites = ['^SPX','^VIX']
# epsylon = 0.01

# hedger = capm.hedger(position_security, position_delta_usd, hedge_securites, benchmark)
# hedger.compute_betas()
# hedger.compute_optimal_hedge(epsylon)
# print('Our position of ' + str(position_delta_usd) + ' million USD of '  + str(position_security))
# print('Has been hedged with ' + str(hedger.hedge_weights[0]) + ' million USD of '+ str(hedge_securites[0])+ ' and ' + str(hedger.hedge_weights[1]) + ' Million USD of ' + str(hedge_securites[1]) )