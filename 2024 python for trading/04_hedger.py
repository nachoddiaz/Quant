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

import capm
importlib.reload(capm)

#inputs
benchmark = '^SPX'
position_security = '^SPX'
position_delta_usd = 10 # in M USDC
hedge_securites = ['GOOG','AAPL', 'MSFT', 'AMZN']
epsylon = 0.0

hedger = capm.hedger(position_security, position_delta_usd, hedge_securites, benchmark)
hedger.compute_betas()


#########################
# Only for 2 securities #
#########################
hedger.compute_hedge_weights(epsylon)
print('Our position of ' + str(position_delta_usd) + ' million USD of '  + str(position_security))
print('Has been hedged with ' + str(hedger.hedge_weights[0]) + ' million USD of '+ str(hedge_securites[0])+ ' and ' + str(hedger.hedge_weights[1]) + ' Million USD of ' + str(hedge_securites[1]) )
hedge_weights_exact = hedger.hedge_weights


########################
#     Generalizing     #
########################
betas = hedger.hedge_betas
target_delta = hedger.position_delta_usd
target_beta = hedger.position_beta_usd
def cost_function(x, betas, target_delta, target_beta):
    dimension = len(x)
    deltas = np.ones([dimension])
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2
    f_beta = (np.transpose(betas).dot(x).item() + target_beta)**2
    #f_penalty =
    f = f_delta + f_beta #+ f_penalty
    return f
    
#initial condition
x0 = -target_delta/len(betas) * np.ones(len(betas))

#This function can cover our posotion_security with 
# as many as securities we want
optimal_result = op.minimize(fun=cost_function, x0=x0,\
                    args=(betas,target_delta,target_beta))

hedge_weights_optimal = optimal_result.x

print('optimal result:')
print(optimal_result)












