# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:32:26 2024

@author: Nacho
"""

#Calculation of Impl_Vol via Newton-Raphson
import numpy as np
import scipy.stats as st
import time

start_time = time.time()
#Market Data
V_market = 2.0 #Option price
K = 130 #Strike
tau = 1 # time-to-maturity
r = 0.05 
S_0 = 100 #Todays stock price
sigmaInit = 0.1 #Arbitrary impl_vol
CP = "c" #C is call / P is put

def ImpliedVolatility(CP, S_0, K, sigma, tau, r):
    error = 1e10 #Max error that we are willing to assume
    optPrice = lambda sigma: BS_Call_Opt_Price(CP, S_0, K, sigma, tau, r)
    vega = lambda sigma: Dv_dsigma(S_0, K, sigma, tau, r)
    
    #Vamos iterando hasta que el error sea menor al indicado
    n = 1
    while error>1e-10:
        g = optPrice(sigma) - V_market
        g_prim = vega(sigma)
        new_sigma = sigma - g/g_prim
        
        error = np.abs(new_sigma - sigma)
        
        sigma = new_sigma
        
        print('iteration {0} with error = {1}'.format(n,error))
        
        n +=1
        
    return sigma
        
    
    
def BS_Call_Opt_Price(CP, S_0, K, sigma, tau, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if (CP == "c"):
        price = S_0 * st.norm.cdf(d1) - K * np.exp(-r * tau) * st.norm.cdf(d2)
    else:
        price = K * np.exp(-r * tau) * st.norm.cdf(-d2) - S_0 * st.norm.cdf(-d1)
    return price
    
    
    
def Dv_dsigma(S_0, K, sigma, tau, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    
    return S_0 * st.norm.pdf(d1) * np.sqrt(tau)


impl_vol = ImpliedVolatility(CP, S_0, K, sigmaInit, tau, r)
result = '''Implied volatility for CallPrice= {}, strike K={}, 
      maturity T= {}, interest rate r= {} and initial stock S_0={} 
      equals to sigma_imp = {:.7f}'''.format(V_market,K,tau,r,S_0,impl_vol)
      
print (result)

#Check if impl_vol returns the option price
BS_vol = BS_Call_Opt_Price(CP, S_0, K, impl_vol, tau, r)
print('Option Price for implied volatility of {0} is equal to {1}'.format(impl_vol, BS_vol))
    
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.4f} seconds")
    