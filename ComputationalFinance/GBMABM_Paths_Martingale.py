# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 25:49:45 2024

@author: Nacho
"""

import numpy as np
import matplotlib.pyplot as plt


def GeneratePathsGBMABM(NPaths, NSteps, T, r, sigma, S_0):
    
    np.random.seed(1)
    
    Z = np.random.normal(0.0, 1.0, [NPaths,NSteps])
    X = np.zeros([NPaths,NSteps+1])
    S = np.zeros([NPaths,NSteps+1])
    time = np.zeros([NSteps+1])
    
    #Se coloca el valor del log(S_0) en la primera columna
    X[:,0] = np.log(S_0)
    
    dt = T / float(NSteps)
    
    for i in range(0,NSteps):
        if NPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        X[:,i+1] = X[:,i] + (r - 0.5* sigma*sigma)* dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] + dt
        
        
    S = np.exp(X)
    paths = {"time":time, "X":X, "S":S}
    return paths


def Example():
    NPaths = 5000
    NSteps = 500
    T = 1
    r = 0.05
    sigma = 0.4
    S_0 = 100
    
    #Money savings account
    M = lambda r,t: np.exp(r*T)
    
    paths = GeneratePathsGBMABM(NPaths, NSteps, T, r, sigma, S_0)
    
    timeGrid = paths["time"]
    X = paths["X"]
    S = paths["S"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    
    #check martingale property II, haya la media de la última columna
    #Media de los retornos
    ES = np.mean(S[:,-1])
    print(ES)
    
    #Discounted stock value
    DSV = np.mean(S[:,-1])/M(r,T)
    print(DSV)
    
    
Example()




    