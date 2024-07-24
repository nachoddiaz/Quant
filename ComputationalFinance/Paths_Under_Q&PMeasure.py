# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:06:56 2024

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
    paths = {"time":time, "S":S}
    return paths


def Example():
    NPaths = 8
    NSteps = 1000
    S_0 = 1
    T = 10
    r = 0.05
    mu = 0.15
    sigma = 0.1
    
    #Money savings account
    M = lambda t: np.exp(r*t)
    
    
    #Montecarlo paths
    PathsQ = GeneratePathsGBMABM(NPaths, NSteps, T, r, sigma, S_0)
    S_Q = PathsQ["S"]
    
    PathsP = GeneratePathsGBMABM(NPaths, NSteps, T, mu, sigma, S_0)
    S_P = PathsP["S"]
    
    timeGrid = PathsQ["time"]
    
    
    #Discounted Stock paths
    S_Qdisc = np.zeros([NPaths,NSteps+1])
    S_Pdisc = np.zeros([NPaths,NSteps+1])
    i=0
    for i, ti in enumerate(timeGrid):
        S_Qdisc[:,i] = S_Q[:,i]/M(ti)
        S_Pdisc[:,i] = S_P[:,i]/M(ti)
        
        
    #Plotting 
    
    plt.figure(1)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    eSM_Q = lambda t: S_0 * np.exp(r *t) / M(t)
    plt.plot(timeGrid,eSM_Q(timeGrid),'r--')
    plt.plot(timeGrid, np.transpose(S_Qdisc),'blue')   
    plt.legend(['E^Q[S(t)/M(t)]','paths S(t)/M(t)'])
    
    plt.figure(2)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    eSM_P = lambda t: S_0 * np.exp(mu *t) / M(t)
    plt.plot(timeGrid,eSM_P(timeGrid),'r--')
    plt.plot(timeGrid, np.transpose(S_Pdisc),'blue')   
    plt.legend(['E^P[S(t)/M(t)]','paths S(t)/M(t)'])
    
    #Only the Q measure is martingale because 
    # eSM_Q is cte
    
    
Example()