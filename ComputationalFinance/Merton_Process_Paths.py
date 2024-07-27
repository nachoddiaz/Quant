# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 02:30:05 2024

@author: Nacho
"""

import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsMerton(NPaths, NSteps, S_0, T, xiP, muJ, sigmaJ, r, sigma):
     
     X = np.zeros([NPaths,NSteps+1])
     S = np.zeros([NPaths,NSteps+1])
     time = np.zeros([NSteps+1])
     dt = T / float(NSteps)
     

     X[:,0] = np.log(S_0)
     S[:,0] = S_0
     
     # Expectation E(e^J) for J~N(muJ,sigmaJ^2)
     EeJ = np.exp(muJ + 0.5*sigmaJ*sigmaJ)
    
     ZPois = np.random.poisson(xiP*dt,[NPaths,NSteps])
     Z = np.random.normal(0.0,1.0,[NPaths,NSteps])
    
     J = np.random.normal(muJ,sigmaJ,[NPaths,NSteps])
     
     
     for i in range(0,NSteps):
         if NPaths > 1:
             Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
             
         X[:,i+1] = X[:,i] + (r - xiP*(EeJ-1) - 0.5*sigma*sigma)* dt + \
             sigma *np.sqrt(dt)* Z[:,i] + J[:,i]*ZPois[:,i]
         time[i+1] = time[i] + dt
         
         
     S = np.exp(X)
     paths = {"time":time, "X":X, "S":S}
     return paths


def Example():
    NPaths = 25
    NSteps = 500
    T = 5
    xiP = 1
    #Parameters of J dsitribution
    muJ = 0
    sigmaJ = 0.5
    sigma = 0.2
    S_0 = 100
    r = 0.05
    
    paths = GeneratePathsMerton(NPaths, NSteps, S_0, T, xiP, muJ, sigmaJ, r, sigma)
    
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
    plt.ylabel("S")
    
    #check martingale property II, haya la media de la última columna
    #Media de los retornos
    ES = np.mean(S[:,-1])
    print(ES)
    
    # plt.figure(3)
    # plt.hist(Xc[:,-1])
    # plt.grid()
   
    
    
Example()