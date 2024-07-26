# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:46:57 2024

@author: Nacho
"""

import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsPoisson(NPaths, NSteps, T, xiP):
    
    X = np.zeros([NPaths,NSteps+1])
    Xc = np.zeros([NPaths,NSteps+1])
    time = np.zeros([NSteps+1])
    
    dt = T / float(NSteps)
    
    Z = np.random.poisson(xiP*dt, [NPaths,NSteps])
    
    for i in range(0,NSteps):
                   
        X[:,i+1] = X[:,i] + Z[:,i]
        #Compensate Poisson Process
        Xc[:,i+1] = Xc[:,i] - xiP * dt + Z[:,i] 
        time[i+1] = time[i] + dt
        
     
    paths = {"time":time, "X":X, "Xcomp":Xc}
    return paths
 
    
def Example():
    NPaths = 25
    NSteps = 500
    T = 300
    xiP = 1
    
    paths = GeneratePathsPoisson(NPaths, NSteps, T, xiP)
    
    timeGrid = paths["time"]
    X = paths["X"]
    Xc = paths["Xcomp"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X),'-b')
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(Xc), '-b')
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Xcomp")
    
    #check martingale property II, haya la media de la última columna
    #Media de los retornos
    ES = np.mean(Xc[:,-1])
    print(ES)
    
    
    
Example()