# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:04:12 2024

@author: Nacho
"""

import numpy as np
import matplotlib.pyplot as plt

def heston_model(S0, v0, T, r, kappa, theta, sigma, rho, n_steps, n_paths):
    dt = T / n_steps
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    v[0] = v0
    
    for t in range(1, n_steps + 1):
        Z1 = np.random.normal(size=n_paths)
        Z2 = np.random.normal(size=n_paths)
        dW_S = Z1 * np.sqrt(dt)
        dW_v = (rho * Z1 + np.sqrt(1 - rho**2) * Z2) * np.sqrt(dt)
        
        v[t] = np.maximum(v[t-1] + kappa * (theta - v[t-1]) * dt + sigma * np.sqrt(v[t-1]) * dW_v, 0)
        S[t] = S[t-1] * np.exp((r - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1]) * dW_S)
    
    return S, v

# Parámetros del modelo
S0 = 100
v0 = 0.04
T = 1
r = 0.03
kappa = 2.0
theta = 0.04
sigma = 0.1
rho = -0.7
n_steps = 1000
n_paths = 10000

# Simulación del modelo de Heston
S, v = heston_model(S0, v0, T, r, kappa, theta, sigma, rho, n_steps, n_paths)

# Graficar algunos caminos del precio del activo
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, T, n_steps + 1), S[:, :100])
plt.xlabel('Time (t)')
plt.ylabel('Asset Price (S)')
plt.title('Heston Model Simulation')
plt.show()
