# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 21:44:52 2024

@author: Nacho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats  as st 
import importlib


import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)

rics = ['^MXX','^SPX','XLK','XLF','XLV','XLP','XLY','XLE','XLI']

rics = ['^MXX','^SPX','^IXIC', '^STOXX', '^GDAXI', '^FCHI','^VIX', \
        'BTC-USD','ETH-USD','USDC-USD','SOL-USD','USDT-USD','DAI-USD']
    
rics = ['BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD']

    
df = capm.model.sync_returns(rics)

# compute variance-covariance matrix

#To only have the returns column
mtx= df.drop(columns=['date'])
mtx_var_cov = np.cov(mtx, rowvar=False) *252

#compute correlation matrix
mtx_correl = np.corrcoef(mtx, rowvar=False)

#compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(mtx_var_cov)
var_explained = eigenvalues / np.sum(eigenvalues)
#matricial product of two vectors
prod = np.matmul(eigenvectors, np.transpose(eigenvectors))


#########################
# 2D PCA Visulasitation #
#########################

#Compute min and max vol, alwaus are the first and the last
vol_min = np.sqrt(eigenvalues[0])
vol_max = np.sqrt(eigenvalues[-1])

#compute PCA base for 2D visualisation
pca_vector_1 = eigenvectors[:,-1]
pca_vector_2 = eigenvectors[:,-2]
pca_eigenvalue_1 = eigenvalues[-1]
pca_eigenvalue_2 = eigenvalues[-2]
pca_var_explained = var_explained[-2:].sum()

#compute min variance portfolio
min_var_vector = eigenvectors[:,0]
min_var_eigenvalue = eigenvalues[0]
min_var_explained = var_explained[0]







