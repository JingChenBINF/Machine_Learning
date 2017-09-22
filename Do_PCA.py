#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 16:49:09 2017

@author: jchen60
"""

# --------------------------------------------------------------- #
#                               PCA                               #
# --------------------------------------------------------------- #

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------
# Step 0: Read the data

df = pd.read_csv(
    filepath_or_buffer='/Users/chenjing/Downloads/dataset_1.csv', 
    header=0, 
    sep=',')

a = np.array(df,dtype=float)

# ---------------------------------------------------------------
# Step 1: Mean-centre X

a.mean(axis=0)
a = a - np.mean(a, axis = 0)  
########Caculate the variance
print "Variance of x: " + str(a.var(axis=0)[0])
print "Variance of y: " + str(a.var(axis=0)[1])
print "Variance of z: " + str(a.var(axis=0)[2])
# ---------------------------------------------------------------
# Step 2: Caculate covariance

cov = np.cov(a, rowvar = False)

########Print the covariance
cov
print "Covariance between x & y: " + str(cov[1][0])
print "Covariance between y & z: " + str(cov[1][2])
# ---------------------------------------------------------------
# Step 3: Caculate eigenvalues and eigenvectors

evalues , evectors = LA.eigh(cov)

idx = np.argsort(evalues)[::-1]
evectors = evectors[:,idx]
evalues = evalues[idx]
b = np.dot(a, evectors) 
b
# ---------------------------------------------------------------
# Step 4: Plot

for i in range(0,len(b)):
    plt.plot(b[i][0],b[i][1], 'ro')
plt.title("PCA result")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


#################################################################
#Caculate eigenvelue and eigenvector for 3                      #
#################################################################
matrix_a = np.array([[0, -1], [2,3]], int)
matrix_a
evalues , evectors = LA.eig(matrix_a)
evalues
evectors
