#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:01:47 2017

@author: jchen60
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
plt.rc('text', usetex = False)
plt.rc('font', family='serif')
marker_size = 7

############################################################################
#    Read the data                                                         #
############################################################################

in_file_name = "/Users/chenjing/Downloads/dataset_1.csv"
data_in = pd.read_csv(in_file_name,header=0)
data = data_in.as_matrix()
v1 = data[:,0]
v2 = data[:,1]
lable = data[:,2]
############################################################################
#    Plot V2 vs V1                                                         #
############################################################################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('V2 vs V1 without lable')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.plot(v1[:], v2[:], linestyle='None', marker='o', markersize=marker_size, color='blue')
#ax.plot(v1[0:30], v2[0:30], linestyle='None', marker='o', markersize=marker_size, color='blue', label='1')
#ax.plot(v1[30:], v2[30:], linestyle='None', marker='o', markersize=marker_size, color='red', label='0')
ax.legend()

fig.show()    
############################################################################
#       PCA Function                                                       #
############################################################################

def do_pca(data):
    result = {}
    data_for_use = data[:,0:2]
    data_for_use.mean(axis=0)
    mean_centred = data_for_use - np.mean(data_for_use, axis = 0) 
    cov = np.cov(mean_centred, rowvar = False)
    evalues , evectors = LA.eigh(cov)
    idx = np.argsort(evalues)[::-1]
    evectors = evectors[:,idx]
    evalues = evalues[idx]
    b = np.dot(mean_centred , evectors)
    result["loadings"]= evectors
    result["ei_value"]= evalues
    result["scores"] = b
    return result

pca_result = do_pca(data)
############################################################################
#    Plot raw data to PC1                                                  #
############################################################################
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title('Raw data to PC1 without label')
ax.scatter(pca_result["scores"][:,0],np.zeros(60), color='blue')
#ax.scatter(pca_result["scores"][:30,0], np.zeros(30), color='red')
#ax.scatter(pca_result["scores"][30:,0],np.zeros(30), color='blue')
ax.set_xlabel('PC1')
ax.set_ylabel()
fig.show()

############################################################################
#    Add PC1 axis                                                          #
############################################################################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Raw data and PC1 axis without label')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.plot(v1[:], v2[:], linestyle='None', marker='o', markersize=marker_size, color='blue')
#ax.plot(v1[0:30], v2[0:30], linestyle='None', marker='o', markersize=marker_size, color='blue', label='1')
#ax.plot(v1[30:], v2[30:], linestyle='None', marker='o', markersize=marker_size, color='red', label='0')
k=-30
ax.plot([0, (-1)*k*pca_result["loadings"][0,0]], [0, (-1)*k*pca_result["loadings"][1,0]],color='green', linewidth=3)
ax.legend()

fig.show()    

############################################################################
#   LDA function                                                           #
############################################################################
def do_lda(class1, class2):
    u1=class1.mean(0)
    u2=class2.mean(0)

    data_mean_1 = class1 - u1
    data_mean_2 = class2 - u2

    Sw=data_mean_1.T.dot(data_mean_1)+data_mean_2.T.dot(data_mean_2)
    w=np.mat((u1-u2))*np.mat(Sw).I
    w = np.array(w).reshape(2,1)  
    return w

X1 = data[:30, :2]
X2 = data[30:, :2]

w = do_lda(X1, X2)
projection = np.matmul(data[:, :2], w)

############################################################################
#    Plot projection on W                                                  #
############################################################################
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Raw data to W without label')
ax.set_xlabel('projection')
ax.set_ylabel('')
ax.plot(projection[:], np.zeros(60), linestyle='None', marker='o', markersize=marker_size, color='blue')
#ax.plot(projection[0:30], np.zeros(30), linestyle='None', marker='o', markersize=marker_size, color='blue', label='1')
#ax.plot(projection[30:], np.zeros(30), linestyle='None', marker='o', markersize=marker_size, color='red', label='0')
ax.legend()

fig.show()  

############################################################################
#    Add W axis                                                            #
############################################################################

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Raw data, PC1 axis and w axis without label')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.plot(v1[:], v2[:], linestyle='None', marker='o', markersize=marker_size, color='blue')
#ax.plot(v1[0:30], v2[0:30], linestyle='None', marker='o', markersize=marker_size, color='blue', label='1')
#ax.plot(v1[30:], v2[30:], linestyle='None', marker='o', markersize=marker_size, color='red', label='0')
k=-30
ax.plot([0, (-1)*k*pca_result["loadings"][0,0]], [0, (-1)*k*pca_result["loadings"][1,0]],color='green', linewidth=3)
ax = fig.add_subplot(1, 1, 1)
w_scaled = w * 12.0 /w[0]
ax.plot([0, w_scaled[1]], [0, w_scaled[0]], color='pink')
ax.legend()

fig.show()     
 
############################################################################
#    Variance and Eigenvalues                                              #
############################################################################
variance_on_pc1 = np.var(pca_result["scores"][:,0])
variance_on_pc2 = np.var(pca_result["scores"][:,1])
eigenvalues = pca_result["ei_value"]

############################################################################
#    Variance on w                                                         #
############################################################################
variance_on_w = np.var(projection)
