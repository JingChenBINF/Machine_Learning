#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:03:11 2017

@author: jchen60
"""

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
import matplotlib.pyplot as plt
plt.rc('text', usetex = False)
plt.rc('font', family='serif')
marker_size = 7

############################################################################
#    Read the data                                                         #
############################################################################

in_file_name = "/Users/chenjing/Downloads/SCLC_study_output_filtered_2.csv"
data_in = pd.read_csv(in_file_name, index_col=0)
X = data_in.as_matrix()
y = np.concatenate((np.zeros(20), np.ones(20)))

II_0 = np.where(y==0)
II_1 = np.where(y==1)

II_0 = II_0[0]
II_1 = II_1[0]
############################################################################
#   LDA function                                                           #
############################################################################
def do_lda(class1, class2):
    u1=class1.mean(0)
    u2=class2.mean(0)

    data_mean_1 = X1 - u1
    data_mean_2 = X2 - u2

    Sw=data_mean_1.T.dot(data_mean_1)+data_mean_2.T.dot(data_mean_2)
    w=np.mat((u1-u2))*np.mat(Sw).I
#    w = np.array(w).reshape(19,1)  
    return w

############################################################################
#   Do LDA                                                                 #
############################################################################

X1 = X[II_0, :]
X2 = X[II_1, :]

w = do_lda(X1, X2)
projection = np.dot(X,w.T)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying LDA to cell line data')
ax.set_xlabel('projection')
ax.set_ylabel('')
ax.plot(projection[0:20], np.zeros(20), linestyle='None', marker='o', markersize=marker_size, color='blue', label='NSCLC')
ax.plot(projection[20:40], np.zeros(20), linestyle='None', marker='o', markersize=marker_size, color='red', label='NSCLC')
ax.legend()

fig.show()    
############################################################################
#   LDA use sklearn                                                        #
############################################################################
sklearn_LDA = LDA(n_components=2)
sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)
sklearn_LDA_projection = -sklearn_LDA_projection
sklearn_LDA_projection

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying sklearn LDA to cell line data')
ax.set_xlabel(r'$W_1$')
ax.set_ylabel('')
ax.plot(sklearn_LDA_projection[II_0], np.zeros(len(II_0)), linestyle='None', marker='o', markersize=marker_size, color='blue', label='NSCLC')
ax.plot(sklearn_LDA_projection[II_1], np.zeros(len(II_1)), linestyle='None', marker='o', markersize=marker_size, color='red', label='SCLC')
ax.legend()
fig.show()




