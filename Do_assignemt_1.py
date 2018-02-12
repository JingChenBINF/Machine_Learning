#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:32:25 2017

@author: chenjing
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn import linear_model


#############################
#       PCA Function        #
#                           #
#############################

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
#    b = np.dot(mean_centred , evectors)
    result["loadings"]= evectors
    result["ei_value"]= evalues
    return result

#############################
#    Linear Regression      #
#       Function            #
#############################   

def do_linear_regression(data):
    result = {}
    x = data[:,0]
    y = data[:,1]
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    cov_xy = np.sum((y - y_bar) * (x - x_bar))
    var_x = np.sum((x - x_bar)**2)
    beta_1_hat = cov_xy / var_x
    beta_0_hat = y_bar - beta_1_hat * x_bar
    result["beta_1_hat"] = beta_1_hat
    result["beta_0_hat"] = beta_0_hat
    return result

#############################
#        Question           #
#          1                #
############################# 
in_file_name = "/Users/chenjing/Downloads/linear_regression_test_data.csv"
dataIn = pd.read_csv(in_file_name,header=0, index_col=0)
data = dataIn.as_matrix()
x = data[:,0]
y = data[:,1]
y_theoretical = data[:,2]
# do pca on x and y
pca_result = do_pca(data)
# plot for (1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, color='blue')
ax.scatter(x, y_theoretical, color='red')
k=5
ax.plot([0, (-1)*k*pca_result["loadings"][0,0]], [0, (-1)*k*pca_result["loadings"][1,0]],color='green', linewidth=3)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
# do linear_regression on x and y
linear_result = do_linear_regression(data)
beta_1 = linear_result["beta_1_hat"]
beta_0 = linear_result["beta_0_hat"]
y_prediction = beta_0 + beta_1 * x
# plot for (2)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, color='blue')
ax.scatter(x, y_theoretical, color='red')
k=5
ax.plot(x, y_prediction, color='pink')
ax.plot([0, (-1)*k*pca_result["loadings"][0,0]], [0, (-1)*k*pca_result["loadings"][1,0]],color='green', linewidth=3)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()
#############################
#        Question           #
#          2                #
############################# 
diabetes = datasets.load_diabetes()
x = diabetes.data[:,2]
y = diabetes.target
data_dia = np.column_stack((x, y))
# get trainning data and testing data
random_testing_data_index = np.random.choice(range(0,len(data_dia)),size = 20,replace=False)
all_index = np.array(range(0,len(data_dia)))
temp_list1 = []
temp_list2 = []
for i in all_index:    
    if i in random_testing_data_index: 
        temp_list1.append(data_dia[i])
        print i
    else: 
        temp_list2.append(data_dia[i])
testing_data = np.array(temp_list1, dtype=float)
trainning_data = np.array(temp_list2, dtype=float)
# run linear_regression_model
testing_x = testing_data[:,0]
testing_y = testing_data[:,1]
trainning_x = trainning_data[:,0]
trainning_y = trainning_data[:,1]
lm_sklearn= linear_model.LinearRegression()
trainning_x = trainning_x.reshape((len(trainning_x), 1))
lm_sklearn.fit(trainning_x, trainning_y)
lm_sklearn_result = {}
lm_sklearn_result['beta_0_hat'] = lm_sklearn.intercept_
lm_sklearn_result['beta_1_hat'] = lm_sklearn.coef_
# get predicted y
presicted_testing_y = lm_sklearn_result['beta_1_hat'] * testing_x + lm_sklearn_result['beta_0_hat']
# plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(testing_x, testing_y, color='blue')
ax.scatter(testing_x, presicted_testing_y, color='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.show()