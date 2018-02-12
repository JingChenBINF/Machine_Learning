#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:40:06 2017

@author: chenjing
"""
import numpy as np
from sklearn.datasets import load_iris

############################################################################
#    Read the data                                                         #
############################################################################
iris = load_iris()
II_virginica = np.where(iris.target==2)
II_versicolor = np.where(iris.target==1)
II_versicolor = II_versicolor[0]
II_virginica = II_virginica[0]
virginica_petal_length_width = iris.data[II_virginica, 2:4]
versicolor_petal_length_width = iris.data[II_versicolor, 2:4]
a_column_min = virginica_petal_length_width.min(axis=0)
a_column_max = virginica_petal_length_width.max(axis=0)
virginica_petal_length_width = (virginica_petal_length_width-a_column_min)/(a_column_max-a_column_min)
b_column_min = versicolor_petal_length_width.min(axis=0)
b_column_max = versicolor_petal_length_width.max(axis=0)
versicolor_petal_length_width = (versicolor_petal_length_width-a_column_min)/(a_column_max-a_column_min)

X = np.vstack((virginica_petal_length_width,versicolor_petal_length_width))
t = np.vstack((np.zeros((50,1)), np.ones((50,1))))

############################################################################
#    Do logistic regression                                                #
############################################################################

Nr_of_correct = 0
Nr_of_wrong=0
alpha = 5

for iteration in range(0,100):    
    theta = np.zeros((99, 2))
    theta[0, :] = 2.5 * np.ones(2)
    # get trainning data
    trainning_list =  range(0,100)
    trainning_list.remove(int(iteration))
    for ind in range(0,98):
        J = np.zeros(99)
        partial_derivative = np.zeros(2)
        for index in range(0,99):
            cur_z = sum(X[trainning_list[index],:] * theta[ind, :])
            cur_y_hat = 1.0 / (1.0 + np.exp(-cur_z))
            cur_residual = cur_y_hat - t[trainning_list[index]]
            partial_derivative = partial_derivative + X[trainning_list[index],:] * cur_residual
            cur_cost = t[trainning_list[index]] * np.log10(cur_y_hat) + (1.0-t[trainning_list[index]]) * np.log10(1.0 - cur_y_hat)
            J[index] = J[index] + cur_cost
        J[ind] = -J[ind] / 99
        theta[ind+1, :] = theta[ind, :] - alpha * partial_derivative / 99
    result = 0
    final_theta = theta[-1, :]
    final_theta = final_theta.reshape((len(final_theta), 1))
    z_predict = np.matmul(X[iteration], final_theta)
    y_predict_probability = 1.0 / (1.0 + np.exp(-z_predict))
    if y_predict_probability >= 0.5:
        result = 1
    else:
        result = 0
    if result == t[iteration]:
        Nr_of_correct+= 1
    else: 
        Nr_of_wrong += 1