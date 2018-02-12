#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:28:25 2017

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
#    Do ANN                                                                #
############################################################################
learning_rate = 0.15
def sigmoid (x):
    return 1/(1 + np.exp(-x))
def derivatives_sigmoid(x):
    return x * (1 - x)
def cost(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
def do_forward_propagation(x,w):
    a = {}
    z = {}
    input_data = x.T
    a['layer_1'] = np.zeros((input_data.shape[0] + 1,input_data.shape[1]))
    a['layer_1'][1:] = input_data
    a['layer_1'][0]= a['layer_1'][0] + 1  # add 1 bias for input layer
    z['layer_2'] = w['layer_1'].dot(a['layer_1'])
    a['layer_2'] = np.zeros((z['layer_2'].shape[0] + 1,z['layer_2'].shape[1]))
    a['layer_2'][1:] = sigmoid(z['layer_2'])
    a['layer_2'][0] = a['layer_2'][0] + 1   # add 1 bias for hidden layer
    z['layer_3'] = w['layer_2'].dot(a['layer_2'])
    a['layer_3'] = sigmoid(z['layer_3'])
    return z,a
def do_back_propagation(y,a,w):
    delta = {}
    error_o =(a['layer_3'] - y)
    delta['layer_2'] = error_o .dot(a['layer_2'].T)
    error_h = np.zeros((z['layer_2'].shape[0],z['layer_2'].shape[1]))
    error_h[0] = error_o*w['layer_2'][0][1] * derivatives_sigmoid(a['layer_2'][1])
    error_h[1] = error_o*w['layer_2'][0][2] * derivatives_sigmoid(a['layer_2'][2])
    delta['layer_1'] = (error_h.dot(a['layer_1'].T))
    return delta

input_node = 2
hidden_node = 2
output_node = 1

Nr_of_correct = 0
Nr_of_wrong=0


for trainning_test in range(0,100):
    theta = []
    initial_theta = {}
    initial_theta['layer_1'] = np.random.uniform(size = (hidden_node, input_node+1))
    initial_theta['layer_2'] =  np.random.uniform(size = (output_node, hidden_node+1))
    theta.append(initial_theta)
    # get trainning data
    trainning_data = np.delete(X,int(trainning_test),0)
    trainning_result = np.delete(t,int(trainning_test),0)
    for index_iter in np.arange(start = 1, stop = 1000, step = 1):
        cur_theta = theta[index_iter -1]    
        z,a = do_forward_propagation(trainning_data, cur_theta)
        h = a['layer_3'] 
        delta = do_back_propagation(trainning_result.T, a, cur_theta)    
        new_theta = {}
        new_theta["layer_2"] = cur_theta["layer_2"] - learning_rate * delta["layer_2"]
        new_theta["layer_1"] = cur_theta["layer_1"] - learning_rate * delta["layer_1"]    
        theta.append(new_theta)
    z,a = do_forward_propagation(np.array([X[trainning_test]]), theta[-1])
    result = 0
    if a['layer_3'] > 0.5:
        result = 1
    if result == t[trainning_test]:
        Nr_of_correct += 1
    else:
        Nr_of_wrong += 1

    
   
    