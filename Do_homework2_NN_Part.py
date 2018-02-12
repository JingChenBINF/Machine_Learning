#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:40:01 2017

@author: jchen60
"""

import numpy as np
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
import matplotlib.pyplot as plt
plt.rc('text', usetex = False)
plt.rc('font', family='serif')
marker_size = 5
fig_width = 15
fig_height = 6


x = np.array([[0.05],[0.1]])
y = np.array([[0.01],[0.99]])
theta = []
cost_list= []

num_iterations = 1000

learning_rate = 0.5
def sigmoid (x):
    return 1/(1 + np.exp(-x))
def derivatives_sigmoid(x):
    return x * (1 - x)
def cost(y, t): 
    return ((t - y)**2).sum()
def do_forward_propagation(x,w):
    a = {}
    z = {}
    a['layer_1'] = np.zeros((3,1))
    a['layer_1'][1:] = x
    a['layer_1'][0]= 1.0
    z['layer_2'] = w['layer_1'].dot(a['layer_1'])
    a['layer_2'] = np.zeros((3,1))
    a['layer_2'][1:] = sigmoid(z['layer_2'])
    a['layer_2'][0] = 1.0
    z['layer_3'] = w['layer_2'].dot(a['layer_2'])
    a['layer_3'] = sigmoid(z['layer_3'])
    return z,a
def do_back_propagation(y,a,w):
    delta = {}
    error_o = (a['layer_3'] - y)*derivatives_sigmoid(a['layer_3'])
    delta['layer_2'] = error_o * a['layer_2'].T
    error_h = error_o*np.array([w['layer_2'][0,1:]]).dot(derivatives_sigmoid(a['layer_2'][1:]))+  error_o*np.array([w['layer_2'][1,1:]]).dot(derivatives_sigmoid(a['layer_2'][1:]))
    delta['layer_1'] = error_h *  a['layer_1'].T
    return delta

input_node = 2
hidden_node = 2
output_node = 2
initial_theta = {}
initial_theta['layer_1'] = np.random.uniform(size = (hidden_node, input_node+1))
initial_theta['layer_2'] =  np.random.uniform(size = (output_node, hidden_node+1))

theta.append(initial_theta)

for index_iter in np.arange(start = 1, stop = num_iterations, step = 1):
    cur_theta = theta[index_iter -1]
    z,a = do_forward_propagation(x,cur_theta)
    h = a['layer_3']
    print h
    cur_cost = cost(y,h)
    
    delta = do_back_propagation(y, a, cur_theta)
    
    new_theta = {}
    new_theta["layer_2"] = cur_theta["layer_2"] - learning_rate * delta["layer_2"]
    new_theta["layer_1"] = cur_theta["layer_1"] - learning_rate * delta["layer_1"]
    
    theta.append(new_theta)
    cost_list.append(cur_cost)


######################################################
#    plot
######################################################    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(range(len(cost_list)), cost_list, color = 'blue')
fig.show()

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(2, 3, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta1_{10}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_1'][0][0], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 2)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta1_{11}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_1'][0][1], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 3)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta1_{12}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_1'][0][2], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 4)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta1_{20}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_1'][1][0], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 5)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta1_{21}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_1'][1][1], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 6)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta1_{22}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_1'][1][2], color='blue', s=marker_size)
fig.show()

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(2, 3, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta2_{10}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_2'][0][0], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 2)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta2_{11}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_2'][0][1], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 3)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta2_{12}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_2'][0][2], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 4)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta2_{20}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_2'][1][0], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 5)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta2_{21}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_2'][1][1], color='blue', s=marker_size)
ax = fig.add_subplot(2, 3, 6)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta2_{22}$')
for i in range(num_iterations):
    ax.scatter(i, theta[i]['layer_2'][1][2], color='blue', s=marker_size)
fig.show()