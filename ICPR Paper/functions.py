from __future__ import division

import random, csv, time, os, pickle, re, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.stats import bernoulli
from scipy.io import savemat

from datetime import datetime

def mle(w, pairs, sigma):    
    out = 1      
    for pair in pairs:
        if pair[0] == -1 or pair[1] == -1:
            continue
        out *= 1/(1+np.exp(-w[pair[0]] + w[pair[1]]))   
    return -np.log(out)

def gradient(w,pairs, sigma):
    grad = []
    for i in range(len(w)):
        gradient = 0

        for pair in pairs:
            if i == pair[0]:
                out = -1
            elif i == pair[1]:
                out = 1  
            else:
                continue
            gradient -= out / (1/((np.exp(w[pair[1]]-w[pair[0]]))/sigma) +1 ) / sigma

        grad.append(-gradient)
        
    return np.array(grad)

def hessian(w, pairs):
    hessian = np.zeros((300, 300))
    for pair in pairs:
        exp_0 = np.exp(w[pair[1]]-w[pair[0]])
        hess = exp_0/(1+exp_0)**2
        hessian[pair[0], pair[1]] = hess
        hessian[pair[1], pair[0]] = hess
        
    return hessian

def init_graph(num_nodes, num_edges, reference_matrix=None):
    # If the number of edges is lower than num_nodes-1, raise error
    if num_edges < num_nodes:
        raise AttributeError('The number of edges must be >= num_nodes')

    # Define list of neighbors (with initial circular connection)
    neighbors = [[i-1, i+1] for i in range(num_nodes)]
    neighbors[0][0], neighbors[-1][1] = num_nodes-1, 0

    # Define list of node indices, maximum degree and number of placed nodes
    nodes, max_degree, num_placed = range(num_nodes), int(np.ceil((2.0 * num_edges) / num_nodes)), num_nodes

    def rewire_pairs(p0, p1):
        # Select existing pair
        ph1 = np.random.choice(num_nodes)
        ph2 = neighbors[ph1][np.random.choice(len(neighbors[ph1]))]
        while p0 in neighbors[ph2] or p1 in neighbors[ph1] or p0 in [ph1, ph2] or p1 in [ph1, ph2]:
            ph1 = np.random.choice(num_nodes)
            ph2 = neighbors[ph1][np.random.choice(len(neighbors[ph1]))]

        # Rewire with current pair
        neighbors[ph1].remove(ph2)
        neighbors[ph2].remove(ph1)
        neighbors[ph1].append(p1)
        neighbors[p1].append(ph1)
        neighbors[ph2].append(p0)
        neighbors[p0].append(ph2)

    # Start placing random edges
    ct = 0
    while num_placed < num_edges:
        # If only one node left, rewire with existing pair
        if len(nodes) == 1:
            rewire_pairs(nodes[0], nodes[0])
            num_placed += 1
            continue

        # Sample random pair of nodes
        pair = np.random.choice(nodes, size=2, replace=False)

        # If pair already connected, do rewiring
        if pair[0] in neighbors[pair[1]]:
            if ct < 100:
                ct += 1
                continue
            rewire_pairs(pair[0], pair[1])
            num_placed += 1

        # Else, place edge between nodes
        else:
            neighbors[pair[0]].append(pair[1])
            neighbors[pair[1]].append(pair[0])
            num_placed += 1

        ct = 0

        # If max degree reached for first node, remove from sampling list
        if len(neighbors[pair[0]]) >= max_degree:
            nodes.remove(pair[0])

        # If max degree reached for second node, remove from sampling list
        if len(neighbors[pair[1]]) >= max_degree:
            nodes.remove(pair[1])

    # Return pairs
    return [(
        (i, j) if reference_matrix is None or reference_matrix[i] > reference_matrix[j] else (j, i)
    ) for i, il in enumerate(neighbors) for j in sorted(il) if i < j]


def matching_func(param, video_score, w_hat):
    return np.linalg.norm(video_score - param[0]*np.array(w_hat) - param[1])

def regularized_vector(video_score,w_hat):
    coeff = optimize.minimize(matching_func, [0, 0], args=(video_score, w_hat))

    a = coeff['x'][0]
    b = coeff['x'][1]
    v = a*np.array(w_hat)+b
    return v

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def noise_generation(video_score, pair):
    difference = (video_score[pair[0]] - video_score[pair[1]])/4
    p = sigmoid(difference)*(1-sigmoid(difference))
    decision = bernoulli.rvs(p,size=1)

    return decision

def R2(yhat, y):
    ybar = np.sum(y)/len(y) 
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    return ssreg/sstot