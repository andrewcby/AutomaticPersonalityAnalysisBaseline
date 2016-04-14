
# coding: utf-8

# In[1]:

import random, csv, time, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.stats import bernoulli
from scipy.io import savemat


# # Functions
#  * **mle(w, pairs)**
#  
#  Calculate the MLE of w.
#  
#  Due to the fact that optimize.minimize only takes 1 input, here the pair information is imported from global variable.
#  
#  
#  * **gradient(w,pairs)**
#  
#  Calculate the gradient of MLE. 
#  
#  Returns a numpy array with value of gradient.
#  
#  
#  * **hessian(w, pairs)**
#  
#  Calculate the hessian of MLE. 
#  
#  Returns a numpy matrix with value of hessian.
#  
#  
#  * **compare_rank(video_score, results, verbose=False, hist=False)**
#  
#  Compute the true rankings and rankings from results. 
#  
#  `verbose` will output with columns: Result Order, True Order, Result Score, Ture Score
#     
#  `hist` will output a histogram
# 
# 
#  * **init_graph(num_nodes, num_edges, reference_matrix=None)**
#  
#  Marc's method of generating pairs
#  
#  
#  * **matching_func(param, video_score, w_hat)**
# 
#  Function used to calculate L2 norm of v and w_star. Used to find a and b.
#  Here param is [a, b]
# 
#     
#  * **regularized_vector(video_score,w_hat)**
# 
#  Function to generate vector v after using matching_func to find optimal a and b
#  
#  
#  
#  * **performance_isabelle(video_score, video_num, w_hat)**
# 
#  Performance evaluation using method proposed by Isabelle.
#  
#  
#  * **performance_nihar(video_score, video_num, w_hat)**
# 
#  Performance evaluation using method proposed by Nihar.
#  
#  
#  * **noise_generation(video_score, pair)**
# 
#  Generate a decision of whether to flip, using proability from normal distribution and then Bernoulli to decide whether to flip
#  
#  
#  * **sigmoid(x)**
#  
#  

# In[2]:

def mle(w, pairs):    
    out = 1      
    for pair in pairs:
        if pair[0] == -1 or pair[1] == -1:
            continue
        out *= 1/(1+np.exp(-w[pair[0]] + w[pair[1]]))   
    return -np.log(out)

def gradient(w,pairs):
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
            gradient -= out / (1/(np.exp(w[pair[1]]-w[pair[0]])) +1 )

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

# def compare_rank(video_score, results, verbose=False, hist=False):
#     true_order = np.array(video_score).argsort()
#     true_ranks = true_order.argsort()

#     temp_o = np.array(results).argsort()
#     temp_r = temp_o.argsort()

#     resolution = 0.1
#     video_score_results = np.round(np.array(results)/resolution)*resolution
    
#     if verbose:
#         print 'Result Order \t True Order \t Result Score \t Ture Score'
#         for i in range(len(temp_r)):
#             print temp_r[i], '\t\t', true_ranks[i], '\t\t', video_score_results[i], '\t\t', video_score[i]
            
#     if hist:
#         diff = np.abs(temp_r - true_ranks)
#         plt.hist(diff, alpha=0.5)

        
def compare_rank(video_score, results, hist=False):
    scale_and_bias = np.dot(np.linalg.pinv(np.concatenate((results[:, None],
                                                           np.ones((len(results), 1))), axis=1)), video_score)

    difference = np.abs(video_score - (results * scale_and_bias[0] + scale_and_bias[1]))
    return difference
        

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

def performance_isabelle(video_score, video_num, w_hat):
    epsilon = 0.0001
    error = 0
    v = regularized_vector(video_score,w_hat)

    true_order = np.array(video_score).argsort()

    what = [ w_hat[i] for i in true_order]
    wstar = [ video_score[i] for i in true_order]

    for i in range(video_num-2):
        error += np.abs((wstar[i+1]-wstar[i])/(wstar[i+2]-wstar[i]+epsilon)-
                        (what[i+1]-what[i])/(what[i+2]-what[i]+epsilon))
    
    return error

def performance_nihar(video_score, w_hat):
    v = regularized_vector(video_score,w_hat)
    return np.linalg.norm(video_score - v)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def noise_generation(video_score, pair):
    difference = video_score[pair[0]] - video_score[pair[1]]
    p = sigmoid(difference)*(1-sigmoid(difference))
    decision = bernoulli.rvs(p,size=1)

    return decision


# # Data Generation
# The cell below generates data for $video\_num$ videos, calculates all possible pairs and stores in $pairs\_truth$ and $total_pairs$ is the number of pairs. 

if __name__ == '__main__':

    video_nums = [625,1250,2500,5000,10000,20000]
    num_edges_pct = [0.01,0.1,1,10,100]

    # video_nums = [20, 50]
    # num_edges_pct = [0.1]

    for iteration in range(10):
        for video_num in video_nums:
            for num_edge_pct in num_edges_pct:

                num_edge = int(num_edge_pct*video_num*np.log(video_num))
                if num_edge < video_num:
                    num_edge = video_num
            
                video_score = np.random.uniform(-5,5,video_num)
                resolution = 0.1
                video_score = np.round(video_score/resolution)*resolution

                w = np.ones(video_num)

                test_pairs = init_graph(video_num, num_edge, video_score)

                # start_time = time.time()

                res = optimize.minimize(mle, w, 
                                        method='Newton-CG',
                                        jac=gradient,
                                        args=(test_pairs,),
                                        tol = 7,
                                        options={'disp': False})
                filename = 'result/'+ str(video_num)+'_at_'+str(num_edge)+'_iter_'+str(iteration)+'.mat'

                savemat(filename, res)
                filename = 'result/truth_'+ str(video_num)+'_at_'+str(num_edge)+'_iter_'+str(iteration)+'.mat'
                scores = {}
                scores['video_score'] = video_score
                savemat(filename, scores)
                # print  'Time Spent: %.2f seconds' %float(time.time() - start_time)


# In[31]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



