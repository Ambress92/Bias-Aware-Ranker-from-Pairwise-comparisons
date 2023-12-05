import math
import numpy as np

from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import linalg
import scipy as sp


def _safe_exp(x):
    x = min(x, 500)
    return math.exp(x)

def sigmoid(x):
    return 1/(1 + _safe_exp(-x))


class Pairwise_with_rev:

    """Optimization-related methods for pairwise-comparison data.
    
    This class provides methods to compute the negative log-likelihood (the
    "objective") and its gradient given model parameters and
    pairwise-comparison data, considering multiple reviewers
    
    data is a set, where each element of the set contains pairs from a reviewer
    
    objective, gradient_scores and gradient_revs are for BARP
    
    """

    def __init__(self, data, penalty, classes):
        self._data = data
        self._classes = classes
        self._penalty = penalty

    def objective(self, params, rev_params):
        """Compute the negative penalized log-likelihood, for the item scores and the reviewers' parameters"""
        val = self._penalty * np.sum(params**2)
        for i in self._data:
            for win, los in self._data[i]:
                val += np.logaddexp(0, -(params[win] + rev_params[i] * self._classes[win] - params[los] - rev_params[i] * self._classes[los]))
        return val

    def gradient_scores(self, params, rev_params):
        """Compute the gradient of the negative log-likelihood wrt the item scores"""
        grad = 2 * self._penalty * params
        for i in self._data:
            for win, los in self._data[i]:
                z = 1 / (1 + _safe_exp(params[win] + rev_params[i] * self._classes[win] - params[los] - rev_params[i] * self._classes[los]))
                grad[win] += -z
                grad[los] += +z
        return grad
    
    def gradient_revs(self, params, rev_params):
        """Compute the gradient of the negative log-likelihood wrt the reviewers' parameters"""
        grad = np.zeros(len(rev_params))
        for i in self._data:
            for win, los in self._data[i]:
                z =  (self._classes[los] - self._classes[win])/ (1 + _safe_exp(params[win] + rev_params[i] * self._classes[win] - params[los] - rev_params[i] * self._classes[los]))
                grad[i] += z
                
        return grad

    def crowdbt_objective(self, params, rev_params):
        """Compute the negative penalized log-likelihood, for the item scores and the reviewers' parameters, for crowdbt"""
        val = self._penalty * np.sum(params**2)
        for i in self._data:
            for win, los in self._data[i]:
                val += -np.log(rev_params[i] * _safe_exp(params[win]) / (_safe_exp(params[win]) + _safe_exp(params[los])) + (1 - rev_params[i]) * _safe_exp(params[los]) / (_safe_exp(params[win]) + _safe_exp(params[los])))
        return val   
    
    
    def crowdbt_gradient_scores(self, params, rev_params):
        grad =  2 * self._penalty * params
        for i in self._data:
            for win, los in self._data[i]:
                z = 1 / (rev_params[i] * _safe_exp(params[win]) / (_safe_exp(params[win]) + _safe_exp(params[los])) + (1 - rev_params[i]) * _safe_exp(params[los]) / (_safe_exp(params[win]) + _safe_exp(params[los]))) * _safe_exp(params[win]) * _safe_exp(params[los]) / ( _safe_exp(params[win]) + _safe_exp(params[los]) )**2 * (2*rev_params[i] -1)
                
                grad[win] += -z
                grad[los] += +z
            
        return grad

    def crowdbt_gradient_revs(self, params, rev_params):
        """Compute the gradient of the negative log-likelihood wrt the reviewers' parameters"""
        grad = np.zeros(len(rev_params))
        for i in self._data:
            for win, los in self._data[i]:
                z =  -1/(rev_params[i] * _safe_exp(params[win]) / (_safe_exp(params[win]) + _safe_exp(params[los])) + (1 - rev_params[i]) * _safe_exp(params[los]) / (_safe_exp(params[win]) + _safe_exp(params[los]))) * ( _safe_exp(params[win]) - _safe_exp(params[los]) ) / (_safe_exp(params[win]) + _safe_exp(params[los]) )
                grad[i] += z
               
        return grad


    ''' no computation of the hessian for now, use methods that doesn't require it
        def hessian(self, params):
        hess = 2 * self._penalty * np.identity(len(params))
        for win, los in self._data:
            z = _safe_exp(params[win] - params[los])
            val =  1 / (1/z + 2 + z)
            hess[(win,los),(los,win)] += -val
            hess[(win,los),(win,los)] += +val
        return hess
    '''


    def FactorBT_objective(self, params, rev_params_g, rev_params_r):
        """Compute the negative penalized log-likelihood, for the item scores and the reviewers' parameters"""
        val = self._penalty * np.sum(params**2)
        for i in self._data:
            for win, los in self._data[i]:
                
                if self._classes[win]>self._classes[los]:
                    temp = 1
                elif self._classes[win]<self._classes[los]:
                    temp = -1
                else:
                    temp = 0
                
                val += -np.log( rev_params_g[i] * sigmoid(params[win] - params[los]) + (1 - rev_params_g[i]) *  sigmoid(rev_params_r[i] * temp) )     
        return val

    def FactorBT_gradient_scores(self, params, rev_params_g, rev_params_r):
        """Compute the gradient of the negative log-likelihood wrt the item scores"""
        grad = 2 * self._penalty * params
        for i in self._data:
            for win, los in self._data[i]:
                
                if self._classes[win]>self._classes[los]:
                    temp = 1
                elif self._classes[win]<self._classes[los]:
                    temp = -1
                else:
                    temp = 0
                    
                z = rev_params_g[i] * sigmoid(params[win] - params[los]) * sigmoid(params[los] - params[win]) / (rev_params_g[i] * sigmoid(params[win] - params[los]) + (1 - rev_params_g[i]) *  sigmoid(rev_params_r[i] * temp))                                                           

                grad[win] += -z
                grad[los] += z
        return grad
    
    def FactorBT_gradient_g(self, params, rev_params_g, rev_params_r):
        """Compute the gradient of the negative log-likelihood wrt the reviewers' parameters"""
        grad = np.zeros(len(rev_params_g))
        for i in self._data:
            for win, los in self._data[i]:
                                                                                                
                if self._classes[win]>self._classes[los]:
                    temp = 1
                elif self._classes[win]<self._classes[los]:
                    temp = -1
                else:
                    temp = 0
                
                z = (sigmoid(params[win] - params[los]) - sigmoid(rev_params_r[i] * temp) ) *  sigmoid(-rev_params_g[i])/(rev_params_g[i] * sigmoid(params[win] - params[los]) + (1 - rev_params_g[i]) *  sigmoid(rev_params_r[i] * temp))                                        
                                                                                                
                grad[i] += -z                                                                                
                                                                                                
                
        return grad

    def FactorBT_gradient_r(self, params, rev_params_g, rev_params_r):
        """Compute the gradient of the negative log-likelihood wrt the reviewers' parameters"""
        grad = np.zeros(len(rev_params_r))
        for i in self._data:
            for win, los in self._data[i]:

                if self._classes[win]>self._classes[los]:
                    temp = 1
                elif self._classes[win]<self._classes[los]:
                    temp = -1
                else:
                    temp = 0
                                                                                                
                z = (1 - rev_params_g[i]) *  sigmoid(rev_params_r[i] * temp) *  sigmoid(-rev_params_r[i] * temp) * temp /(rev_params_g[i] * sigmoid(params[win] - params[los]) + (1 - rev_params_g[i]) *  sigmoid(rev_params_r[i] * temp))  

                grad[i] += -z 
                     
        return grad
                                                                                                
def _sample_pairs(scores, n_pairs ):
    pairs = []
    numbers = np.arange(len(scores))
    for i in range(n_pairs):
        
        a, b = np.random.choice(numbers, size=2, replace=False) #replace = False to ensure a != b
        #while (a, b) in pairs or (b, a) in pairs: #sample the pair, in this version the reviewer can evaluate the same pair only once
            #a, b = np.random.choice(numbers, size=2, replace=False)

        #make them play
        if np.random.rand() < (np.exp(scores[a])/(np.exp(scores[a])+np.exp(scores[b]))):
        #if scores[a]>scores[b]: #deterministic version
        # this block of code will be executed with probability p
            pairs.append((a, b)) #who win is the first of the pair!!! i.e. a won
        else:
        # this block of code will be executed with probability 1-p   
            pairs.append((b, a)) #b won
    return(pairs)

def _create_matrix_biased_scores(original,rev_bias,classes):
    '''this matrix represents how much bias each reviewer has
    original: the original scores for the items
    rev_bias: the vector with the reviewers' biases
    classes: the items' classes 
    return:
    biases_scores: the matrix with the scores as 'seen' by each reviewer'''
    #matrix of biased scores, each reviewer correspond to a column
    biased_scores = np.zeros((len(original),len(rev_bias)))
    for col,bias in enumerate(rev_bias):
        for row,value in enumerate(classes):
            if value == 1:
                biased_scores[row,col] = original[row] + bias #add bias to reviewers ranking
                                                                                    
            elif value == 0:
                biased_scores[row,col] = original[row] 
    #biased_scores[biased_scores <= 0] = 0.00001
    return biased_scores

def _create_pc_set_for_reviewers(biased_scores,pair_per_reviewer):
    revs_set = {}
    for i in range(np.shape(biased_scores)[1]):
        revs_set.update({i:_sample_pairs(biased_scores[:,i], n_pairs = pair_per_reviewer )})
        
    return revs_set

def _pc_without_reviewers(revs_set):
    ''' input: the set of pc for each reviewer
        output: pc without the reviewer info'''
    return [[val1, val2] for sublist in revs_set.values() for val1, val2 in sublist]

def _alternate_optim(size, num_reviewers, pc_with_revs, iters = 101, tol = 1e-5, gtol = 1e-5):
    '''x0 is the estimated scores
       y0 is the estimated bias for each reviewer'''
    x0 = np.zeros(size)
    y0 = np.zeros(num_reviewers)


    for i in range(iters):

        # minimize with x fixed and update y
        res_y = minimize(lambda y: pc_with_revs.objective(x0, y), y0,tol = tol, jac=lambda y: pc_with_revs.gradient_revs(x0, y), options={"gtol": gtol,'maxiter': 1})
        y0 = res_y.x

        # minimize with y fixed and update x
        res_x = minimize(lambda x: pc_with_revs.objective(x, y0), x0,tol = tol, jac=lambda x: pc_with_revs.gradient_scores(x, y0), options={"gtol": gtol,'maxiter': 1})
        x0 = res_x.x


        #if ((i) % 100 == 0):
            #print(f"Iteration {i}: x = {x0}, y = {y0}")
            #print(res_x.success)
            #print(res_y.success)

        if res_x.success and res_y.success:
            #print("Minimum found!")
            #print(f"Iteration {i}: x = {x0}, y = {y0}")
            #print(res_x.success)
            #print(res_y.success)
            break
    return x0,y0



######### Rank Centrality implementation
def _matrix_of_comparisons(size,l, reg = 1):
    ''' Input: 
    size = the number of items
    l = the list of pairwise comparisons (a list of pairs, with the first of the pair is the one that has been preferred)
        Output:
    a (size x size) matrix, where a_ij represents the fraction of times object j has been preferred to
    object i'''
    
    A = np.zeros((size,size))
    for i,j in l:
        A[j,i] += 1 #i won
    
    B = np.zeros((size,size)) 
    for i in range(size):
        for j in range(size):
            if A[i,j]!=0:
                B[i,j] = A[i,j]/(A[i,j] + A[j,i])
                
    return B + reg * (np.ones((size,size)) - np.eye(size))
    
    
    

def _trans_prob(A):
    '''This function takes the matrix of comparisons, rescale by the maximum outdegree and add self loops'''
    n = np.shape(A)[0]
    d_max = np.max(np.count_nonzero(A,axis = 1)) #maximunm out degree
    P = A/d_max #rescale values 
    sum_by_row = np.sum(P,axis=1)
    for i in np.arange(0,n):
        P[i,i] = 1 - sum_by_row[i]
    return P

def _la(P,left=True,right=False):
    return linalg.eig(P,left=True,right=False)

def _stationary_dist(P):
    val, vec = _la(P)
    largest_eigenvector = vec[:, np.argmax(val)]
    #w_estim = largest_eigenvector/sum(largest_eigenvector)
    return largest_eigenvector

  