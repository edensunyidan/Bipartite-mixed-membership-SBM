#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
from scipy import stats

CONST_PI = pow(2*math.pi, -1/2)
SEED = 0


def generate_base_model(N, omega, alpha_1, alpha_2, 
                        poi_gen=False, deg_var=False, 
                        pareto=True, mix=False, a=3, sigma=0.5, ratio=0.5, m=1, norm=False):
    """Generate the simulated data.
    Input
    -----
    N: tuple or list
    omega: (K1, K2) array
    alpha_1: (K1,) array
    alpha_2: (K2,) array
    seed: int, optional, default=None
    poi_gen: bool, default=False
        If True, use Poisson edge generation distribution
    deg_var: bool, default=False
        If True, incorporate the degree correction parameter
    """ 
    
    np.random.seed(SEED)
    
    N1, N2 = N[0], N[1]
    #K1, K2 = K[0], K[1]
    
    pi_1 = np.random.dirichlet(alpha_1, size=N1) #(N1, K1)
    pi_2 = np.random.dirichlet(alpha_2, size=N2) #(N2, K2)
    z_12 = np.array([np.random.multinomial(n=1, pvals=pi_1[i,], size=N2) for i in range(N1)]) #(N1, N2, K1) 
    z_21 = np.array([np.random.multinomial(n=1, pvals=pi_2[j,], size=N1) for j in range(N2)]) #(N2, N1, K2)
    B = np.zeros((N1, N2), dtype=np.uint16)
    
    if deg_var == False:
        theta_1 = None
        theta_2 = None

        for i in range(N1):
            for j in range(N2):
                prob_cand = np.dot(np.dot(z_12[i][j], omega), z_21[j][i])
                if poi_gen == False:
                    prob = min((prob_cand, 1))
                    B[i,j] = np.random.binomial(n=1, p=prob, size=None) 
                else:
                    prob = prob_cand
                    B[i,j] = np.random.poisson(lam=prob, size=None)
                          
    else:
        theta_1, theta_2 = generate_theta(N=N, pareto=pareto, mix=mix, a=a, sigma=sigma, ratio=ratio, m=m, norm=norm)
        
        for i in range(N1):
            for j in range(N2):
                prob_cand = np.dot(np.dot(z_12[i][j], omega), z_21[j][i])*theta_1[i]*theta_2[j]
                if poi_gen == False:
                    prob = min((prob_cand, 1))
                    B[i,j] = np.random.binomial(n=1, p=prob, size=None) 
                else:
                    prob = prob_cand
                    B[i,j] = np.random.poisson(lam=prob, size=None)  
                    
    output = {'B':B, 'pi_1':pi_1, 'pi_2':pi_2, 'z_12':z_12, 'z_21':z_21, 'theta_1':theta_1, 'theta_2':theta_2}
    
    return output


def generate_theta(N, pareto, mix, a, sigma, ratio, m, norm):
    """
    1. Pareto: b*x_min^b / x^{b+1}
    2. mixture
    3. continuous
        
    Input
    -----
    pareto: bool, if ture, generate theta from Pareto distribution
    mix:    bool, if true, generate theta from mixture distribution
    a:      a>2; shape parameter of the Pareto distribution
    norm:   bool, if true, normalize theta_i so that average of theta equals to 1 """

    np.random.seed(SEED)
    N1, N2 = N[0], N[1]
    
    if pareto==True:
        x_m = (a-1)/a
        theta_1 = stats.pareto.rvs(b=a, loc=0, scale=x_m, size=N1, random_state=None)
        theta_2 = stats.pareto.rvs(b=a, loc=0, scale=x_m, size=N2, random_state=None)
        
    else:            
        if mix==True:
            theta_1 = mixture_distr(size=N1, ratio=ratio, m=m)
            theta_2 = mixture_distr(size=N2, ratio=ratio, m=m)
        else:
            theta_1 = continuous_distr(size=N1, sigma=sigma)
            theta_2 = continuous_distr(size=N2, sigma=sigma)

    if norm==True:
        factor_1 = sum(theta_1)/N1
        theta_1 = theta_1/factor_1
        
        factor_2 = sum(theta_2)/N2
        theta_2 = theta_2/factor_2
    
    return theta_1, theta_2

def continuous_distr(size, sigma=0.5):
    """
    Z ~ N(0, scale)
    theta = abs(Z) + 1 - (2*\pi)^{-1/2}
    
    Input
    -----
    sigma: standard deviation of the normal distribution """
    
    np.random.seed(SEED)
    z = np.random.normal(loc=0, scale=sigma, size=size)
    x = abs(z) + 1 - CONST_PI
    return x

def mixture_distr(size, ratio=0.5, m=1):
    """Create a mixture distribution for sampling, with a uniform distribution and a discrete distribution
    Input
    -----
    ratio: \in [0,1]
    m: \in 1:10 """
    
    np.random.seed(SEED)
    prob_vec = np.array([ratio, 1-ratio]) #prob_vec = prob_vec/prob_vec.sum()
    
    #create a discrete random variable class
    xk = np.array([2/(m+1), 2*m/(m+1)])
    pk = np.array([1/2, 1/2])
    discrete = stats.rv_discrete(name='discrete', values=(xk, pk))
    
    distr_list = [
        {"type": np.random.uniform, "kwargs": {"low": 0, "high": 2, "size": None}},
        {"type": discrete.rvs, "kwargs": {"size": None}}, ]
    
    distr_num = len(distr_list)
    data = np.zeros((size, distr_num))
    for idx, distr in enumerate(distr_list):
        data[:, idx] = distr["type"](size=(size,), **distr["kwargs"])
        
    random_idx = np.random.choice(a=np.arange(distr_num), size=(size,), replace=True, p=prob_vec)
    sample = data[np.arange(size), random_idx]

    return sample

