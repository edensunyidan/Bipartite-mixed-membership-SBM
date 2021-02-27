#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import csv

import copy

import numpy as np
from scipy.special import gammaln, digamma, polygamma, softmax
from numpy.linalg import inv

#from multiprocessing import Pool, Process, current_process, sharedctypes
#import sklearn
#import pymc3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#random_seed = 0
#np.random.seed(seed=random_seed)
#MAXINT = sys.maxsize

SEED = 0
LAG = 10

GAMMA_CONST = -1*digamma(1)

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny
EPS_INV = 1e-10

TOL = 1e-05

DRS_MAX_ITER = 1000
DRS_CONVERGED = TOL

DA_MAX_ITER = 1000
DA_CONVERGED = TOL

NEWTON_MAX_ITER = 1000
NEWTON_THRESH = TOL

#PHI_VAR_CONVERGED = 1e-09
PHI_VAR_CONVERGED = 1e-100
VAR_CONVERGED = 1e-07
#VAR_MAX_ITER = -1
VAR_MAX_ITER = 1
#VAR_MAX_ITER = 10
#VAR_MAX_ITER = 100

EM_CONVERGED = 1e-05
#EM_MAX_ITER = 10
#EM_MAX_ITER = 50
EM_MAX_ITER = 100
#EM_MAX_ITER = 1000

class MixedBiSBM:
    """Fit the Mixed membership bipartite SBM. The base model without noisy nodes.
    Input
    -----  
    K: tuple or list
        Number of mixture clusters on each side
        
    poi_model: bool, defaults to True
    
    deg_cor: bool, defaults to False
        Degree corrected version of algorithm  
        1. deg_cor=True  --> poi_model=True
        2. deg_cor=False --> poi_model=True/False
        
    est_alpha: bool, defaults to False
        If set to False, then alpha does not change from iteration to iteration
        If set to True,  then alpha is estimated along with the variational distributions """

    def __init__(self, B, K, poi_model=True, deg_cor=False, est_alpha=True, sym_dir=True, alpha_1=None, alpha_2=None, constr_case=1):

        self.B = B
        self.N1, self.N2 = B.shape[0], B.shape[1]  
        self.K1, self.K2 = K[0], K[1]

        self.poi_model = poi_model
        self.deg_cor = deg_cor 
        self.est_alpha = est_alpha
        self.sym_dir = sym_dir
        self.alpha_1 = copy.deepcopy(alpha_1)
        self.alpha_2 = copy.deepcopy(alpha_2)        
        self.constr_case = constr_case

        if self.poi_model == True:
            self.gammaln_B = (gammaln(self.B+1))[:,:,np.newaxis,np.newaxis]       
        
        if (self.deg_cor == True) and (self.poi_model == False):
            sys.exit("If deg_cor = True, then poi_model = True!")
        
        if alpha_1 is None:
            self.alpha_1 = np.array([50/self.K1] * self.K1, dtype=np.float64)
        else:
            self.alpha_1 = copy.deepcopy(alpha_1)

        if alpha_2 is None:
            self.alpha_2 = np.array([50/self.K2] * self.K2, dtype=np.float64)
        else:
            self.alpha_2 = copy.deepcopy(alpha_2)

        if (self.est_alpha == False) and ((self.alpha_1 is None) or (self.alpha_2 is None)):
            sys.exit("input alpha_1 and alpha_2 required!")
   

    def init_var_params(self, init_phi_1, init_phi_2):
        """Initialize variational parameters in the beginning of variational EM 
        self.phi_1: (N1, N2, K1)
        self.phi_2: (N1, N2, K2)
        
        self.gamma_1: (N1, K1)
        self.gamma_2: (N2, K2) """

        #smooth to avoid log(0)
        if (init_phi_1 < TINY).any():
            #init_phi_1[np.nonzero(phi_1_copy < EPS)] = EPS
            init_phi_1 = init_phi_1 + EPS
            init_phi_1 = init_phi_1/((init_phi_1.sum(axis=2))[:,:,np.newaxis]) 

        self.phi_1_init = copy.deepcopy(init_phi_1)
        self.phi_1 = copy.deepcopy(init_phi_1)
        
        if (init_phi_2 < TINY).any():
            #init_phi_2[np.nonzero(phi_2_copy < EPS)] = EPS
            init_phi_2 = init_phi_2 + EPS
            init_phi_2 = init_phi_2/((init_phi_2.sum(axis=2))[:,:,np.newaxis]) 

        self.phi_2_init = copy.deepcopy(init_phi_2)
        self.phi_2 = copy.deepcopy(init_phi_2)

        self.gamma_1 = self.update_gamma(self.phi_1, self.alpha_1, 1)
        self.gamma_2 = self.update_gamma(self.phi_2, self.alpha_2, 0) 


    def _init_var_params(self):
        """Initialize variational parameters in the beginning of each E-step 
        self.phi_1: (N1, N2, K1)
        self.phi_2: (N1, N2, K2)
        
        self.gamma_1: (N1, K1)
        self.gamma_2: (N2, K2) """
        
        self.phi_1 = copy.deepcopy(self.phi_1_init)
        self.phi_2 = copy.deepcopy(self.phi_2_init)
            
        self.gamma_1 = self.update_gamma(self.phi_1, self.alpha_1, 1) 
        self.gamma_2 = self.update_gamma(self.phi_2, self.alpha_2, 0)   

                       
    def init_lkh_params(self, init_omega=None):
        """Initialize likelihood parameters
        self.omega: (K1, K2)
        self.theta_1: (N1,)
        self.theta_2: (N2,)
        
        init_omega: defaults to None; otherwise, (K1, K2) array
        """   
        self.theta_1 = np.ones(self.N1, dtype=np.float64)
        self.theta_2 = np.ones(self.N2, dtype=np.float64) 
        
        if init_omega is None:
            #self.omega = self.update_omega(self.phi_1, self.phi_2, self.theta_1, self.theta_2)
            self.omega = np.ones((self.K1, self.K2), dtype=np.float64)
        else:
            self.omega = copy.deepcopy(init_omega)
    
                        
    def log_edge_pmf(self, omega, theta_1, theta_2):
        """Calculate the log likelihood of B, given the parameter theta_1*theta_2*omega. 
        Natural logarithm.

        Probability Mass Function:
        Poi: exp(-mu) * mu**k / k!  -->  k*np.log(mu) - mu - gammaln(k+1)
        Ber: p**k * (1-p)**(1-k)    -->  k*np.log(p) + (1-k)*np.log(1-p)

        Input
        -----
        self.B:     (N1, N2)
        self.omega: (K1, K2)

        self.theta_1: (N1,)
        self.theta_2: (N2,)

        Output
        ------
        log_pmf_edge: (N1, N2, K1, K2) """

        poi_mean = self.theta_1[:,np.newaxis,np.newaxis,np.newaxis]*self.theta_2[np.newaxis,:,np.newaxis,np.newaxis]*self.omega[np.newaxis,np.newaxis,:,:]
        # poi_mean[np.nonzero(poi_mean < TINY)] = TINY

        if self.poi_model == True:
            log_pmf_edge = self.B[:,:,np.newaxis,np.newaxis]*np.log(poi_mean) - poi_mean - self.gammaln_B
        else:
            log_pmf_edge = self.B[:,:,np.newaxis,np.newaxis]*np.log(poi_mean) + (1-self.B[:,:,np.newaxis,np.newaxis])*np.log(1-poi_mean)
        
        return log_pmf_edge


    def update_phi(self, phi, gamma, omega, theta_1, theta_2, side, phi_constr=False): 
        """Update phi, by side
        if side=1, update phi_1 based on phi_2
        else,      update phi_2 based on phi_1 
        
        phi_constr: bool, default = False
            If True, then take the constraint of theta also as the constraint of phi
            If self.deg_cor==False, then phi_constr must be False
            
        Input
        -----
        side: 1/2
        phi: (N1, N2, K2) / (N1, N2, K1)
        gamma: (N1, K1) / (N2, K2)
        omega: (K1, K2)
        theta_1: (N1,)
        theta_2: (N2,)
        
        Output
        ------
        phi_updated: (N1, N2, K1) / (N1, N2, K2) """ 
        
        if (self.deg_cor == False) and (phi_constr == True):
            sys.exit("update phi without constraint if deg_cor = False!")
        
        if (self.constr_case == 1) and (phi_constr == True):
            sys.exit("update theta with gamma constraint!")
        
        if side == 1:
            pmf = self.log_edge_pmf(omega, theta_1, theta_2)   
            ss_gamma_1 = self.dir_expc_suff_stats(gamma)
            exponent = np.sum(phi[:,:,np.newaxis,:] * pmf, axis=3) + ss_gamma_1[:,np.newaxis,:]
            if phi_constr == False:
                phi_updated = softmax(exponent, axis=2) 
            else:
                phi_updated = self.update_phi_dual_ascent(exponent, theta_1[:,np.newaxis,np.newaxis])    
        else:
            pmf = self.log_edge_pmf(omega, theta_1, theta_2)
            ss_gamma_2 = self.dir_expc_suff_stats(gamma)
            exponent = np.sum(phi[:,:,:,np.newaxis] * pmf, axis=2) + ss_gamma_2[np.newaxis,:,:]
            if phi_constr == False:
                phi_updated = softmax(exponent, axis=2) 
            else:
                phi_updated = self.update_phi_dual_ascent(exponent, theta_2[np.newaxis,:,np.newaxis])
            
        return phi_updated


    def update_phi_dual_ascent(self, a, theta, mu=0.01, damping=0.95):
        """Update phi by dual ascent in the degree-corrected version.
        
        Input
        -----
        a: (N1, N2, K) 
        theta: (N1, 1, 1) or (1, N2, 1). shape matched with ``a" for broadcasting
        
        Output
        ------
        phi: (N1, N2, K) """
        
        N1, N2, K = a.shape
        y = np.zeros(K)
        tht = theta - 1 
        
        da_iter = 0
        da_converged = 1
        while (da_converged > DA_CONVERGED) and (da_iter < DA_MAX_ITER):
            
            y_old = y
            exponent = a - y[np.newaxis, np.newaxis, :] * tht     
            phi = softmax(exponent, axis=2) 
            y = y + mu * np.sum(tht * phi, axis=(0,1))
            mu = mu * damping

            da_converged = self.err(y, y_old)
            da_iter = da_iter + 1
            
        return phi

        
    def update_gamma(self, phi, alpha, axis):
        """
        Input
        -----
        phi: (N1, N2, K1) / (N1, N2, K2)
        alpha: (K1,) / (K2,)
        axis: 1/0
        
        Output
        ------
        gamma_updated: (N1, K1) / (N2, K2) """
        
        gamma_updated = np.sum(phi, axis=axis) + alpha[np.newaxis, :]
        return gamma_updated


    def update_theta(self, theta, omega, phi_1, phi_2, gamma=None, side=1, sub_case=1):
        """Update theta. Douglas-Rachford (DR) splitting algorithm.

        side=1, update theta_1 based on theta_2
        else,   update theta_2 based on theta_1   
        
        constr_case 1: 
            sub_case 1: gamma_bar.T * theta = gamma_bar.T * 1
            sub_case 2: gamma_bar.T * theta = 1

        constr_case 2:
            sub_case 1: varphi.T * theta = varphi.T * 1
            sub_case 2: varphi.T * theta = 1            
                
        Input
        -----
        side: 1/2
        theta: (N2,) / (N1,)
        phi_1: (N1, N2, K1)
        phi_2: (N1, N2, K2)
        gamma: (N1, K1) / (N2, K2) if constr_case == 0; None if constr_case == 1
        
        Output
        ------
        theta_updated: (N1,) / (N2,) """     
        
        if side==1:
            a = np.sum(phi_1[:,:,:,np.newaxis] * phi_2[:,:,np.newaxis,:] * self.B[:,:,np.newaxis,np.newaxis], axis=(1,2,3))
            b = np.sum(phi_1[:,:,:,np.newaxis] * phi_2[:,:,np.newaxis,:] * theta[np.newaxis,:,np.newaxis,np.newaxis] * omega[np.newaxis,np.newaxis,:,:], axis=(1,2,3))
            if self.constr_case == 1:
                wei = gamma / gamma.sum(axis=1)[:, np.newaxis] 
            else: 
                wei = phi_1.sum(axis=1)

        else:
            a = np.sum(phi_1[:,:,:,np.newaxis] * phi_2[:,:,np.newaxis,:] * self.B[:,:,np.newaxis,np.newaxis], axis=(0,2,3))
            b = np.sum(phi_1[:,:,:,np.newaxis] * phi_2[:,:,np.newaxis,:] * theta[:,np.newaxis,np.newaxis,np.newaxis] * omega[np.newaxis,np.newaxis,:,:], axis=(0,2,3))
            if self.constr_case == 1:
                wei = gamma / gamma.sum(axis=1)[:, np.newaxis]         
            else: 
                wei = phi_2.sum(axis=0)
        
        theta_updated = self.update_theta_drsplit(wei, a, b, sub_case)
        return theta_updated
            

    def update_theta_drsplit(self, wei, a, b, sub_case):
        """
        Input
        -----
        side: 1/2
        wei: (N1, K1) / (N2, K2)
        a: (N1,) / (N2,)
        b: (N1,) / (N2,)
        
        Output
        ------
        theta: (N1,) / (N2,) """

        N, K = wei.shape

        #WEI = np.dot(wei.T, wei) + EPS*np.eye(K)
        WEI = np.dot(wei.T, wei) + EPS_INV*np.eye(K)

        
        theta = np.zeros(N) #
        xi = np.zeros(N)
        u = np.zeros(N)
        
        drs_iter = 0
        drs_converged = 1
        while (drs_converged > DRS_CONVERGED) and (drs_iter < DRS_MAX_ITER):
        
            theta_old = theta

            theta = self.prox_theta(a, b, xi-u)
            if sub_case == 1:
                xi = theta + u - np.dot((np.dot(wei, inv(WEI))),
                                        (np.dot(wei.T, theta + u) - wei.T.sum(axis=1)))
            else:
                xi = theta + u - np.dot((np.dot(wei,inv(WEI))),
                                        (np.dot(wei.T, theta + u) - 1))  
            u = u + theta - xi
            
            drs_converged = self.err(theta, theta_old)
            drs_iter = drs_iter + 1
            
        return theta
    

    def err(self, x, xo, norm_type=None):  
        """
        norm_type | norm for vectors
        ----------------------------
        np.inf    | max(abs(x))
        None      | 2-norm 
        """
        x = x.flatten(order='C')
        xo = xo.flatten(order='C')

        out = np.linalg.norm(x - xo, ord=norm_type)/np.amax([1, np.linalg.norm(xo, ord=norm_type)]) 
        return out


    def norm(self, mat, norm_type=np.inf):
        """
        norm_type | norm for vectors
        ----------------------------
        np.inf    | max(abs(x)), np.amax(np.absolute())
        None      | 2-norm 
        """
        mat = np.ravel(mat)
        out = np.linalg.norm(mat, ord=norm_type)
        return out


    def prox_theta(self, a, b, x, t=1):
        """Proximal operator.
        a, b: parameter vector
        x: variable vector
        t: parameter, default = 1 """

        return 0.5*(-(b*t - x) + np.sqrt((b*t - x)**2 + 4*a*t))   


    def update_omega(self, phi_1, phi_2, theta_1, theta_2):

        numerator = (phi_1[:,:,:,np.newaxis] * phi_2[:,:,np.newaxis,:] * self.B[:,:,np.newaxis,np.newaxis]).sum(axis=(0,1))
        denominator = (phi_1[:,:,:,np.newaxis] * phi_2[:,:,np.newaxis,:] * theta_1[:,np.newaxis,np.newaxis,np.newaxis] * theta_2[np.newaxis,:,np.newaxis,np.newaxis]).sum(axis=(0,1)) 
        # denominator[np.nonzero(denominator < TINY)] = TINY

        omega_updated = numerator / denominator

        return omega_updated


    def init_alpha_newton(self, gamma, method, side):
        """Initialize alpha for Newton's method
        Input
        -----
        gamma: (N, K) shape array. K>=2. N>K.
            variational parameter. pi_i ~ Dir(gamma_i)

        method: 
            Alpha: self.alpha
            Blei: 
            Ronning: global minimum
            Dishon: method of moments
            Wicker: method of moments

        Returns
        -------
        alpha: (K,) shape array """

        N, K = gamma.shape

        if method == 'Alpha':
            if side==1:
                alpha_init = copy.deepcopy(self.alpha_1)
            else: #side==2
                alpha_init = copy.deepcopy(self.alpha_2)


        if method == 'Blei':
            alpha_init = np.repeat(10, K) 
            

        if method == 'Ronning':
            mean_pi = gamma / gamma.sum(axis=1)[:, np.newaxis] #first moment / mean, expectation of pi_i
            min_mean_pi = np.amin(mean_pi) 
            alpha_init = np.repeat(max(min_mean_pi, EPS), K) 


        if method == 'Dishon':
            mean_pi = gamma / gamma.sum(axis=1)[:, np.newaxis] #first moment / mean, expectation of pi_i
            mean_pi_mean = mean_pi.mean(axis=0)

            alpha_plus_one = (gamma.sum(axis=1) + 1)
            log_alpha_plus_one = (1/N) * np.sum(np.log(alpha_plus_one))

            alpha_plus = np.exp(log_alpha_plus_one) - 1
            alpha_init = alpha_plus * mean_pi_mean


        if method == 'Wicker':
            mean_pi = gamma / gamma.sum(axis=1)[:, np.newaxis] #first moment / mean, expectation of pi_i
            mean_pi_mean = mean_pi.mean(axis=0)

            mean_pi_sumlog = np.sum(np.log(mean_pi), axis=0)
            numerator = N*(K-1)*GAMMA_CONST
            #numerator = N*(K-1)*0.5  
            denominator = N * (mean_pi_mean * np.log(mean_pi_mean)).sum() - (mean_pi_mean * mean_pi_sumlog).sum()

            alpha_plus = numerator / denominator        
            alpha_init = alpha_plus * mean_pi_mean         

        return alpha_init    
    
    
    def update_alpha(self, gamma, init_method='Blei', side=1):
        """Iterative Newton-Ralphson method
        alpha_new = alpha_old - H^-1 * g

        Input
        -----
        gamma: (N, K) shape array 
        init_method: 
            1. Alpha: self.alpha
            2. Blei: alpha_init = 10
            3. Ronning: 
            4. Dishon: 
            5. Wicker:

        Returns
        -------
        alpha: (K,) shape array 

        Note
        ----
        alpha_init =/= initial of alpha in the algorithm beginning """

        init_method_list = ['Alpha', 'Blei', 'Ronning', 'Dishon', 'Wicker']

        if init_method in init_method_list:
            pass
        else:
            sys.exit("init_method of update_alpha() is not in" + str(init_method_list) + "!")

        if (self.sym_dir == True) and (init_method != 'Blei'):
            sys.exit("sym_dir == True, but init_method != 'Blei' !")

        alpha_init = self.init_alpha_newton(gamma, method=init_method, side=side)   
        ss_gamma = self.dir_expc_suff_stats(gamma)
        N, K = gamma.shape

        if self.sym_dir == True:
            opt_alpha = self.update_alpha_sym(alpha_init, ss_gamma, N, K)
        else:
            opt_alpha = self.update_alpha_asym(alpha_init, ss_gamma, N, K)

        return opt_alpha
    
    
    def d_alhood(self, a, ss, N, K):
        result = N * (K * digamma(K * a) - K * digamma(a)) + np.sum(ss)
        return result


    def d2_alhood(self, a, N, K):
        result = N * (K * K * polygamma(1, K * a) - K * polygamma(1, a))
        return result


    def update_alpha_sym(self, alpha_init, ss, N, K):
        """
        symmetric Dirichlet: alpha = np.array([a] * K)
        update log(a) instead of a: f(a) = f(exp(log(a)))
        """

        init_a = alpha_init[0]
        log_a = np.log(init_a)
        a = np.exp(log_a)

        diff = 1
        df = 1
        newton_iter = 0

        while (diff > TINY) and (abs(df) > NEWTON_THRESH) and (newton_iter < NEWTON_MAX_ITER): #converged < 0 
            #print("newton_iter: ", newton_iter)
            #print("a: ", a)

            newton_iter = newton_iter + 1
            a = np.exp(log_a)
            a_old = a

            if np.isnan(a):
                init_a = init_a * 10
                print("warning :  alpha is nan; new init = %f \n" % init_a)
                a = init_a
                log_a = np.log(a)

            df = self.d_alhood(a, ss, N, K)
            d2f = self.d2_alhood(a, N, K)
            log_a = log_a - df/(d2f * a + df)
            a = np.exp(log_a)
            diff = abs(a - a_old)

        opt_a = np.exp(log_a)
        opt_alpha = np.array([opt_a]*K)
        return opt_alpha
    

    def update_alpha_asym(self, alpha_init, ss, N, K):

        alpha = alpha_init

        diff = 1
        g = np.ones(K)
        newton_iter = 0    
        while (diff > TINY) and (max(abs(g)) > NEWTON_THRESH) and (newton_iter < NEWTON_MAX_ITER): #converged < 0      
            newton_iter = newton_iter + 1
            alpha_old = alpha
            print("newton_iter: ", newton_iter)
            print("alpha: ", alpha)

            g = - N * self.dir_expc_suff_stats(alpha) + np.sum(ss, axis=0)
            h_inv_g = self.hessian_inv_gradient(N, alpha, g)
            #h_inv_g = solve(N, alpha, g)
            alpha = alpha - h_inv_g
            alpha[np.nonzero(alpha < EPS)[0]] = EPS
            diff = max(abs(alpha - alpha_old))

        opt_alpha = alpha
        return opt_alpha
    
    
    def hessian_inv_gradient(self, N, alpha, g):
        """Calculate H^{-1} * g
        Input
        -----
        alpha: (K,) shape array. 
        g: (K,) shape array. gradient 
        """
        q = - N * polygamma(1, alpha) 
        z = N * polygamma(1, np.sum(alpha))
        b = np.sum(g/q)/(1/z + np.sum(1/q))
        h_inv_g = (g - b)/q 
        return h_inv_g


    def dir_expc_suff_stats(self, alpha):
        """
        For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.

        Input
        -----
        alpha: (K,) / (N, K) [gamma]

        Output
        ------
        ss: (K,) / (N, K)
        """
        if (len(alpha.shape) == 1):
            ss = digamma(alpha) - digamma(np.sum(alpha))
        else:
            ss = digamma(alpha) - digamma(np.sum(alpha, 1))[:, np.newaxis]

        #return ss.astype(alpha.dtype) # keep the same precision as input
        return ss
    
    
    def expc_logpdf_dir(self, alpha, ss):
        """Expectation of log of the dirichlet density function, w.r.t variational distribution

        Input
        -----
        alpha: 
            (N, K) shape array [gamma]
            (K,) shape array [alpha]
            scalar if self.sym_dir == True [alpha]

        ss: (N, K) shape array; 
            .. math:: psi(gamma) - psi(gamma.sum(axis=1))[:, np.newaxis] """

        if np.ndim(alpha) == 2: 
            out = np.sum(gammaln(np.sum(alpha, 1)) - np.sum(gammaln(alpha), 1)) + np.sum((alpha-1) * ss)

        elif np.ndim(alpha) == 1: 
            N, K = ss.shape
            out = N * (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))) + np.sum((alpha-1)[np.newaxis, :] * ss)

        else: #np.ndim(alpha) == 0: #np.isscalar()
            N, K = ss.shape
            out = N * (gammaln(K * alpha) - K * gammaln(alpha)) + np.sum((alpha-1) * ss)

        return out


    def calculate_lower_bound(self):
        """Calculate the variational lower bound
        """
        
        pmf = self.log_edge_pmf(self.omega, self.theta_1, self.theta_2) 
        
        ss_gamma_1 = self.dir_expc_suff_stats(self.gamma_1)
        ss_gamma_2 = self.dir_expc_suff_stats(self.gamma_2)

        #phi_1_copy = copy.deepcopy(self.phi_1)
        #phi_2_copy = copy.deepcopy(self.phi_2)

        #phi_1_copy[np.nonzero(phi_1_copy < TINY)] = TINY
        #phi_2_copy[np.nonzero(phi_2_copy < TINY)] = TINY

        obj_func = (np.sum(self.phi_1[:,:,:,np.newaxis] * self.phi_2[:,:,np.newaxis,:] * pmf) 
                    + np.sum(self.phi_1 * ss_gamma_1[:,np.newaxis,:]) + np.sum(self.phi_2 * ss_gamma_2[np.newaxis,:,:])
                    + self.expc_logpdf_dir(self.alpha_1, ss_gamma_1) + self.expc_logpdf_dir(self.alpha_2, ss_gamma_2)
                    - self.expc_logpdf_dir(self.gamma_1, ss_gamma_1) - self.expc_logpdf_dir(self.gamma_2, ss_gamma_2)
                    - np.sum(self.phi_1 * np.log(self.phi_1)) - np.sum(self.phi_2 * np.log(self.phi_2)))

        return obj_func  


    def run_e_step_naive(self, init_repeat=True, var_elbo=True, phi_constr=False):
        """Update the variational parameters
        """
        if init_repeat == True:
            self._init_var_params()
                
        if var_elbo==True:
            likelihood_old = self.calculate_lower_bound()
            converged = 1
            var_iter = 0
            while (converged > VAR_CONVERGED) and ((var_iter < VAR_MAX_ITER) or (VAR_MAX_ITER == -1)):  #converged < 0
                print("E-step: ", var_iter + 1)

                self.phi_1 = self.update_phi(self.phi_2, self.gamma_1, self.omega, self.theta_1, self.theta_2, side=1, phi_constr=phi_constr)
                self.phi_2 = self.update_phi(self.phi_1, self.gamma_2, self.omega, self.theta_1, self.theta_2, side=2, phi_constr=phi_constr)

                self.gamma_1 = self.update_gamma(self.phi_1, self.alpha_1, 1)
                self.gamma_2 = self.update_gamma(self.phi_2, self.alpha_2, 0)
                
                #self.omega = self.update_omega(self.phi_1, self.phi_2, self.theta_1, self.theta_2)

                #if self.est_alpha == True:
                    #self.alpha_1 = self.update_alpha(gamma=self.gamma_1, init_method='Blei', side=1)
                    #self.alpha_2 = self.update_alpha(gamma=self.gamma_2, init_method='Blei', side=2)

                likelihood = self.calculate_lower_bound()
                converged = (likelihood_old - likelihood) / likelihood_old
                likelihood_old = likelihood 
                var_iter = var_iter + 1
        
        else:
            converged = 1
            var_iter = 0
            while (converged > VAR_CONVERGED) and ((var_iter < VAR_MAX_ITER) or (VAR_MAX_ITER == -1)): #converged < 0    
                print("E-step: ", var_iter + 1)

                phi_1_old = self.phi_1    
                phi_2_old = self.phi_2
                
                gamma_1_old = self.gamma_1
                gamma_2_old = self.gamma_2   

                self.phi_1 = self.update_phi(self.phi_2, self.gamma_1, self.omega, self.theta_1, self.theta_2, side=1, phi_constr=phi_constr)
                self.phi_2 = self.update_phi(self.phi_1, self.gamma_2, self.omega, self.theta_1, self.theta_2, side=2, phi_constr=phi_constr)

                self.gamma_1 = self.update_gamma(self.phi_1, self.alpha_1, 1)
                self.gamma_2 = self.update_gamma(self.phi_2, self.alpha_2, 0)

                #self.omega = self.update_omega(self.phi_1, self.phi_2, self.theta_1, self.theta_2)

                delta_1 = np.amax([self.norm(self.phi_1 - phi_1_old), self.norm(self.gamma_1 - gamma_1_old)])
                delta_2 = np.amax([self.norm(self.phi_2 - phi_2_old), self.norm(self.gamma_2 - gamma_2_old)])
                converged = np.amax([delta_1, delta_2])
                var_iter = var_iter + 1
    
    
    def run_m_step_naive(self):
        """Update the likelihood parameters
        """
        print("M-step")

        if self.est_alpha == True:
            self.alpha_1 = self.update_alpha(gamma=self.gamma_1, init_method='Blei', side=1)
            self.alpha_2 = self.update_alpha(gamma=self.gamma_2, init_method='Blei', side=2)
        
        self.omega = self.update_omega(self.phi_1, self.phi_2, self.theta_1, self.theta_2)
        
        if self.deg_cor:
            self.theta_1 = self.update_theta(self.theta_2, self.omega, self.phi_1, self.phi_2, self.gamma_1, side=1)     
            self.theta_2 = self.update_theta(self.theta_1, self.omega, self.phi_1, self.phi_2, self.gamma_2, side=2)


    def run_variational_em_naive(self, save_dir, init_repeat=True, var_elbo=True, phi_constr=False, plot_option=True):
        """Variational Bayes + EM:
            iteratively update the variational parameters and likelihood parameters until convergence.
        
        Input
        -----
        var_elbo: default to True
            If True,  then use elbo as the convergence score 
            If False, then use norm as the convergence score
        
        init_repeat: default to True
            Initialize the variational parameters {phi, gamma} at each interation (variational inference cycle)  
        
        plot_option: default to True
            Plotting the loglikelihoods at each step
        
        Output 
        ------
        Parameters of interest: omega, alpha, gamma, phi, theta
        lower_bound: the lower bound on the log-likelihood of the observed data 
        
        Note
        ----
        For clarity/simplicity, 
        self.parameter is taken as an input arguments in each updating function; 
        even if it can be used a global variable (class instance variable) without being explicitly input
        
        When theta = np.ones(K), it recovers the non-degree-corrected version.
        """

        #global VAR_MAX_ITER 
        
        np.random.seed(SEED)  
        
        lower_bound = []
        lower_bound.append(self.calculate_lower_bound())

        em_iter_max = 0
        lower_bound_max = lower_bound[-1]
        phi_1_max = copy.deepcopy(self.phi_1)
        phi_2_max = copy.deepcopy(self.phi_2)
        gamma_1_max = copy.deepcopy(self.gamma_1)
        gamma_2_max = copy.deepcopy(self.gamma_2)
        alpha_1_max = copy.deepcopy(self.alpha_1)
        alpha_2_max = copy.deepcopy(self.alpha_2)
        omega_max = copy.deepcopy(self.omega)
        theta_1_max = copy.deepcopy(self.theta_1)
        theta_2_max = copy.deepcopy(self.theta_2)
        
        em_iter = 0
        converged = 1
        while ((converged < 0) or (converged > EM_CONVERGED) or (em_iter <= 5)) and (em_iter < EM_MAX_ITER): 
            em_iter = em_iter + 1
            print("EM Iteration: ", em_iter)
            
            self.run_e_step_naive(init_repeat=init_repeat, var_elbo=var_elbo, phi_constr=phi_constr) 
            lower_bound.append(self.calculate_lower_bound())
            #print(self.omega)
            #print("converged: ", converged)
            self.run_m_step_naive()
            
            if lower_bound[-1] > lower_bound_max:
                
                em_iter_max = em_iter
                lower_bound_max = lower_bound[-1]
                phi_1_max = copy.deepcopy(self.phi_1)
                phi_2_max = copy.deepcopy(self.phi_2)
                gamma_1_max = copy.deepcopy(self.gamma_1)
                gamma_2_max = copy.deepcopy(self.gamma_2)
                alpha_1_max = copy.deepcopy(self.alpha_1)
                alpha_2_max = copy.deepcopy(self.alpha_2)
                omega_max = copy.deepcopy(self.omega)  
                theta_1_max = copy.deepcopy(self.theta_1)
                theta_2_max = copy.deepcopy(self.theta_2)              
        
            converged = (lower_bound[-2]-lower_bound[-1])/lower_bound[-2]
            
            # if (converged < 0) and (VAR_MAX_ITER != -1):
            #     VAR_MAX_ITER = VAR_MAX_ITER * 2
                
                
        if plot_option == True:
            plt.figure(0)
            plt.plot(np.arange(len(lower_bound)-1), lower_bound[1:], 'bo-') 
            plt.title('Lower Bound of Log-Likelihood in Variational EM')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Lower Bound')
            plt.savefig(save_dir + '/lower_bound_iter.pdf')
            

        np.savetxt(save_dir + '/em_iter.csv', np.array([em_iter]), delimiter = '\n')
        np.savetxt(save_dir + '/lower_bound.csv', np.array(lower_bound), delimiter = '\n')
        np.savetxt(save_dir + '/phi_1.csv', self.phi_1.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/phi_2.csv', self.phi_2.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/gamma_1.csv', self.gamma_1.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/gamma_2.csv', self.gamma_2.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/alpha_1.csv', self.alpha_1.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/alpha_2.csv', self.alpha_2.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/omega.csv', self.omega.flatten(order='C'), delimiter = '\n')  
        np.savetxt(save_dir + '/theta_1.csv', self.theta_1.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/theta_2.csv', self.theta_2.flatten(order='C'), delimiter = '\n')        

        np.savetxt(save_dir + '/em_iter_max.csv', np.array([em_iter_max]), delimiter = '\n')
        np.savetxt(save_dir + '/lower_bound_max.csv', np.array([lower_bound_max]), delimiter = '\n')
        np.savetxt(save_dir + '/phi_1_max.csv', phi_1_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/phi_2_max.csv', phi_2_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/gamma_1_max.csv', gamma_1_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/gamma_2_max.csv', gamma_2_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/alpha_1_max.csv', alpha_1_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/alpha_2_max.csv', alpha_2_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/omega_max.csv', omega_max.flatten(order='C'), delimiter = '\n') 
        np.savetxt(save_dir + '/theta_1_max.csv', theta_1_max.flatten(order='C'), delimiter = '\n')
        np.savetxt(save_dir + '/theta_2_max.csv', theta_2_max.flatten(order='C'), delimiter = '\n')    

        
        return 




