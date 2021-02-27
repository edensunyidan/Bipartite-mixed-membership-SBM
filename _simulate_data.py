#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import os
import time

import math
import numpy as np
from scipy import stats

from sklearn.cluster import spectral_clustering
from sim_funcs.simulate_funcs import generate_base_model
from class_object.fit_mixed_bi_sbm import MixedBiSBM


EPS = np.finfo(float).eps
PERTURB_RATIO = 0.10

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse the command-line arguments')
    #group = parser.add_mutually_exclusive_group()

    init_type_vec = ['const', 'sp', 'random-multinomial', 'random-dir', 'z', 'pi', 'perturb-z', 'perturb-pi']

    parser.add_argument('--deg_var',     type=int,   required=True, choices=[0, 1])
    parser.add_argument('--deg_cor',     type=int,   required=True, choices=[0, 1])
    parser.add_argument('--alpha',       type=float, required=True, choices=[0.05, 0.10, 0.25])
    parser.add_argument('--p',           type=float, required=True, choices=[0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--q',           type=float, required=True, choices=[0.1, 0.2, 0.3, 0.4, 0.5])

    parser.add_argument('--init_type',   type=str,   required=True, choices=init_type_vec)
    parser.add_argument('--save_dir',    type=str,   required=True)

    parser.add_argument('--est_alpha',   type=int,   default=1, choices=[0, 1])
    parser.add_argument('--sym_dir',     type=int,   default=1, choices=[0, 1])
    parser.add_argument('--init_repeat', type=int,   default=0, choices=[0, 1])
    parser.add_argument('--var_elbo',    type=int,   default=1, choices=[0, 1])
    parser.add_argument('--phi_constr',  type=int,   default=0, choices=[0, 1])

    #parser.print_help()
    args = parser.parse_args()

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    #N = [100, 200]
    N = [600, 800]
    K = [4, 4]
    
    omega = np.ones((K[0],K[0]),dtype=int)*args.q + np.eye(K[0],dtype=int)*(args.p-args.q)

    alpha_1 = np.array([args.alpha] * K[0], dtype=np.float64)  
    alpha_2 = np.array([args.alpha] * K[1], dtype=np.float64) 

    gen_data = generate_base_model(N=N, omega=omega, alpha_1=alpha_1, alpha_2=alpha_2, poi_gen=True, deg_var=args.deg_var, 
                                   pareto=True, mix=False, a=3, sigma=0.5, ratio=0.5, m=1, norm=False)
    
    B, pi_1, pi_2, z_12, z_21, theta_1, theta_2 = gen_data['B'], gen_data['pi_1'], gen_data['pi_2'], gen_data['z_12'], gen_data['z_21'], gen_data['theta_1'], gen_data['theta_2']

    np.savetxt(args.save_dir + '/true_B.csv', B.flatten(order='C'), delimiter = '\n')
    np.savetxt(args.save_dir + '/true_pi_1.csv', pi_1.flatten(order='C'), delimiter = '\n')
    np.savetxt(args.save_dir + '/true_pi_2.csv', pi_2.flatten(order='C'), delimiter = '\n')
    np.savetxt(args.save_dir + '/true_z_12.csv', z_12.flatten(order='C'), delimiter = '\n')
    np.savetxt(args.save_dir + '/true_z_21.csv', z_21.flatten(order='C'), delimiter = '\n')

    np.savetxt(args.save_dir + '/true_alpha_1.csv', alpha_1.flatten(order='C'), delimiter = '\n')
    np.savetxt(args.save_dir + '/true_alpha_2.csv', alpha_2.flatten(order='C'), delimiter = '\n')  
    np.savetxt(args.save_dir + '/true_omega.csv', omega.flatten(order='C'), delimiter = '\n')

    if args.deg_var == True:
        np.savetxt(args.save_dir + '/true_theta_1.csv', theta_1.flatten(order='C'), delimiter = '\n')
        np.savetxt(args.save_dir + '/true_theta_2.csv', theta_2.flatten(order='C'), delimiter = '\n')

    if args.est_alpha:
        mmb_object = MixedBiSBM(B=B, K=K, poi_model=True, deg_cor=args.deg_cor, est_alpha=args.est_alpha, sym_dir=args.sym_dir, alpha_1=None, alpha_2=None)
    else:
        mmb_object = MixedBiSBM(B=B, K=K, poi_model=True, deg_cor=args.deg_cor, est_alpha=args.est_alpha, sym_dir=args.sym_dir, alpha_1=alpha_1, alpha_2=alpha_2)   

    
    if args.init_type == 'const':
        phi_1_init = np.full((N[0], N[1], K[0]), 1/K[0], dtype=np.float64)
        phi_2_init = np.full((N[0], N[1], K[1]), 1/K[1], dtype=np.float64)


    if args.init_type == 'sp':
        if K[0] != K[1]:
            sys.exit("K[0] != K[1]!")
        A = np.concatenate((np.concatenate((np.zeros((N[0], N[0])), B), axis=1),
                            np.concatenate((B.T, np.zeros((N[1], N[1]))), axis=1)), axis=0)

        sp_label = spectral_clustering(affinity=A, n_clusters=K[0], n_components=None, eigen_solver=None, random_state=None, n_init=10, eigen_tol=0.0, assign_labels='kmeans')

        sp_label_1 = sp_label[: N[0]]
        sp_label_2 = sp_label[N[0]: ]

        idx_label = np.arange(K[0])

        sp_label_1 = (sp_label_1[:, np.newaxis] == idx_label[np.newaxis, :])*1
        sp_label_2 = (sp_label_2[:, np.newaxis] == idx_label[np.newaxis, :])*1 

        sp_label_1 = sp_label_1[:, np.newaxis, :] 
        sp_label_2 = sp_label_2[np.newaxis, :, :]

        init_phi_1 = np.repeat(sp_label_1.astype(np.float64), repeats=N[1], axis=1)
        init_phi_2 = np.repeat(sp_label_2.astype(np.float64), repeats=N[0], axis=0)      


    if args.init_type == 'z':
        z_12 = z_12.astype(np.float64) 
        z_21 = np.transpose(z_21.astype(np.float64), axes=[1,0,2]).copy() 

        init_phi_1 = z_12 
        init_phi_2 = z_21


    if args.init_type == 'pi':
        pi_1_ = pi_1[:, np.newaxis, :] 
        pi_2_ = pi_2[np.newaxis, :, :]

        init_phi_1 = np.repeat(pi_1_, repeats=N[1], axis=1)
        init_phi_2 = np.repeat(pi_2_, repeats=N[0], axis=0)


    if args.init_type == 'perturb-z':
        z_12 = z_12.astype(np.float64) 
        z_21 = np.transpose(z_21.astype(np.float64), axes=[1,0,2]).copy() 

        init_phi_truth_1 = z_12 
        init_phi_truth_2 = z_21

        dir_noise_1 = np.random.dirichlet(np.ones(K[0]) * (0.5), size=N[0]) #(N1, K1)
        dir_noise_2 = np.random.dirichlet(np.ones(K[1]) * (0.5), size=N[1]) #(N2, K2)

        dir_noise_1 = dir_noise_1[:, np.newaxis, :]
        dir_noise_2 = dir_noise_2[np.newaxis, :, :]

        dir_noise_1 = np.repeat(dir_noise_1, repeats=N[1], axis=1)
        dir_noise_2 = np.repeat(dir_noise_2, repeats=N[0], axis=0)

        init_phi_1 = PERTURB_RATIO * init_phi_truth_1 + (1 - PERTURB_RATIO) * dir_noise_1
        init_phi_2 = PERTURB_RATIO * init_phi_truth_2 + (1 - PERTURB_RATIO) * dir_noise_2     


    if args.init_type == 'perturb-pi':
        pi_1_ = pi_1[:, np.newaxis, :] 
        pi_2_ = pi_2[np.newaxis, :, :]

        init_phi_truth_1 = np.repeat(pi_1_, repeats=N[1], axis=1)
        init_phi_truth_2 = np.repeat(pi_2_, repeats=N[0], axis=0)

        dir_noise_1 = np.random.dirichlet(np.ones(K[0]) * (0.5), size=N[0]) #(N1, K1)
        dir_noise_2 = np.random.dirichlet(np.ones(K[1]) * (0.5), size=N[1]) #(N2, K2)

        dir_noise_1 = dir_noise_1[:, np.newaxis, :]
        dir_noise_2 = dir_noise_2[np.newaxis, :, :]

        dir_noise_1 = np.repeat(dir_noise_1, repeats=N[1], axis=1)
        dir_noise_2 = np.repeat(dir_noise_2, repeats=N[0], axis=0)

        init_phi_1 = PERTURB_RATIO * init_phi_truth_1 + (1 - PERTURB_RATIO) * dir_noise_1
        init_phi_2 = PERTURB_RATIO * init_phi_truth_2 + (1 - PERTURB_RATIO) * dir_noise_2  


    if args.init_type == 'random-multinomial':
        z_12 = np.array([np.random.multinomial(n=1, pvals=[1/K[0]]*K[0], size=N[1]) for i in range(N[0])]) #(N1, N2, K1) 
        z_21 = np.array([np.random.multinomial(n=1, pvals=[1/K[1]]*K[1], size=N[0]) for j in range(N[1])]) #(N2, N1, K2)

        z_12 = z_12.astype(np.float64)
        z_21 = np.transpose(z_21.astype(np.float64), axes=[1,0,2]).copy()

        init_phi_1 = z_12
        init_phi_2 = z_21


    if args.init_type == 'random-dir':
        pi_1 = np.random.dirichlet(np.ones(K[0]) * (N[0]/K[0]), size=N[0]) #(N1, K1)
        pi_2 = np.random.dirichlet(np.ones(K[1]) * (N[1]/K[1]), size=N[1]) #(N2, K2)       

        pi_1_ = pi_1[:, np.newaxis, :] 
        pi_2_ = pi_2[np.newaxis, :, :]

        init_phi_1 = np.repeat(pi_1_, repeats=N[1], axis=1)
        init_phi_2 = np.repeat(pi_2_, repeats=N[0], axis=0) 


    mmb_object.init_var_params(init_phi_1=init_phi_1, init_phi_2=init_phi_2)
    
    if K[0] == K[1]:
        p, q = 0.10, 0.01
        init_omega = np.ones((K[0], K[0]),dtype=int)*q + np.eye(K[0],dtype=int)*(p-q)
        mmb_object.init_lkh_params(init_omega=init_omega) 
        #mmb_object.init_lkh_params(init_omega=None) 
    else:
        mmb_object.init_lkh_params(init_omega=None)

    mmb_object.run_variational_em_naive(save_dir=args.save_dir, init_repeat=args.init_repeat, var_elbo=args.var_elbo, phi_constr=args.phi_constr, plot_option=True)



