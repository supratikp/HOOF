# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 23:35:16 2018

@author: supaul
"""

import numpy as np

class WIS():
    def __init__(self, n_env, n_steps):
        self.ne = n_env
        self.ns = n_steps

    def get_old_pol_stats(self, actions, neg_ll, traj_rwds):
        self.acts = actions
        old_neg_ll = np.reshape(neg_ll, (self.ne, self.ns))
        self.old_neg_ll = np.sum(old_neg_ll, axis=1)
        self.rwd_min = np.min(traj_rwds)
        self.rwd_max = np.max(traj_rwds)
        self.norm_rwds = (traj_rwds - self.rwd_min)/(self.rwd_max - self.rwd_min)
        
    def est_polval(self, new_neg_ll):
        new_neg_ll = np.reshape(new_neg_ll, (self.ne, self.ns))
        new_neg_ll = np.sum(new_neg_ll, axis=1)
        weights = np.exp(-new_neg_ll + self.old_neg_ll)
#        print(weights)
        pol_val_est = np.dot(weights, self.norm_rwds)/np.sum(weights)
#        sd_pol_val = np.linalg.norm(weights*(self.norm_rwds-pol_val_est))
        ess = np.sum(weights)**2/np.sum(weights**2)
        return pol_val_est, ess
