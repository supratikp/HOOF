# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 23:35:16 2018

@author: supaul
"""

import numpy as np
 
    
def wis_estimate(n_env, n_steps, undisc_rwds, lik_ratio):
    lik_ratio = np.reshape(lik_ratio, (n_env, n_steps))
    is_wts = np.product(lik_ratio, axis=1)
    norm_val_rwd = (undisc_rwds- np.min(undisc_rwds))/(np.max(undisc_rwds) - np.min(undisc_rwds))
    norm_val_rwd = np.reshape(norm_val_rwd, (n_env, n_steps))
    norm_rets = np.sum(norm_val_rwd, axis=1)
    est_val = np.dot(is_wts, norm_rets)/np.sum(is_wts)
    return est_val
