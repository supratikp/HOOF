# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:13:25 2019

@author: supaul
"""

import argparse
import yaml
import numpy as np
import os

from baselines.common.cmd_util import make_vec_env
from baselines import logger
from algos.second_order.hoof_all_npg import learn_hoof_all
from algos.second_order.vanilla_sec_order import learn_npg_variant
from algos.second_order.hoof_no_uvfa_npg import learn_hoof_no_lambgam


parser = argparse.ArgumentParser()
parser.add_argument('mjenv', type=str)
parser.add_argument('algo_type', type=str)
parser.add_argument('start', type=int)
args = parser.parse_args()

rs = 100*(args.start+1)
np.random.seed(rs)

with open('experiment_params.yaml', 'r') as stream:
    params = yaml.load(stream)
env_type = 'mujoco'
env_id = params[args.mjenv]
params['algo_type'] = args.algo_type
params['rseed'] = rs

log_path = os.path.join('results_NPG', env_id, args.algo_type, 'rs_'+str(rs))
logger.configure(dir=os.path.join(os.getcwd(), log_path))

# dump the params once in the folder
with open(os.path.join(log_path,'params.yaml'), 'w') as outfile:
    yaml.dump(params, outfile, default_flow_style=False)

env = make_vec_env(env_id, env_type, 1, rs)

if args.algo_type=='HOOF_All':
    learnt_model = learn_hoof_all(env, 
                                 env_type, 
                                 timesteps_per_batch=params['batch_size'],
                                 total_timesteps=params['total_ts'], 
                                 kl_range=params['kl_bound'],
                                 gamma_range=params['discount_bound'],
                                 lam_range=params['lambda_bound'],
                                 num_kl=params['all_n_kl'], 
                                 num_gamma_lam=params['all_n_gl'],
                                 )
elif args.algo_type=='HOOF_no_lamgam':
    learnt_model = learn_hoof_no_lambgam(env, 
                                 env_type, 
                                 timesteps_per_batch=params['batch_size'],
                                 total_timesteps=params['total_ts'], 
                                 kl_range=params['kl_bound'],
                                 gamma_range=params['discount_bound'],
                                 lam_range=params['lambda_bound'],
                                 num_kl=params['all_n_kl'], 
                                 num_gamma_lam=params['all_n_gl'],
                                 )
elif args.algo_type=='TRPO':
    learn_npg_variant(args.algo_type,
                      env, 
                      env_type, 
                      timesteps_per_batch=params['batch_size'],
                      total_timesteps=params['total_ts'])
else:
    raise NotImplementedError
