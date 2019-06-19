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

parser = argparse.ArgumentParser()
parser.add_argument('yaml_file', type=str)
parser.add_argument('algo_type', type=str)
parser.add_argument('start', type=int)
args = parser.parse_args()

rs = 100*(args.start+1)
np.random.seed(rs)

with open(os.path.join('yaml_files', args.yaml_file+'.yaml'), 'r') as stream:
    params = yaml.load(stream)
env_type = 'mujoco'
env_id = params['env']

log_path = os.path.join('results', env_id, args.algo_type)
logger.configure(dir=os.path.join(os.getcwd(), log_path, 'rs_'+str(rs)))
env = make_vec_env(env_id, env_type, 1, rs)

"""
You can run HOOF with some hyperparameters fixed while HOOF searches over the others.
Examples:
1. To fix gamma=0.98 while searching over KL and lambda set gamma_range=0.98
2. To fix gamma=0.98 and lambda=0.999 while searching over KL, set gamma_range=0.98 and lam_range=0.999.
    IMPORTANT: You must also set num_gamma_lam=1 in this case. Otherwise it will evaluate the same policy multiple times
3. To fix kl=0.05 while searching over gamma and lambda set kl_range=0.05
    IMPORTANT: also set num_kl=1 in this case. Otherwise it will evaluate the same policy multiple times
4. If you want to set any of these to their Baselines default, just set kl_range/gamma_range/lam_range='fixed'
"""
if args.algo_type=='TNPG_HOOF_All':
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
elif args.algo_type=='Baseline_TRPO':
    learn_npg_variant('TRPO',
                      env, 
                      env_type, 
                      timesteps_per_batch=params['batch_size'],
                      total_timesteps=params['total_ts'])
else:
    raise NotImplementedError
