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

from algos.first_order.vanilla_a2c import learn_vanilla_a2c
from algos.first_order.hoof_lr_a2c import learn_hoof_lr_a2c
from algos.first_order.ent_hoof_a2c import learn_ent_hoof_a2c


parser = argparse.ArgumentParser()
parser.add_argument('yaml_file', type=str)
parser.add_argument('algo_type', type=str)
parser.add_argument('start', type=int)
parser.add_argument('nenvs', type=int)
args = parser.parse_args()

rs = 100*(args.start+1)
np.random.seed(rs)

with open(os.path.join('yaml_files', args.yaml_file+'.yaml'), 'r') as stream:
    params = yaml.load(stream)
env_type = 'mujoco'
env_id = params['env']
log_path = os.path.join('results', env_id, args.algo_type)


if args.algo_type=='A2C_RMSProp_HOOF_LR':
    log_path = os.path.join(log_path, 'max_kl_'+str(params['a2c_max_kl']), 'rs_'+str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    env = make_vec_env(env_id, env_type, args.nenvs, rs)
    learnt_model = learn_hoof_lr_a2c('mlp', 
                                  env, 
                                  optimiser='RMSProp',
                                  total_timesteps=params['total_ts'],
                                  lr_upper_bound=params['a2c_lr_ub'],
                                  num_lr=params['a2c_n_lr'],
                                  max_kl=params['a2c_max_kl'],
                                  seed=rs)
elif args.algo_type=='A2C_SGD_HOOF_LR':
    log_path = os.path.join(log_path, 'max_kl_'+str(params['a2c_max_kl']), 'rs_'+str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    env = make_vec_env(env_id, env_type, args.nenvs, rs)
    learnt_model = learn_hoof_lr_a2c('mlp', 
                                  env, 
                                  optimiser='SGD',
                                  total_timesteps=params['total_ts'],
                                  lr_upper_bound=params['a2c_lr_ub'],
                                  num_lr=params['a2c_n_lr'],
                                  max_kl=params['a2c_max_kl'],
                                  max_grad_norm=0.5,
                                  seed=rs)

elif args.algo_type=='A2C_RMSProp_HOOF_Ent':
    log_path = os.path.join(log_path, 'max_kl_'+str(params['a2c_max_kl']), 'rs_'+str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    env = make_vec_env(env_id, env_type, args.nenvs, rs)
    learnt_model = learn_ent_hoof_a2c('mlp', 
                                  env, 
                                  optimiser='RMSProp',
                                  total_timesteps=params['total_ts'],
                                  lr_upper_bound=params['a2c_lr_ub'],
                                  ent_upper_bound=params['a2c_ent_ub'],
                                  num_lr=params['a2c_n_lr'],
                                  num_ent_coeff=params['a2c_n_ent'],
                                  max_kl=params['a2c_max_kl'],
                                  seed=rs)
elif args.algo_type=='A2C_SGD_HOOF_Ent':
    log_path = os.path.join(log_path, 'max_kl_'+str(params['a2c_max_kl']), 'rs_'+str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    env = make_vec_env(env_id, env_type, args.nenvs, rs)
    learnt_model = learn_ent_hoof_a2c('mlp', 
                                  env, 
                                  optimiser='SGD',
                                  total_timesteps=params['total_ts'],
                                  lr_upper_bound=params['a2c_lr_ub'],
                                  ent_upper_bound=params['a2c_ent_ub'],
                                  num_lr=params['a2c_n_lr'],
                                  num_ent_coeff=params['a2c_n_ent'],
                                  max_kl=params['a2c_max_kl'],
                                  seed=rs)

elif args.algo_type=='RMSProp_Baseline_A2C':
    log_path = os.path.join(log_path, 'rs_'+str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    env = make_vec_env(env_id, env_type, args.nenvs, rs)
    learnt_model = learn_vanilla_a2c('mlp',
                                     env,
                                     optimiser='RMSProp',
                                     total_timesteps=params['total_ts'],
                                     seed=rs) 
elif args.algo_type=='SGD_Baseline_A2C':
    log_path = os.path.join(log_path, 'rs_'+str(rs))
    logger.configure(dir=os.path.join(os.getcwd(), log_path))
    env = make_vec_env(env_id, env_type, args.nenvs, rs)
    learnt_model = learn_vanilla_a2c('mlp',
                                     env,
                                     optimiser='SGD',
                                     total_timesteps=params['total_ts'],
                                     seed=rs) 
else:
    raise NotImplementedError
