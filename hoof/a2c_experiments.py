# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:13:25 2019

@author: supaul
"""

import numpy as np
import os, yaml, argparse

from baselines.common.cmd_util import make_vec_env
from baselines import logger
from algos.a2c.hoof_a2c import learn_hoof_a2c

parser = argparse.ArgumentParser()
parser.add_argument('mjenv', type=str)
parser.add_argument('algo_type', type=str)
parser.add_argument('optimiser', type=str)
parser.add_argument('max_kl', type=float)
parser.add_argument('start', type=int)
args = parser.parse_args()

rs = 100*(args.start+1)
np.random.seed(rs)

with open('experiment_params.yaml', 'r') as stream:
    params = yaml.load(stream)
env_type = 'mujoco'
env_id = params[args.mjenv]
params['algo_type'] = args.algo_type
params['optimiser'] = args.optimiser
params['max_kl'] = args.max_kl
params['rseed'] = rs

log_path = os.path.join('results_A2C', env_id, args.algo_type+'_'+args.optimiser, 'max_kl_'+str(args.max_kl), 'rs_'+str(rs))
logger.configure(dir=os.path.join(os.getcwd(), log_path))
with open(os.path.join(log_path,'params.yaml'), 'w') as outfile:
    yaml.dump(params, outfile, default_flow_style=False)

env = make_vec_env(env_id, env_type, int(params['n_envs']), rs)


if args.algo_type=='Vanilla_A2C':
    learnt_model = learn_hoof_a2c('mlp', 
                                  env,
                                  optimiser=args.optimiser,
                                  total_timesteps=params['total_ts'],
                                  seed=rs)
    learnt_model.save(os.path.join(log_path, 'learnt_model'))

if args.algo_type=='HOOF_A2C_LR':
    learnt_model = learn_hoof_a2c('mlp', 
                                  env,
                                  optimiser=args.optimiser,
                                  max_kl=args.max_kl,
                                  lr_upper_bound=params['lr_ub'],
                                  num_lr=params['num_lr'],
                                  total_timesteps=params['total_ts'],
                                  seed=rs)
    learnt_model.save(os.path.join(log_path, 'learnt_model'))

if args.algo_type=='HOOF_A2C_Ent_LR':
    learnt_model = learn_hoof_a2c('mlp', 
                                  env,
                                  optimiser=args.optimiser,
                                  max_kl=args.max_kl,
                                  lr_upper_bound=params['lr_ub'],
                                  num_lr=params['num_lr'],
                                  ent_upper_bound=params['ent_ub'],
                                  num_ent_coeff=params['num_ent_coeff'],
                                  total_timesteps=params['total_ts'],
                                  seed=rs)
    learnt_model.save(os.path.join(log_path, 'learnt_model'))
