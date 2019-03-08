# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 00:18:09 2019

@author: supaul
"""

import argparse
import numpy as np
import os.path as osp

from algos.hoof_a2c import hoof_a2c
from algos.hoof_a2c.a2c_env_makers import make_vec_env
from baselines import logger
import os


parser = argparse.ArgumentParser()
parser.add_argument('env_id', type=str)
parser.add_argument('algo_type', type=str)
parser.add_argument('max_kl', type=float)
parser.add_argument('start', type=int)
args = parser.parse_args()    


rs = 100*(args.start+1)
np.random.seed(rs)

env_type = 'mujoco'
env_id = args.env_id
if args.algo_type in ['hoof_full', 'hoof_additive']:
    log_path = osp.join('bl_results', env_id, args.algo_type, 'max_kl_'+str(args.max_kl), 'rs_'+str(rs))
else:
    log_path = osp.join('bl_results', env_id, args.algo_type, 'rs_'+str(rs))
logger.configure(dir=os.path.join(os.getcwd(), log_path))
env = make_vec_env(env_id, env_type, 20, rs)

learnt_model = hoof_a2c.learn(
                'mlp',
                env,
                algo_type=args.algo_type,
                seed=rs,
                nsteps=5,
                total_timesteps=int(1e6),
                lr=7e-4,
                lr_upper_bound=1e-2 if args.algo_type in ['hoof_full', 'hoof_additive'] else None,
                max_kl=args.max_kl if args.algo_type in ['hoof_full', 'hoof_additive'] else None,
                gamma=0.99)
learnt_model.save(os.path.join(log_path, 'learnt_model'))
