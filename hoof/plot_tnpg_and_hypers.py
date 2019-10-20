# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:27:25 2017

@author: supaul
"""

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu
from baselines.common.plot_util import symmetric_ema

import seaborn as sns
sns.set_style("whitegrid")
legendsize = 14
majorsize = 12
fontsize = 14

def load_data(item, path_a, window):
    results = pu.load_results(path_a)
    try: # for A2C
        xys = [(np.array(res.progress['total_timesteps'][1:]), np.array(res.progress[item][1:])) for res in results]
    except: # for TNPG
        xys = [(np.array(res.progress['TimestepsSoFar']), np.array(res.progress[item])) for res in results]
    origxs = [xy[0] for xy in xys]
    print([x[-1] for x in origxs])
    low  = max(x[0] for x in origxs)
    high = min(x[-1] for x in origxs)
    resample = len(origxs[0])
    ts = np.linspace(low, high, resample)+1
    ys = []
    for (x, y) in xys:
        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=window)[1])
    val = np.array(ys).T
    mean_val = np.mean(val, axis=1)
    sd_val = np.std(val, axis=1)
    med_val = np.median(val, axis=1)
    q1_val = np.percentile(val, 25, axis=1)
    q3_val = np.percentile(val, 75, axis=1)
    return {'ts':ts, 'median':med_val, 'q1':q1_val, 'q3':q3_val, 'mean':mean_val, 'std':sd_val}


def plot_stuff(items, labels, stat='median'):
    for i,el in enumerate(items):
        if stat=='median':
            plt.plot(el['ts'], el['median'], lw=2, label=labels[i])
            plt.fill_between(el['ts'], el['q1'], el['q3'], alpha=0.3)
        if stat=='mean':
            plt.plot(el['ts'], el['mean'], lw=2, label=labels[i])
            plt.fill_between(el['ts'], el['mean']-el['std'], el['mean']+el['std'], alpha=0.3)

window = 10


#%%
""" TRPO/NPG Returns plot """

envs = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2']

# Plot 2: HOOF TNPG vs Baseline TRPO
for i,env in enumerate(envs):
    HOOF = load_data('EpRewMean', osp.join("results_NPG", env, 'HOOF_All'), window)
    TRPO = load_data('EpRewMean', osp.join("results_NPG", env, 'TRPO'), window)
    plot_stuff([HOOF, TRPO], ['HOOF-TNPG', 'Baseline TRPO'])
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    plt.legend(loc=0, prop={'size': legendsize})
    plt.ylabel("Returns", fontsize=fontsize)
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.show()


#%%
# Plot 3: learnt hyperparameters for TNPG
hypers = ['Opt_KL', 'gamma', 'lam']
labels = ['KL Constraint', r'$\gamma$', r'$\lambda$']
for i,hyper in enumerate(hypers):
    cheetah = load_data(hyper, osp.join("results_NPG", 'HalfCheetah-v2', 'HOOF_All'), window)
    hopper = load_data(hyper, osp.join("results_NPG", 'Hopper-v2', 'HOOF_All'), window)
    ant = load_data(hyper, osp.join("results_NPG", 'Ant-v2', 'HOOF_All'), window)
    walker = load_data(hyper, osp.join("results_NPG", 'Walker2d-v2', 'HOOF_All'), window)

    plot_stuff([cheetah, hopper, ant, walker], ['HalfCheetah', 'Hopper', 'Ant', 'Walker'])
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    plt.legend(loc=0, prop={'size': legendsize})
    plt.ylabel(labels[i], fontsize=fontsize)
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.show()


#%%

# Plot 4: learnt LR for A2C
envs = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2']
labels = ['HalfCheetah', 'Hopper', 'Ant', 'Walker']
data = []
for _,env in enumerate(envs):
    data += [load_data('opt_lr', osp.join('results_A2C',env,'HOOF_A2C_LR_RMSProp', 'max_kl_0.03'), window)]
plot_stuff(data, labels)
plt.tick_params(axis='both', which='major', labelsize=majorsize)
plt.legend(loc=0, prop={'size': legendsize})
plt.ylabel('Learnt '+r'$\alpha$', fontsize=fontsize)
plt.xlabel("Iterations", fontsize=fontsize)
plt.yscale('log')
plt.xlim([0,5*10**6])
plt.show()


#%%

# Plot 7: HOOF TNPG vs HOOF TNPG without (lam,gam) conditioned vf

envs = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2']
for i,env in enumerate(envs):
    HOOF = load_data('EpRewMean', osp.join("results_NPG", env, 'HOOF_All'), window)
    no_lamgam = load_data('EpRewMean', osp.join("results_NPG", env, 'HOOF_no_lamgam'), window)
    plot_stuff([HOOF, no_lamgam], ['HOOF-TNPG', 'HOOF-no-'+r'$(\gamma,\lambda)$'])
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    plt.legend(loc=0, prop={'size': legendsize})
    plt.ylabel("Returns", fontsize=fontsize)
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.show()
