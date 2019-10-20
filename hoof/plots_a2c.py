# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:51:36 2019

@author: supaul
"""

import glob
import os
import numpy as np
from algos.plot_util import symmetric_ema
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
legendsize = 12
majorsize = 12
fontsize = 12

def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_returns_data(indir, starts=10, bin_size=1000, st=1, window=20):
    rwds = []
    ts = []
    for i in range(starts):
        path_full = os.path.join(indir, "rs_"+str(100*(i+st)))
        datas = []
        infiles = glob.glob(os.path.join(path_full, '*.monitor.csv'))
    
        for inf in infiles:
            with open(inf, 'r') as f:
                f.readline()
                f.readline()
                for line in f:
                    tmp = line.replace('\x00', '')
                    tmp = tmp.split(',')
                    t_time = float(tmp[2])
                    tmp = [t_time, int(tmp[1]), float(tmp[0])]
                    datas.append(tmp)
    
        datas = sorted(datas, key=lambda d_entry: d_entry[0])
        result = []
        timesteps = 0
        for i in range(len(datas)):
            result.append([timesteps, datas[i][-1]])
            timesteps += datas[i][1]
    
        x, y = np.array(result)[:, 0], np.array(result)[:, 1]
        x, y = fix_point(x, y, bin_size)
        low  = min(x)
        high = max(x)
        resample = len(x)
        y = symmetric_ema(np.array(x), np.array(y), low, high, resample, decay_steps=window)[1]
        rwds += [np.array(y)]
        ts += [np.array(x)]
    min_ts = min(arr[-1] for arr in ts)
    idx = np.where(ts[0]==min_ts)[0][0]
    timesteps = ts[0][:idx]
    rewards = np.zeros((idx, starts))
    for i in range(starts):
        rewards[:,i] = rwds[i][:idx]
    mean_rwds = np.mean(rewards, axis=1)
    sd_rwds = np.std(rewards, axis=1)
    med_rwds = np.median(rewards, axis=1)
    q1_rwds = np.percentile(rewards, 25, axis=1)
    q3_rwds = np.percentile(rewards, 75, axis=1)
    return timesteps, med_rwds, q1_rwds, q3_rwds, mean_rwds, sd_rwds


def plot_stuff(items, labels, stat='median', shade=True):
    for i,el in enumerate(items):
        if stat=='median':
            plt.plot(el['ts'], el['median'], lw=2, label=labels[i])
            if shade:
                plt.fill_between(el['ts'], el['q1'], el['q3'], alpha=0.3)
        if stat=='mean':
            plt.plot(el['ts'], el['mean'], lw=2, label=labels[i])
            if shade:
                plt.fill_between(el['ts'], el['mean']-el['std'], el['mean']+el['std'], alpha=0.3)
            

envs = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2']

#%%

# Plot 1: performance of HOOF vs Baseline A2C (RMSProp)
for env in envs:
    hoof = os.path.join('results_A2C', env, 'HOOF_A2C_LR_RMSProp', 'max_kl_0.03')
    ts, med, q1, q3, mean, std = load_returns_data(hoof)
    hoof = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}
    
    baseline_a2c = os.path.join('results_A2C', env, 'Vanilla_A2C_RMSProp', 'max_kl_-1.0')
    ts, med, q1, q3, mean, std = load_returns_data(baseline_a2c)
    baseline_a2c = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}
    
    plot_stuff([hoof, baseline_a2c], ['HOOF', 'Baseline A2C'])
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    plt.legend(prop={'size': legendsize})
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.ylabel("Returns", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.show()

#%%
# Plot 5: performance of HOOF with KL=0.03 vs HOOF with no KL
for env in envs:
    hoof = os.path.join('results_A2C', env, 'HOOF_A2C_LR_RMSProp', 'max_kl_0.03')
    ts, med, q1, q3, mean, std = load_returns_data(hoof)
    hoof = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}
    
    hoof_no_kl = os.path.join('results_A2C', env, 'HOOF_A2C_LR_RMSProp', 'max_kl_-1.0')
    ts, med, q1, q3, mean, std = load_returns_data(hoof_no_kl)
    hoof_no_kl = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}
    
    plot_stuff([hoof, hoof_no_kl], ['KL = 0.03', 'no KL constraint'])
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    plt.legend(prop={'size': legendsize})
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.ylabel("Returns", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.show()



#%%

# Plot 6: HOOF vs Basline A2C but with SGD for both (instead of RMSProp)
for i, env in enumerate(envs):
    hoof = os.path.join('results_A2C', env, 'HOOF_A2C_LR_SGD', 'max_kl_0.03')
    ts, med, q1, q3, mean, std = load_returns_data(hoof)
    hoof = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}
    
    baseline_a2c = os.path.join('results_A2C', env, 'Vanilla_A2C_SGD', 'max_kl_-1.0')
    ts, med, q1, q3, mean, std = load_returns_data(baseline_a2c)
    baseline_a2c = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}
    
    plot_stuff([hoof, baseline_a2c], ['SGD HOOF', 'SGD Baseline A2C'])
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    if i==0:
        plt.legend(prop={'size': legendsize})
        plt.ylabel("Returns", fontsize=fontsize)
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.show()

#%%

# Table 1: performance of HOOF for different KL constraints
def process_data(env, vals):
    data = []
    lbl = []
    last_val = []
    for i,val in enumerate(vals):
        print(str(val))
        log = os.path.join('results_A2C',env,'HOOF_A2C_LR_RMSProp','max_kl_'+str(val))
        ts, med, q1, q3, mean, std = load_returns_data(log, starts=10)
        data += [{'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}]
        lbl += [r'$\epsilon = $'+str(val)]
        last_val += [med[-1]]
    return data, lbl, last_val

kl_effect = {}
for i, env in enumerate(envs):
    print(env)
    data, label, last_val = process_data(env, np.round(np.linspace(0.01, 0.07, 7), 2))
    kl_effect[env] = last_val
print(kl_effect)    