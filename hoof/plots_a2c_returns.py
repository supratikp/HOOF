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
legendsize = 14
majorsize = 12
fontsize = 14

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
            

#%%

# A2C results

envs = ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'Walker2d-v2']

for i,env in enumerate(envs):
    log = os.path.join('results',env,'RMSProp_Baseline_A2C')
    ts, med, q1, q3, mean, std = load_returns_data(log, starts=10)
    baseline_RMSProp = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}

    log = os.path.join('results',env,'A2C_RMSProp_HOOF_LR', 'max_kl_0.03')
    ts, med, q1, q3, mean, std = load_returns_data(log, starts=10)
    HOOF_LR_RMSProp = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}

    log = os.path.join('results',env,'SGD_Baseline_A2C')
    ts, med, q1, q3, mean, std = load_returns_data(log, starts=10)
    baseline_SGD = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}

    log = os.path.join('results',env,'A2C_SGD_HOOF_LR', 'max_kl_0.03')
    ts, med, q1, q3, mean, std = load_returns_data(log, starts=10)
    HOOF_LR_SGD = {'ts': ts, 'median':med, 'q1':q1, 'q3':q3, 'mean':mean, 'std':std}

    plot_stuff([HOOF_LR_RMSProp, baseline_RMSProp], ['HOOF A2C RMSProp', 'Baseline A2C RMSProp'], 'median', shade=True)
    plot_stuff([HOOF_LR_SGD, baseline_SGD], ['HOOF A2C SGD', 'Baseline A2C SGD'], 'median', shade=True)
    plt.tick_params(axis='both', which='major', labelsize=majorsize)
    plt.legend(prop={'size': fontsize})
    plt.ylabel("Returns", fontsize=fontsize)
    plt.xlabel("Timesteps", fontsize=fontsize)
    plt.xlim([0,5*10**6])
    plt.title(env, fontsize=fontsize)
    plt.show()