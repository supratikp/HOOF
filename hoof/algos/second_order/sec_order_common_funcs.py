import numpy as np

def traj_segment_generator_with_gl(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, _, _, _ = pi.step(np.hstack((ob, np.zeros((1,2)))), stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "new" : news,
                    "ac" : acs, "prevac" : prevacs, 
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv_with_gl(pi, seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    lg_shape = seg['ob'].shape[:-1] + (1,)
    ob_lam_gam = np.concatenate((seg['ob'], 
                                 gamma*np.ones(lg_shape), 
                                 lam*np.ones(lg_shape)), axis=2)
    vpred = pi._evaluate([pi.vf], ob_lam_gam)[0]
    nextvpred = vpred[-1]*(1-seg['new'][-1])
    vpred_next = np.append(vpred, nextvpred)
    T = len(seg["rew"])
    adv = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred_next[t+1] * nonterminal - vpred_next[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    tdlamret = adv + vpred
    return ob_lam_gam, vpred, adv, tdlamret


def add_vtarg_and_adv_without_gl(pi, seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    lg_shape = seg['ob'].shape[:-1] + (1,)
    ob_lam_gam = np.concatenate((seg['ob'], 
                                 np.ones(lg_shape), 
                                 np.ones(lg_shape)), axis=2)
    vpred = pi._evaluate([pi.vf], ob_lam_gam)[0]
    nextvpred = vpred[-1]*(1-seg['new'][-1])
    vpred_next = np.append(vpred, nextvpred)
    T = len(seg["rew"])
    adv = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred_next[t+1] * nonterminal - vpred_next[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    tdlamret = adv + vpred
    return ob_lam_gam, vpred, adv, tdlamret


def wis_estimate(seg, lik_ratio):
    norm_val_rwd = np.copy(seg['rew'])
    norm_val_rwd = (norm_val_rwd - np.min(norm_val_rwd))/(np.max(norm_val_rwd) - np.min(norm_val_rwd))
    traj_idx = [0] + list(np.where(seg['new']==1)[0]) + [len(seg['new'])]
    is_wts = []
    norm_rets = []
    for n in range(len(traj_idx)-1):
        is_wts += [lik_ratio[traj_idx[n]:traj_idx[n+1]]]
        norm_rets += [norm_val_rwd[traj_idx[n]:traj_idx[n+1]]]
    is_wts = [el for el in is_wts if len(el)!=0]
    norm_rets = [el for el in norm_rets if len(el)!=0]
    is_wts = np.exp(np.array([np.sum(np.log(el)) for el in is_wts]))
    norm_rets = np.array([np.sum(el) for el in norm_rets])
    return np.dot(is_wts, norm_rets)/np.sum(is_wts)
