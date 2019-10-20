# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:25:42 2019

@author: supaul
"""

from baselines.trpo_mpi import defaults
from baselines.trpo_mpi.trpo_mpi import flatten_lists, get_variables, get_trainable_variables, get_vf_trainable_variables, get_pi_trainable_variables
from algos.second_order.sec_order_common_funcs import traj_segment_generator_with_gl, add_vtarg_and_adv_without_gl, wis_estimate
from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.policies import build_policy
from contextlib import contextmanager


def learn_hoof_no_lambgam(env, env_type, timesteps_per_batch, 
                   total_timesteps, kl_range, gamma_range, lam_range, 
                   num_kl=25, num_gamma_lam=20, **network_kwargs):
    params = defaults.mujoco()

    if gamma_range is 'fixed' and lam_range is 'fixed':
        num_gamma_lam = 1
    if kl_range is 'fixed':
        num_kl = 1

    run_hoof_no_lamgam(network=params['network'],
                env=env,
                total_timesteps=total_timesteps,
                timesteps_per_batch=int(timesteps_per_batch/env.num_envs),
                kl_range=kl_range if kl_range is not 'fixed' else params['max_kl'],
                gamma_range=gamma_range if gamma_range is not 'fixed' else params['gamma'],
                lam_range=lam_range if lam_range is not 'fixed' else params['lam'],
                num_kl=num_kl,
                num_gamma_lam = num_gamma_lam,
                cg_iters=params['cg_iters'],
                seed=None,
                cg_damping=params['cg_damping'],
                vf_stepsize=params['vf_stepsize'],
                vf_iters=params['vf_iters'],
                normalize_observations=params['normalize_observations']
                )
    

#-------------------------------------------------

def run_hoof_no_lamgam(
        network,
        env,
        total_timesteps,
        timesteps_per_batch, # what to train on
        kl_range,
        gamma_range,
        lam_range, # advantage estimation
        num_kl,
        num_gamma_lam,
        cg_iters=10,
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm
    Parameters:
    ----------
    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets
    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class
    timesteps_per_batch     timesteps per gradient estimation batch
    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )
    ent_coef                coefficient of policy entropy term in the optimization objective
    cg_iters                number of iterations of conjugate gradient algorithm
    cg_damping              conjugate gradient damping
    vf_stepsize             learning rate for adam optimizer used to optimie value function loss
    vf_iters                number of iterations of value function optimization iterations per each policy optimization step
    total_timesteps           max number of timesteps
    max_episodes            max number of episodes
    max_iters               maximum number of policy optimization iterations
    callback                function to be called with (locals(), globals()) each policy optimization step
    load_path               str, path to load the model from (default: None, i.e. no model is loaded)
    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
    Returns:
    -------
    learnt model
    '''

    MPI = None
    nworkers = 1
    rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))


    policy = build_policy(env, network, value_network='copy', **network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    # +2 for gamma, lambda
    ob = tf.placeholder(shape=(None, env.observation_space.shape[0]+2), 
                                dtype=env.observation_space.dtype, 
                                name='Ob')
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi")
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    compute_ratio = U.function([ob, ac, atarg], ratio) # IS ratio - used for computing IS weights

    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    U.initialize()
    if load_path is not None:
        pi.load(load_path)

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_with_gl(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    kl_range = np.atleast_1d(kl_range)
    gamma_range = np.atleast_1d(gamma_range)
    lam_range = np.atleast_1d(lam_range)

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        
        thbefore = get_flat()

        rand_gamma = gamma_range[0] + (gamma_range[-1]-gamma_range[0])*np.random.rand(num_gamma_lam)
        rand_lam = lam_range[0] + (lam_range[-1]-lam_range[0])*np.random.rand(num_gamma_lam)
        rand_kl = kl_range[0] + (kl_range[-1]-kl_range[0])*np.random.rand(num_kl)

        opt_polval = -10**8
        est_polval = np.zeros((num_gamma_lam,num_kl))
        ob_lam_gam = []
        tdlamret = []
        vpred = []
        
        for gl in range(num_gamma_lam):
            oblg, vpredbefore, atarg, tdlr = add_vtarg_and_adv_without_gl(pi, seg, rand_gamma[gl], rand_lam[gl])
            
            ob_lam_gam  += [oblg]
            tdlamret += [tdlr]
            vpred += [vpredbefore]
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            pol_ob = np.concatenate((seg['ob'], np.zeros(seg['ob'].shape[:-1] + (2,))), axis=-1)
            args = pol_ob, seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]
            def fisher_vector_product(p):
                return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

            assign_old_eq_new() # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=False)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                surrbefore = lossbefore[0]
                
                for m, kl in enumerate(rand_kl):
                    lm = np.sqrt(shs/kl)
                    fullstep = stepdir/lm
                    thnew = thbefore + fullstep
                    set_from_flat(thnew)
                
                    # compute the IS estimates
                    lik_ratio = compute_ratio(*args)
                    est_polval[gl, m] = wis_estimate(seg, lik_ratio)
    
                    # update best policy found so far
                    if est_polval[gl, m]>opt_polval:
                        opt_polval = est_polval[gl, m]
                        opt_th = thnew
                        opt_kl = kl
                        opt_gamma = rand_gamma[gl]
                        opt_lam = rand_lam[gl]
                        opt_vpredbefore = vpredbefore
                        opt_tdlr = tdlr
                        meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                        improve = surr - surrbefore
                        expectedimprove = g.dot(fullstep)
                    set_from_flat(thbefore)
        logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
        set_from_flat(opt_th)
        
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        ob_lam_gam = np.concatenate(ob_lam_gam, axis=0)
        tdlamret = np.concatenate(tdlamret, axis=0)
        vpred = np.concatenate(vpred, axis=0)
        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((ob_lam_gam, tdlamret),
                include_final_partial_batch=False, batch_size=num_gamma_lam*64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpred, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("Opt_KL", opt_kl)
        logger.record_tabular("gamma", opt_gamma)
        logger.record_tabular("lam", opt_lam)

        if rank==0:
            logger.dump_tabular()

    return pi