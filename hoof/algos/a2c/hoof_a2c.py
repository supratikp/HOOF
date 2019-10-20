"""
Updated version of Baselines A2C: LR + Ent coeff learnt using HOOF
"""
import time
import functools
import tensorflow as tf
import numpy as np
from baselines import logger

import baselines.common.tf_util as U
from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy
from baselines.a2c.utils import Scheduler, find_trainable_variables

from algos.a2c.hoof_runner import HOOF_Runner
from tensorflow import losses

from baselines.ppo2.ppo2 import safemean
from collections import deque

def wis_estimate(n_env, n_steps, undisc_rwds, lik_ratio):
    lik_ratio = np.reshape(lik_ratio, (n_env, n_steps))
    is_wts = np.product(lik_ratio, axis=1)
    norm_val_rwd = (undisc_rwds- np.min(undisc_rwds))/(np.max(undisc_rwds) - np.min(undisc_rwds))
    norm_val_rwd = np.reshape(norm_val_rwd, (n_env, n_steps))
    norm_rets = np.sum(norm_val_rwd, axis=1)
    est_val = np.dot(is_wts, norm_rets)/np.sum(is_wts)
    return est_val

class HOOF_Model(object):
    def __init__(self, policy, env, nsteps, optimiser,
            ent_coef, vf_coef, max_grad_norm, total_timesteps,
            alpha, epsilon # defaults for RMSProp
            ):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps

        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        Ent_Coeff = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*Ent_Coeff + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        if optimiser=='RMSProp':
            trainer = tf.train.RMSPropOptimizer(learning_rate=1.0, decay=alpha, epsilon=epsilon)
        elif optimiser=='SGD':
            trainer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        _train = trainer.apply_gradients(grads)

        #https://stackoverflow.com/a/45624533
        _slot_vars = [trainer.get_slot(var, name) for name in trainer.get_slot_names() for var in params]
        SLOTS = [tf.placeholder(tf.float32, slot.shape) for slot in _slot_vars]
        _set_slots = [var.assign(SLOTS[i]) for i, var in enumerate(_slot_vars)]

        def get_opt_state():
            return sess.run(_slot_vars)

        def set_opt_state(state):
            feed = {k: v for k, v in zip(SLOTS, state)}
            return sess.run(_set_slots, feed)

        def train(obs, states, rewards, masks, actions, values, ent_coeff):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, Ent_Coeff:ent_coeff}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def get_mean_std_neg_ll(obs, actions):
            td_map = {train_model.X:obs, A:actions}
            vals = sess.run([train_model.pd.mean, train_model.pd.std, neglogpac], td_map)
            return vals

        self.get_mean_std_neg_ll = get_mean_std_neg_ll
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        self.get_opt_state = get_opt_state
        self.set_opt_state = set_opt_state
        tf.global_variables_initializer().run(session=sess)

#--------------------------------------------------------------------------


def learn_hoof_a2c(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None, # Baselines default settings till here
    optimiser='RMSProp', 
    lr_upper_bound=None,
    ent_upper_bound=None,
    num_lr=None,
    num_ent_coeff=None,
    max_kl=-1.0, # -1.0 is for no KL constraint
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    # overwrite default params if using HOOF
    if lr_upper_bound is not None:
        lr = 1.0
        lrschedule = 'constant'
    else:
        num_lr = 1
        
    if ent_upper_bound is None:
        num_ent_coeff = 1 
    
    # Instantiate the model object (that creates step_model and train_model)
    model = HOOF_Model(policy=policy, env=env, nsteps=nsteps, optimiser=optimiser,
                        ent_coef=ent_coef, vf_coef=vf_coef, 
                        max_grad_norm=max_grad_norm, total_timesteps=total_timesteps,
                        alpha=alpha, epsilon=epsilon # defaults for RMSProp
                        )

    runner = HOOF_Runner(env, model, nsteps=nsteps, gamma=gamma)
    epinfobuf = deque(maxlen=100)

    lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # model helper functions
    model_params = find_trainable_variables("a2c_model")
    get_flat = U.GetFlat(model_params)
    set_from_flat = U.SetFromFlat(model_params)

    # for Gaussian policies
    def kl(new_mean, new_sd, old_mean, old_sd):
        approx_kl = np.log(new_sd/old_sd) + (old_sd**2 + (old_mean - new_mean)**2)/(2.0*new_sd**2 + 10**-8) - 0.5
        approx_kl = np.sum(approx_kl, axis=1)
        approx_kl = np.mean(approx_kl)
        return approx_kl

    if max_kl==-1.0: # set max kl to a high val in case there is no constraint
        max_kl=10**8
        
    # Start total timer
    tstart = time.time()

    for update in range(1, int(total_timesteps//nbatch+1)):
        obs, states, rewards, masks, actions, values, undisc_rwds, epinfos = runner.run()
        epinfobuf.extend(epinfos)
        old_mean, old_sd, old_neg_ll = model.get_mean_std_neg_ll(obs, actions)
        for step in range(len(obs)):
            cur_lr = lr.value()

        opt_pol_val = -10**8
        old_params = get_flat()
        rms_weights_before_upd = model.get_opt_state()
        approx_kl = np.zeros((num_ent_coeff, num_lr))
        epv = np.zeros((num_ent_coeff, num_lr))
        rand_lr = lr_upper_bound*np.random.rand(num_lr) if lr_upper_bound is not None else [cur_lr]
        rand_lr = np.sort(rand_lr)
        rand_ent_coeff = ent_upper_bound*np.random.rand(num_ent_coeff) if ent_upper_bound is not None else [ent_coef]
        
        for nec in range(num_ent_coeff):
            # reset policy and optimiser
            set_from_flat(old_params)
            model.set_opt_state(rms_weights_before_upd)

            # get grads for loss fn with given entropy coeff
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values, rand_ent_coeff[nec])
            new_params = get_flat()
            ent_grads = new_params - old_params
            
            # enumerate over different LR
            for nlr in range(num_lr):
                new_params = old_params + rand_lr[nlr]*ent_grads
                set_from_flat(new_params)
                new_mean, new_sd, new_neg_ll = model.get_mean_std_neg_ll(obs, actions)
                lik_ratio = np.exp(-new_neg_ll + old_neg_ll)
                est_pol_val = wis_estimate(nenvs, nsteps, undisc_rwds, lik_ratio)
                approx_kl[nec, nlr] = kl(new_mean, new_sd, old_mean, old_sd)
                epv[nec, nlr] = est_pol_val

                if (nec==0 and nlr==0) or (est_pol_val>opt_pol_val and approx_kl[nec, nlr]<max_kl):
                    opt_pol_val = est_pol_val
                    opt_pol_params = get_flat()
                    opt_rms_wts = model.get_opt_state()
                    opt_lr = rand_lr[nlr]
                    opt_ent_coeff = rand_ent_coeff[nec]
                    opt_kl = approx_kl[nec, nlr]
        
        # update policy and rms prop to optimal wts
        set_from_flat(opt_pol_params)
        model.set_opt_state(opt_rms_wts)

        # Shrink LR search space if too many get rejected
        if lr_upper_bound is not None:
            rejections = np.sum(approx_kl>max_kl)/num_lr
            if rejections>0.8:
                lr_upper_bound *= 0.8
            if rejections==0:
                lr_upper_bound *= 1.25
        
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("opt_lr", float(opt_lr))
            logger.record_tabular("opt_ent_coeff", float(opt_ent_coeff))
            logger.record_tabular("approx_kl", float(opt_kl))
            if lr_upper_bound is not None: 
                logger.record_tabular("rejections", rejections)
                logger.record_tabular("lr_ub", lr_upper_bound)
            logger.record_tabular("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.record_tabular("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.dump_tabular()
    return model

