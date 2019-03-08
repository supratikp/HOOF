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
from baselines.a2c.runner import Runner as Vanilla_Runner

from algos.hoof_a2c.hoof_runner import Hoof_Runner
from algos.hoof_a2c.a2c_WIS import WIS
from tensorflow import losses

class Vanilla_Model():
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

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
        LR = tf.placeholder(tf.float32, [])

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

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

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
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        # Only this bit added
        def get_neg_ll(obs, actions):
            td_map = {train_model.X:obs, A:actions}
            neg_ll = sess.run(neglogpac, td_map)
            return neg_ll

        # Only this bit added
        def get_mean_std(obs):
            td_map = {train_model.X:obs}
            vals = sess.run([train_model.pd.mean, train_model.pd.std], td_map)
            return vals

        self.get_neg_ll = get_neg_ll
        self.get_mean_std = get_mean_std
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)

#----------------------------------------------------------------------

class LR_Model(object):
    def __init__(self, policy, env, nsteps, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):

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
        LR = tf.placeholder(tf.float32, [])

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

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

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
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:1.0}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        # Only this bit added
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
        tf.global_variables_initializer().run(session=sess)


#------------------------------------------------------------


class Additive_LR_Model(object):
    def __init__(self, policy, env, nsteps, vf_coef=0.5, max_grad_norm=0.5,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):

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
        LR = tf.placeholder(tf.float32, [])

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

        pgvf_loss = pg_loss + vf_loss * vf_coef
        ent_loss = - entropy

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients

        pgvf_grads = tf.gradients(pgvf_loss, params)
        ent_grads = tf.gradients(ent_loss, params)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            pgvf_grads, pgvf_grad_norm = tf.clip_by_global_norm(pgvf_grads, max_grad_norm)
            ent_grads, ent_grad_norm = tf.clip_by_global_norm(ent_grads, max_grad_norm)
        pgvf_grads = list(zip(pgvf_grads, params))
        ent_grads = list(zip(ent_grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        pgvf_trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _pgvf_train = pgvf_trainer.apply_gradients(pgvf_grads)

        ent_trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _ent_train = ent_trainer.apply_gradients(ent_grads)

        def train(grad_type, obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:1.0}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            
            if grad_type=='pgvf':
                policy_loss, value_loss, policy_entropy, _ = sess.run(
                    [pg_loss, vf_loss, entropy, _pgvf_train],
                    td_map
                )
            if grad_type=='ent':
                policy_loss, value_loss, policy_entropy, _ = sess.run(
                    [pg_loss, vf_loss, entropy, _ent_train],
                    td_map
                )                
            return policy_loss, value_loss, policy_entropy

        # Only this bit added
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
        tf.global_variables_initializer().run(session=sess)

#--------------------------------------------------------------------------


def learn(
    network,
    env,
    algo_type,
    seed=None,
    nsteps=5,
    total_timesteps=int(1e6),
    lr=7e-4,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    lr_upper_bound=None,
    max_kl=None,
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

    # Instantiate the model object (that creates step_model and train_model)
    if algo_type=='hoof_full':
        model = LR_Model(policy=policy, env=env, nsteps=nsteps, total_timesteps=total_timesteps)
        wis = WIS(env.num_envs, nsteps)
        rnnr = Hoof_Runner(env, model, nsteps=nsteps, gamma=gamma)
        if lr_upper_bound is None:
            raise ValueError("specify lr bounds for HOOF")
        if max_kl is None:
            raise ValueError("specify max KL for HOOF")
    elif algo_type=='vanilla':
        model = Vanilla_Model(policy=policy, env=env, nsteps=nsteps, lr=lr, total_timesteps=total_timesteps)
        rnnr = Vanilla_Runner(env, model, nsteps=nsteps, gamma=gamma)
    elif algo_type=='hoof_additive':
        model = Additive_LR_Model(policy=policy, env=env, nsteps=nsteps, total_timesteps=total_timesteps)
        wis = WIS(env.num_envs, nsteps)
        rnnr = Hoof_Runner(env, model, nsteps=nsteps, gamma=gamma)
        if lr_upper_bound is None:
            raise ValueError("specify lr bounds for HOOF")
        if max_kl is None:
            raise ValueError("specify max KL for HOOF")
    else:
        raise ValueError("algo_type must be hoof_full/vanilla")
        
    if load_path is not None:
        model.load(load_path)
    
    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # model helper functions
    model_params = find_trainable_variables("a2c_model")
    get_flat = U.GetFlat(model_params)
    set_from_flat = U.SetFromFlat(model_params)

    def kl(new_mean, new_sd, old_mean, old_sd):
        approx_kl = np.log(new_sd/old_sd) + (old_sd**2 + (old_mean - new_mean)**2)/(2.0*new_sd**2 + 10**-8) - 0.5
        approx_kl = np.sum(approx_kl, axis=1)
        approx_kl = np.mean(approx_kl)
        return approx_kl
        
    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        if algo_type=='vanilla':
            old_params = get_flat()
            obs, states, rewards, masks, actions, values = rnnr.run()
            old_mean, old_sd = model.get_mean_std(obs)
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
            new_params = get_flat()
            grad_norm = np.linalg.norm(new_params-old_params)
            logger.record_tabular("grad_norm", grad_norm)
            new_mean, new_sd = model.get_mean_std(obs)
            approx_kl = kl(new_mean, new_sd, old_mean, old_sd)
            logger.record_tabular("approx_kl", float(approx_kl))
        elif algo_type=='hoof_full':
            old_params = get_flat()
            obs, states, rewards, masks, actions, values, traj_rwds = rnnr.run()
            old_mean, old_sd, old_neg_ll = model.get_mean_std_neg_ll(obs, actions)
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
            new_params = get_flat()
            grads = new_params - old_params
            logger.record_tabular("grad_norm", np.linalg.norm(grads))
            wis.get_old_pol_stats(actions, old_neg_ll, traj_rwds)            
            lr_samples = np.random.rand(100)*lr_upper_bound
            lr_samples = np.sort(lr_samples)
            approx_kl = np.zeros(100)
            epv = np.zeros(100)
            sd_pv = np.zeros(100)
            for i in range(len(lr_samples)):
                new_params = old_params + lr_samples[i]*grads
                set_from_flat(new_params)
                new_mean, new_sd, new_neg_ll = model.get_mean_std_neg_ll(obs, actions)
                est_pol_val, sd_pol_val = wis.est_polval(new_neg_ll)
                approx_kl[i] = kl(new_mean, new_sd, old_mean, old_sd)
                epv[i] = est_pol_val
                sd_pv[i] = sd_pol_val
            kl_idx = np.where(approx_kl<max_kl)[0]
            if len(kl_idx)==0:
                opt_idx = 0
            else:
                opt_idx = np.argmax(epv[kl_idx])
            opt_kl = approx_kl[opt_idx]
            opt_lrs = lr_samples[opt_idx]
            opt_pol_params = old_params + opt_lrs*grads
            set_from_flat(opt_pol_params)
            logger.record_tabular("hoof_lr", float(opt_lrs))
            logger.record_tabular("approx_kl", float(opt_kl))
        elif algo_type=='hoof_additive':
            old_params = get_flat()
            obs, states, rewards, masks, actions, values, traj_rwds = rnnr.run()
            old_mean, old_sd, old_neg_ll = model.get_mean_std_neg_ll(obs, actions)
            wis.get_old_pol_stats(actions, old_neg_ll, traj_rwds)

            policy_loss, value_loss, policy_entropy = model.train('ent', obs, states, rewards, masks, actions, values)
            ent_params = get_flat()
            ent_grads = ent_params - old_params
            set_from_flat(old_params)
            
            policy_loss, value_loss, policy_entropy = model.train('pgvf', obs, states, rewards, masks, actions, values)
            pgvf_params = get_flat()
            pgvf_grads = pgvf_params - old_params
            set_from_flat(old_params)

            # col 0 for pgvf grads and 1 for ent grads
            lr_samples = np.sort(np.random.rand(10)*lr_upper_bound)
            pgvf_lrs = np.repeat(lr_samples,10)
            ent_lrs = np.random.rand(100)*pgvf_lrs*0.05
            lr_samples = np.vstack((pgvf_lrs, ent_lrs)).T
            approx_kl = np.zeros(100)
            epv = np.zeros(100)
            sd_pv = np.zeros(100)
            for i in range(len(lr_samples)):
                new_params = old_params + lr_samples[i,0]*pgvf_grads + lr_samples[i,1]*ent_grads
                set_from_flat(new_params)
                new_mean, new_sd, new_neg_ll = model.get_mean_std_neg_ll(obs, actions)
                est_pol_val, sd_pol_val = wis.est_polval(new_neg_ll)
                approx_kl[i] = kl(new_mean, new_sd, old_mean, old_sd)
                epv[i] = est_pol_val
                sd_pv[i] = sd_pol_val
            kl_idx = np.where(approx_kl<max_kl)[0]
            if len(kl_idx)==0:
                opt_idx = np.argmin(approx_kl)
            else:
                opt_idx = np.argmax(epv[kl_idx])
            opt_kl = approx_kl[opt_idx]
            opt_lrs = lr_samples[opt_idx]
            opt_pol_params = old_params + opt_lrs[0]*pgvf_grads + opt_lrs[1]*ent_grads
            set_from_flat(opt_pol_params)
            logger.record_tabular("hoof_lr_pgvf", float(opt_lrs[0]))
            logger.record_tabular("hoof_lr_ent", float(opt_lrs[1]))
            logger.record_tabular("approx_kl", float(opt_kl))
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
            logger.dump_tabular()
    return model

