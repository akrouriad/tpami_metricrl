import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import roboschool
import gym
import data_handling as dat
import rllog
from rl_shared import MLP, RunningMeanStdFilter, ValueFunction, GaussianPolicy, ValueFunctionList
import rl_shared as rl
import time


def learn(envid, nb_vfunc=2, seed=0, max_ts=1e6, norma='None', log_name=None):
    env = gym.make(envid)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    h_layer_width = 64
    h_layer_length = 2
    lr = 3e-4
    nb_epochs = 10
    eps_ppo = .2

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    obs_filter = RunningMeanStdFilter(s_dim, min_clamp=-5, max_clamp=5)
    rwd_filter = RunningMeanStdFilter(1, min_clamp=-5, max_clamp=5, center=False)

    input_sizes = [s_dim] + [h_layer_width] * h_layer_length

    value_mlp, value_fct, value_optim = [], [], []
    for k in range(nb_vfunc):
        value_mlp.append(MLP(input_sizes + [1], preproc=obs_filter))
        value_fct.append(ValueFunction(value_mlp[k]))
        value_optim.append(torch.optim.Adam(value_fct[k].parameters(), lr=lr))

    value_from_list = ValueFunctionList(value_fct)

    policy_mlp = MLP(input_sizes + [a_dim], preproc=obs_filter)
    policy_torch = GaussianPolicy(policy_mlp, a_dim)
    policy = lambda obs: policy_torch(torch.tensor(obs, dtype=torch.float)).detach().numpy()
    p_optim = torch.optim.Adam(policy_torch.parameters(), lr=lr)

    discount = .99
    lam = .95
    min_sample_per_iter = 3200

    cum_ts = 0
    iter = 0
    if log_name is not None:
        logger = rllog.PolicyIterationLogger(log_name)
    while True:
        p_paths = dat.rollouts(env, policy, min_sample_per_iter, render=False)
        iter_ts = len(p_paths['rwd'])
        cum_ts += iter_ts
        obs = torch.tensor(p_paths['obs'], dtype=torch.float)
        act = torch.tensor(p_paths['act'], dtype=torch.float)
        rwd = torch.tensor(p_paths['rwd'], dtype=torch.float)
        avg_rwd = np.sum(p_paths['rwd']) / np.sum(p_paths['done'])
        if log_name is not None:
            logger.next_iter_path(p_paths['rwd'][:, 0], p_paths['done'], policy_torch.entropy())

        rwd_filter.update(rwd)
        if norma == 'All':
            rwd = rwd_filter(rwd)

        # update policy and v
        adv = rl.get_adv(v_func=lambda obs: value_from_list(obs, reduce_fct=torch.max), obs=p_paths['obs'], rwd=rwd.numpy(), done=p_paths['done'], discount=discount, lam=lam)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        torch_adv = torch.tensor(adv, dtype=torch.float)
        # old_log_p = policy_torch.log_prob(obs, act).detach()
        old_pol_dist = policy_torch.distribution(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        v_target, _ = rl.get_targets(value_from_list, p_paths['obs'], rwd.numpy(), p_paths['done'], discount, lam)
        torch_targets = torch.tensor(v_target, dtype=torch.float)

        logging_ent = policy_torch.entropy()
        logging_verr = F.mse_loss(value_fct[0](obs), torch_targets)

        if cum_ts > max_ts:
            break
        # compute update filter, v_values and policy
        if norma == 'All' or norma == 'Obs':
            obs_filter.update(obs)
        for epoch in range(nb_epochs):
            for batch_idx in dat.next_batch_idx(h_layer_width, iter_ts):
                # update value network
                for kv in range(nb_vfunc):
                    value_optim[kv].zero_grad()
                    mse = F.mse_loss(value_fct[kv](obs[batch_idx]), torch_targets[batch_idx])
                    mse.backward()
                    value_optim[kv].step()

                # update policy
                p_optim.zero_grad()
                prob_ratio = torch.exp(policy_torch.log_prob(obs[batch_idx], act[batch_idx]) - old_log_p[batch_idx])
                clipped_ratio = torch.clamp(prob_ratio, 1 - eps_ppo, 1 + eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * torch_adv[batch_idx], clipped_ratio * torch_adv[batch_idx]))
                loss.backward()
                p_optim.step()

        new_pol_dist = policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl_divergence(new_pol_dist, old_pol_dist))
        iter += 1
        print("iter {}: rewards {} entropy {} vf_loss {} kl {}".format(iter, avg_rwd, logging_ent, logging_verr, logging_kl))


if __name__ == '__main__':
    # learn(envid='MountainCarContinuous-v0')
    # learn(envid='BipedalWalker-v2', max_ts=3e6, seed=0, norma='None', log_name='test_bip')
    # learn(envid='RoboschoolInvertedDoublePendulum-v1', max_ts=1e6, seed=0, norma='All', log_name='test_idp')
    # learn(envid='RoboschoolHalfCheetah-v1')
    # learn(envid='RoboschoolHopper-v1',  max_ts=3e6, seed=0, norma='None', log_name='test_hop')
    learn(envid='RoboschoolHumanoid-v1',  max_ts=1e7, seed=0, norma='None', log_name='test_hum')
    # learn(envid='RoboschoolAnt-v1', seed=0, norma='None')
