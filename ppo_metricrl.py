import torch
import torch.nn.functional as F
import numpy as np
import gym
import roboschool
import data_handling as dat
import rllog
from rl_shared import MLP, RunningMeanStdFilter, ValueFunction, ValueFunctionList
import rl_shared as rl
from policies import MetricPolicy
import gaussian_proj as proj

def learn(envid, nb_vfunc=2, seed=0, max_ts=1e6, norma='None', log_name=None, aggreg_type='Min', min_sample_per_iter=3000):
    print('Metric RL')
    print('Params: nb_vfunc {} norma {} aggreg_type {} max_ts {} seed {} log_name {}'.format(nb_vfunc, norma, aggreg_type, max_ts, seed, log_name))
    env = gym.make(envid)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    h_layer_width = 64
    h_layer_length = 2
    lr_v = 3e-4
    lr_p = 1e-3
    nb_epochs_v = 10
    nb_epochs_p = 10
    max_kl = .034

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    obs_filter = RunningMeanStdFilter(s_dim, min_clamp=-5, max_clamp=5)
    rwd_filter = RunningMeanStdFilter(1, min_clamp=-5, max_clamp=5, center=False)

    input_sizes = [s_dim] + [h_layer_width] * h_layer_length

    value_mlp, value_fct, value_optim = [], [], []
    for k in range(nb_vfunc):
        value_mlp.append(MLP(input_sizes + [1], preproc=obs_filter))
        value_fct.append(ValueFunction(value_mlp[k]))
        value_optim.append(torch.optim.Adam(value_fct[k].parameters(), lr=lr_v))

    value_from_list = ValueFunctionList(value_fct)

    policy_torch = MetricPolicy(a_dim)
    policy = lambda obs: torch.squeeze(policy_torch(torch.tensor(obs, dtype=torch.float)), dim=0).detach().numpy()
    p_optim = torch.optim.Adam(policy_torch.parameters(), lr=lr_p)

    discount = .99
    lam = .95

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
        if aggreg_type == 'None':
            adv = rl.get_adv(v_func=value_fct[0], obs=p_paths['obs'], rwd=rwd.numpy(), done=p_paths['done'],
                             discount=discount, lam=lam)
        else:
            aggreg_dic = {'Max': torch.max, 'Min': torch.min, 'Median': torch.median, 'Mean': torch.mean}
            adv = rl.get_adv(v_func=lambda obs: value_from_list(obs, reduce_fct=aggreg_dic[aggreg_type]), obs=p_paths['obs'],
                             rwd=rwd.numpy(), done=p_paths['done'], discount=discount, lam=lam)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        torch_adv = torch.tensor(adv, dtype=torch.float)
        old_chol = policy_torch.get_chol()
        old_means = policy_torch.get_weighted_means(obs)
        old_pol_dist = policy_torch.distribution(obs)
        old_log_p = policy_torch.log_prob(obs, act).detach()
        index = np.argmax(adv)
        if iter % 3 == 0:
        # if adv[index] > 1.5:
            # add new cluster
            policy_torch.add_cluster(obs[[index]], act[[index]])
            p_optim.add_param_group({"params": [policy_torch.rootweights_list[-1], policy_torch.means_list[-1]]})
            print('--> adding new cluster. Adv {}, cluster count {}'.format(adv[index], len(policy_torch.rootweights)))
            klcluster = torch.mean(torch.distributions.kl_divergence(policy_torch.distribution(obs), old_pol_dist))
            print('--> KL after adding cluster', klcluster)


        v_target, _ = rl.get_targets(value_from_list, p_paths['obs'], rwd.numpy(), p_paths['done'], discount, lam)
        torch_targets = torch.tensor(v_target, dtype=torch.float)

        logging_ent = policy_torch.entropy()
        logging_verr = F.mse_loss(value_fct[0](obs), torch_targets)

        if cum_ts > max_ts:
            break
        # compute update filter, v_values and policy
        if norma == 'All' or norma == 'Obs':
            obs_filter.update(obs)
        for epoch in range(max(nb_epochs_v, nb_epochs_p)):
            for batch_idx in dat.next_batch_idx(h_layer_width, iter_ts):
                # update value network
                if epoch < nb_epochs_v:
                    for kv in range(nb_vfunc):
                        value_optim[kv].zero_grad()
                        mse = F.mse_loss(value_fct[kv](obs[batch_idx]), torch_targets[batch_idx])
                        mse.backward()
                        value_optim[kv].step()

                # update policy
                if epoch < nb_epochs_p:
                    p_optim.zero_grad()
                    prob_ratio = torch.exp(policy_torch.log_prob(obs[batch_idx], act[batch_idx]) - old_log_p[batch_idx])
                    eps_ppo = .2
                    clipped_ratio = torch.clamp(prob_ratio, 1 - eps_ppo, 1 + eps_ppo)
                    loss = -torch.mean(torch.min(prob_ratio * torch_adv[batch_idx], clipped_ratio * torch_adv[batch_idx]))
                    loss.backward()
                    p_optim.step()
                    policy_torch.cat_params()

        new_pol_dist = policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl_divergence(new_pol_dist, old_pol_dist))
        iter += 1
        print("iter {}: rewards {} entropy {} vf_loss {} kl {}".format(iter, avg_rwd, logging_ent, logging_verr, logging_kl))
        # print(policy_torch.rootweights)
        # print(torch.exp(policy_torch.logtemp))
        avgm = torch.mean(policy_torch.membership(obs), dim=0)
        # print('avg membership', avgm)
        print('avg membership', avgm[avgm > torch.max(avgm) / 100])
        chol_u = proj.utils_from_chol(old_chol)
        proj_u = proj.gauss_kl_proj(policy_torch.get_weighted_means(obs), policy_torch.get_chol(), old_means, chol_u['cov'], chol_u['prec'], chol_u['logdetcov'], .01)
        print('init_kl {}, final_kl {}, eta_mean {}, eta_cov {}'.format(proj_u['init_kl'], proj_u['final_kl'], proj_u['eta_mean'], proj_u['eta_cov']))


if __name__ == '__main__':
    # learn(envid='MountainCarContinuous-v0')
    learn(envid='BipedalWalker-v2', max_ts=3e6, seed=0, norma='None', log_name='test_bip')
    # learn(envid='RoboschoolInvertedDoublePendulum-v1', max_ts=1e6, seed=0, norma='All', log_name='test_idp')
    # learn(envid='RoboschoolHalfCheetah-v1')
    # learn(envid='RoboschoolHopper-v1',  max_ts=3e6, seed=0, norma='None', log_name='test_hop')
    # learn(envid='RoboschoolHumanoid-v1',  nb_vfunc=2, max_ts=1e7, seed=0, norma='None', log_name='test_hum', aggreg_type='Max')
    # learn(envid='RoboschoolAnt-v1', seed=0, norma='None')

