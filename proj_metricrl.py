import torch
import torch.nn.functional as F
import numpy as np
import gym
# import roboschool
import data_handling as dat
import rllog
from rl_shared import MLP, RunningMeanStdFilter, ValueFunction, ValueFunctionList
import rl_shared as rl
from policies import MetricPolicy
import gaussian_proj as proj
from cluster_weight_proj import cweight_mean_proj, ls_cweight_mean_proj


class TwoPhaseEntropProfile:
    def __init__(self, policy, e_reduc):
        self.init_entropy = policy.entropy()
        self._policy = policy
        self._e_reduc = e_reduc

    def get_e_lb(self):
        if self._policy.entropy() > self.init_entropy / 2:
            return -10000.
        else:
            return self._policy.entropy() - self._e_reduc



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
    batch_size_pupdate = 64
    nb_epochs_clus = 20
    nb_epochs_params = 20
    max_kl = .015
    kl_cluster = max_kl / 2.

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
    cw_optim = torch.optim.Adam(policy_torch.cweights_list.parameters(), lr=lr_p)
    p_optim = torch.optim.Adam([par for par in policy_torch.means_list.parameters()] + [policy_torch.logsigs], lr=lr_p)
    e_profile = TwoPhaseEntropProfile(policy_torch, max_kl)

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
            np_adv = rl.get_adv(v_func=value_fct[0], obs=p_paths['obs'], rwd=rwd.numpy(), done=p_paths['done'],
                             discount=discount, lam=lam)
        else:
            aggreg_dic = {'Max': torch.max, 'Min': torch.min, 'Median': torch.median, 'Mean': torch.mean}
            np_adv = rl.get_adv(v_func=lambda obs: value_from_list(obs, reduce_fct=aggreg_dic[aggreg_type]), obs=p_paths['obs'],
                             rwd=rwd.numpy(), done=p_paths['done'], discount=discount, lam=lam)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = torch.tensor(np_adv, dtype=torch.float)

        old_chol = policy_torch.get_chol().detach()
        wq = policy_torch.unormalized_membership(obs).detach()
        old_cweights = policy_torch.cweights.detach()
        old_cmeans = policy_torch.means.detach()
        old_means = policy_torch.get_weighted_means(obs).detach()
        old_pol_dist = policy_torch.distribution(obs)
        old_log_p = policy_torch.log_prob(obs, act).detach()
        e_lb = e_profile.get_e_lb()

        if iter % 3 == 0:
        # if adv[index] > 1.5:
            # add new cluster
            index = np.argmax(np_adv)
            policy_torch.add_cluster(obs[[index]], act[[index]])
            cw_optim.add_param_group({"params": [policy_torch.cweights_list[-1]]})
            p_optim.add_param_group({"params": [policy_torch.means_list[-1]]})
            print('--> adding new cluster. Adv {}, cluster count {}'.format(np_adv[index], len(policy_torch.cweights)))
            klcluster = torch.mean(torch.distributions.kl_divergence(policy_torch.distribution(obs), old_pol_dist))
            # print('--> KL after adding cluster', klcluster)
            old_cweights = torch.cat([old_cweights, torch.tensor([0.])])
            old_cmeans = torch.cat([old_cmeans, act[[index]]])
            wq = torch.cat([wq, torch.zeros(wq.size()[0], 1)], dim=1)

        np_v_target, _ = rl.get_targets(value_from_list, p_paths['obs'], rwd.numpy(), p_paths['done'], discount, lam)
        v_targets = torch.tensor(np_v_target, dtype=torch.float)

        logging_ent = policy_torch.entropy()
        logging_verr = F.mse_loss(value_fct[0](obs), v_targets)

        if cum_ts > max_ts:
            break
        # compute update filter, v_values and policy
        if norma == 'All' or norma == 'Obs':
            obs_filter.update(obs)
        for epoch in range(nb_epochs_v):
            for batch_idx in dat.next_batch_idx(h_layer_width, iter_ts):
                # update value network
                for kv in range(nb_vfunc):
                    value_optim[kv].zero_grad()
                    mse = F.mse_loss(value_fct[kv](obs[batch_idx]), v_targets[batch_idx])
                    mse.backward()
                    value_optim[kv].step()

        old_cov_d = proj.utils_from_chol(old_chol)

        # update cluster weights
        for epoch in range(nb_epochs_clus):
            for batch_idx in dat.next_batch_idx(batch_size_pupdate, iter_ts):
                # batch_idx = range(rwd.size()[0])
                cw_optim.zero_grad()
                w = policy_torch.unormalized_membership(obs[batch_idx])
                means = (w / torch.sum(w, dim=-1, keepdim=True)).mm(policy_torch.means)
                # eta = ls_cweight_mean_proj(w, means, wq[batch_idx], old_means[batch_idx], old_cov_d['prec'], kl_cluster, cmeans=old_cmeans)
                eta = cweight_mean_proj(w, means, wq[batch_idx], old_means[batch_idx], old_cov_d['prec'], kl_cluster)
                policy_torch.cweights = eta * policy_torch.cweights + (1 - eta) * old_cweights
                prob_ratio = torch.exp(policy_torch.log_prob(obs[batch_idx], act[batch_idx]) - old_log_p[batch_idx])
                loss = -torch.mean(prob_ratio * adv[batch_idx])
                loss.backward()
                cw_optim.step()
                policy_torch.update_clustering()

        # overriding the cluster weights with the projected ones
        w = policy_torch.unormalized_membership(obs)
        means = policy_torch.get_weighted_means(obs)
        init_kl = proj.mean_diff(means, old_means, old_cov_d['prec'])
        # eta = ls_cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], kl_cluster, cmeans=old_cmeans)
        eta = cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], kl_cluster)
        weta = eta * w + (1. - eta) * wq
        weta /= torch.sum(weta, dim=1, keepdim=True)
        final_kl = proj.mean_diff(weta.mm(policy_torch.means), old_means, old_cov_d['prec'])
        cweights = eta * torch.abs(policy_torch.cweights) + (1 - eta) * torch.abs(old_cweights)
        for k, cwp in enumerate(policy_torch.cweights_list):
            cwp.data = cweights[[k]]
        policy_torch.update_clustering()

        new_pol_dist = policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
        print('init_kl {}, eta {}, final kl {}, final kl torch {}'.format(init_kl, eta, final_kl, logging_kl))
        # if logging_kl > max_kl:
        #     print('y')
        #     eta = cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], max_kl)

        # update sub-policies means and cov
        intermediate_means = policy_torch.get_weighted_means(obs).detach()
        for epoch in range(nb_epochs_params):
            for batch_idx in dat.next_batch_idx(batch_size_pupdate, iter_ts):
                # batch_idx = range(rwd.size()[0])
                if proj.mean_diff(intermediate_means[batch_idx], old_means[batch_idx], old_cov_d['prec']) < max_kl - 1e-6:
                    p_optim.zero_grad()
                    means = policy_torch.get_weighted_means(obs[batch_idx])
                    chol = policy_torch.get_chol()
                    proj_d = proj.lin_gauss_kl_proj(means, chol, intermediate_means[batch_idx], old_means[batch_idx], old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'], max_kl, e_lb)
                    proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])
                    prob_ratio = torch.exp(proj_distrib.log_prob(act[batch_idx])[:, None] - old_log_p[batch_idx])
                    loss = -torch.mean(prob_ratio * adv[batch_idx])
                    loss.backward()
                    p_optim.step()
                    policy_torch.update_clustering()

        # overriding means and chol with projected ones
        means = policy_torch.get_weighted_means(obs)
        chol = policy_torch.get_chol()
        proj_d = proj.lin_gauss_kl_proj(means, chol, intermediate_means, old_means,
                                        old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'], max_kl, e_lb)
        cmeans = proj_d['eta_mean'] * policy_torch.means + (1 - proj_d['eta_mean']) * old_cmeans
        # if proj_d['eta_mean'] < 1.:
        #     print('eta_mean', proj_d['eta_mean'])
        #     proj.lin_gauss_kl_proj(means, chol, intermediate_means, old_means,
        #                            old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'], max_kl)

        for k, cmp in enumerate(policy_torch.means_list):
            cmp.data = cmeans[[k]]
        policy_torch.logsigs.data = torch.log(torch.diag(proj_d['chol']))
        policy_torch.update_clustering()

        print('init_kl {}, final_kl {}, eta_mean {}, eta_cov {}'.format(proj_d['init_kl'], proj_d['final_kl'], proj_d['eta_mean'], proj_d['eta_cov']))

        new_pol_dist = policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl_divergence(new_pol_dist, old_pol_dist))
        iter += 1
        print("iter {}: rewards {} entropy {} vf_loss {} kl {} e_lb {}".format(iter, avg_rwd, logging_ent, logging_verr, logging_kl, e_lb))
        # print(policy_torch.rootweights)
        # print(torch.exp(policy_torch.logtemp))
        avgm = torch.mean(policy_torch.membership(obs), dim=0)
        # print('avg membership', avgm)
        print('avg membership', torch.sort(avgm[avgm > torch.max(avgm) / 100], descending=True)[0].detach().numpy())
        print('avg membership of first ten clusters', torch.sum(torch.sort(avgm, descending=True)[0][:10]).detach().numpy())


if __name__ == '__main__':
    # learn(envid='MountainCarContinuous-v0')
    learn(envid='BipedalWalker-v2', max_ts=3e6, seed=0, norma='None', log_name='exp_bip')
    # learn(envid='RoboschoolInvertedDoublePendulum-v1', max_ts=1e6, seed=0, norma='All', log_name='test_idp')
    # learn(envid='RoboschoolHalfCheetah-v1')
    # learn(envid='RoboschoolHopper-v1',  max_ts=3e6, seed=0, norma='None', log_name='test_hop')
    # learn(envid='RoboschoolHumanoid-v1',  nb_vfunc=2, max_ts=1e7, seed=0, norma='None', log_name='test_hum', aggreg_type='Max')
    # learn(envid='RoboschoolAnt-v1', seed=0, norma='None')

