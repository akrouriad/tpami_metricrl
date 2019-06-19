import numpy as np
import torch

from mushroom.algorithms.agent import Agent
from mushroom.policy import Policy
from mushroom.utils.dataset import parse_dataset

from cluster_weight_proj import cweight_mean_proj
import gaussian_proj as proj


class TwoPhaseEntropProfile:
    def __init__(self, policy, e_reduc, e_thresh):
        self.init_entropy = policy.entropy()
        self._policy = policy
        self._e_reduc = e_reduc
        self._e_thresh = e_thresh
        self._phase = 1
        self._iter = 0

    def get_e_lb(self):
        if self._phase == 1 and self._policy.entropy() > self._e_thresh:
            return -10000.
        else:
            self._phase = 2
            self._iter += 1
            return self._e_thresh - self._iter * self._e_reduc


class TmpPolicy(Policy):

    def __init__(self, network):
        self._network = network

    def __call__(self, *args):
        raise NotImplementedError

    def draw_action(self, state):
        return torch.squeeze(self._network(torch.tensor(state, dtype=torch.float)), dim=0).detach().numpy()

    def reset(self):
        pass


class ProjectionMetricRL(Agent):
    def __init__(self, policy_torch, mdp_info, lr_p, lr_cw, e_reduc, nb_max_clusters, nb_epochs_clus, nb_epochs_params):

        self._nb_max_clusters = nb_max_clusters
        self._nb_epochs_clus = nb_epochs_clus
        self._nb_epochs_params = nb_epochs_params

        self._policy_torch = policy_torch

        self._cw_optim = torch.optim.Adam(policy_torch.cweights_list.parameters(), lr=lr_cw)
        self._p_optim = torch.optim.Adam([par for par in policy_torch.means_list.parameters()] + [policy_torch.logsigs],
                                   lr=lr_p)
        self._e_profile = TwoPhaseEntropProfile(policy_torch, e_reduc=e_reduc, e_thresh=policy_torch.entropy() / 2)

        self._max_kl = .015
        self._max_kl_cw = self._max_kl / 2.
        self._max_kl_cdel = 2 * self._max_kl / 3.

        self._lambda = 0.95

        policy = TmpPolicy(policy_torch)

        super().__init__(policy, mdp_info, None)

    def fit(self, dataset):
        x, u, r, xn, ab, last = parse_dataset(dataset)


        iter_ts = len(p_paths['rwd'])
        cum_ts += iter_ts
        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
        rwd = torch.tensor(r, dtype=torch.float)
        avg_rwd = np.sum(r) / np.sum(last)  # TODO check this

        # if log_name is not None:
        #     logger.next_iter_path(p_paths['rwd'][:, 0], p_paths['done'], self._policy_torch.entropy())

        # update policy and v
        if aggreg_type == 'None':
            np_adv = rl.get_adv(v_func=value_fct[0], obs=p_paths['obs'], rwd=rwd.numpy(), done=p_paths['done'],
                                discount=discount, lam=lam)
        else:
            aggreg_dic = {'Max': torch.max, 'Min': torch.min, 'Median': torch.median, 'Mean': torch.mean}
            np_adv = rl.get_adv(v_func=lambda obs: value_from_list(obs, reduce_fct=aggreg_dic[aggreg_type]),
                                obs=p_paths['obs'],
                                rwd=rwd.numpy(), done=p_paths['done'], discount=self.mdp.info.gamma, lam=self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = torch.tensor(np_adv, dtype=torch.float)

        old_chol = self._policy_torch.get_chol().detach()
        wq = self._policy_torch.unormalized_membership(obs).detach()
        old_cweights = self._policy_torch.cweights.detach()
        old_cmeans = self._policy_torch.get_cmeans_params().detach()
        old_means = self._policy_torch.get_weighted_means(obs).detach()
        old_pol_dist = self._policy_torch.distribution(obs)
        old_log_p = self._policy_torch.log_prob(obs, act).detach()
        old_cov_d = proj.utils_from_chol(old_chol)
        e_lb = self._e_profile.get_e_lb()

        nb_cluster = len(self._policy_torch.cweights)
        deleted_clu = []
        if nb_cluster < self._nb_max_clusters:
            # adding new cluster
            _, indices = torch.topk(adv, self._nb_max_clusters - nb_cluster, dim=0)
            for index in indices:
                new_mean = np.clip(act[[index]], -1, 1)
                self._policy_torch.add_cluster(obs[[index]], new_mean)
                self._cw_optim.add_param_group({"params": [self._policy_torch.cweights_list[-1]]})
                self._p_optim.add_param_group({"params": [self._policy_torch.means_list[-1]]})
                print('--> adding new cluster. Adv {}, cluster count {}'.format(np_adv[index],
                                                                                len(self._policy_torch.cweights)))
                klcluster = torch.mean(torch.distributions.kl_divergence(self._policy_torch.distribution(obs), old_pol_dist))
                # print('--> KL after adding cluster', klcluster)
                old_cweights = torch.cat([old_cweights, torch.tensor([0.])])
                old_cmeans = torch.cat([old_cmeans, new_mean])
                wq = torch.cat([wq, torch.zeros(wq.size()[0], 1)], dim=1)
                nb_cluster += 1
            print('nb cluster is', nb_cluster)

        np_v_target, _ = rl.get_targets(value_from_list, p_paths['obs'], rwd.numpy(), p_paths['done'], discount, lam)
        v_targets = torch.tensor(np_v_target, dtype=torch.float)

        logging_ent = self._policy_torch.entropy()
        logging_verr = F.mse_loss(value_fct[0](obs), v_targets)

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

        # update cluster weights
        for epoch in range(self._nb_epochs_clus):
            for batch_idx in dat.next_batch_idx(batch_size_pupdate, iter_ts):
                # batch_idx = range(rwd.size()[0])
                self._cw_optim.zero_grad()
                w = self._policy_torch.unormalized_membership(obs[batch_idx])
                means = self._policy_torch.get_weighted_means(obs[batch_idx])
                # eta = ls_cweight_mean_proj(w, means, wq[batch_idx], old_means[batch_idx], old_cov_d['prec'], kl_cluster, cmeans=old_cmeans)
                eta = cweight_mean_proj(w, means, wq[batch_idx], old_means[batch_idx], old_cov_d['prec'], max_kl_cw)
                self._policy_torch.cweights = eta * self._policy_torch.cweights + (1 - eta) * old_cweights
                prob_ratio = torch.exp(self._policy_torch.log_prob(obs[batch_idx], act[batch_idx]) - old_log_p[batch_idx])
                loss = -torch.mean(prob_ratio * adv[batch_idx])
                loss.backward()
                self._cw_optim.step()
                self._policy_torch.update_clustering()

        # overriding the cluster weights with the projected ones
        w = self._policy_torch.unormalized_membership(obs)
        means = self._policy_torch.get_weighted_means(obs)
        init_kl = proj.mean_diff(means, old_means, old_cov_d['prec'])
        # eta = ls_cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], kl_cluster, cmeans=old_cmeans)
        eta = cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], self._max_kl_cw)
        weta = eta * w + (1. - eta) * wq
        weta /= torch.sum(weta, dim=1, keepdim=True) + 1  # !
        final_kl = proj.mean_diff(weta.mm(self._policy_torch.get_cmeans_params()), old_means, old_cov_d['prec'])
        cweights = eta * torch.abs(self._policy_torch.cweights) + (1 - eta) * torch.abs(old_cweights)
        self._policy_torch.set_cweights_param(cweights)

        # deleting clusters
        if iter % 5 == 0:
            avgm, order = torch.sort(torch.mean(self._policy_torch.membership(obs), dim=0))
            for k in order:
                if nb_cluster > 1:
                    # trying to delete cluster k
                    init_weight = self._policy_torch.cweights[k].clone()
                    self._policy_torch.cweights[k] = 0.
                    means = self._policy_torch.get_weighted_means(obs)
                    if proj.mean_diff(means, old_means, old_cov_d['prec']) < self._max_kl_cdel and old_cweights[k] > 0.:
                        nb_cluster -= 1
                        deleted_clu.append(k)
                        self._policy_torch.zero_cweight_param(k)
                    else:
                        self._policy_torch.cweights[k] = init_weight
            print('deleted {} clusters'.format(len(self._policy_torch.cweights) - nb_cluster))

        new_pol_dist = self._policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
        print('init_kl {}, eta {}, proj kl {}, kl after del {}'.format(init_kl, eta, final_kl, logging_kl))
        # if logging_kl > max_kl:
        #     print('y')
        #     eta = cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], max_kl)

        # update sub-policies means and cov
        intermediate_means = self._policy_torch.get_weighted_means(obs).detach()
        for epoch in range(self._nb_epochs_params):
            for batch_idx in dat.next_batch_idx(batch_size_pupdate, iter_ts):
                # batch_idx = range(rwd.size()[0])
                if proj.mean_diff(intermediate_means[batch_idx], old_means[batch_idx],
                                  old_cov_d['prec']) < self._max_kl - 1e-6:
                    self._p_optim.zero_grad()
                    means = self._policy_torch.get_weighted_means(obs[batch_idx])
                    chol = self._policy_torch.get_chol()
                    proj_d = proj.lin_gauss_kl_proj(means, chol, intermediate_means[batch_idx], old_means[batch_idx],
                                                    old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'],
                                                    self._max_kl, e_lb)
                    proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])
                    prob_ratio = torch.exp(proj_distrib.log_prob(act[batch_idx])[:, None] - old_log_p[batch_idx])
                    loss = -torch.mean(prob_ratio * adv[batch_idx])
                    loss.backward()
                    self._p_optim.step()
                    self._policy_torch.update_clustering()

        # overriding means and chol with projected ones
        means = self._policy_torch.get_weighted_means(obs)
        chol = self._policy_torch.get_chol()
        proj_d = proj.lin_gauss_kl_proj(means, chol, intermediate_means, old_means,
                                        old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'], self._max_kl, e_lb)
        cmeans = proj_d['eta_mean'] * self._policy_torch.get_cmeans_params() + (1 - proj_d['eta_mean']) * old_cmeans

        self._policy_torch.set_cmeans_param(cmeans)
        self._policy_torch.logsigs.data = torch.log(torch.diag(proj_d['chol']))

        # print('init_kl {}, final_kl {}, eta_mean {}, eta_cov {}'.format(proj_d['init_kl'], proj_d['final_kl'],
        #                                                                 proj_d['eta_mean'], proj_d['eta_cov']))
        #
        # self._policy_torch.delete_clusters(deleted_clu)
        # new_pol_dist = self._policy_torch.distribution(obs)
        # logging_kl = torch.mean(torch.distributions.kl_divergence(new_pol_dist, old_pol_dist))
        # iter += 1
        # print("iter {}: rewards {} entropy {} vf_loss {} kl {} e_lb {}".format(iter, avg_rwd, logging_ent, logging_verr,
        #                                                                        logging_kl, e_lb))
        # print(self._policy_torch.rootweights)
        # print(torch.exp(self._policy_torch.logtemp))
        # avgm = torch.mean(self._policy_torch.membership(obs), dim=0)
        # print('avg membership', avgm)
        # print('avg membership', torch.sort(avgm[avgm > torch.max(avgm) / 100], descending=True)[0].detach().numpy())
        # print('avg membership of first ten clusters',
        #       torch.sum(torch.sort(avgm, descending=True)[0][:10]).detach().numpy())
        # print('cweights', self._policy_torch.cweights.detach().numpy())
        # print('-------------------------------------------------------------------------------------------------------')
        # print('iter time', time.time() - iter_start)
        # if log_name is not None:
        #     policy_saver.next_iter()
