import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.agent import Agent
from mushroom.policy import Policy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.minibatches import minibatch_generator

from cluster_weight_proj import cweight_mean_proj
from policies import MetricPolicy
import gaussian_proj as proj


def get_targets(v_func, x, x_n, rwd, absorbing, last, discount, lam):
    v = v_func(x)
    v_next = v_func(x_n)
    gen_adv = np.empty_like(v)
    for rev_k, _ in enumerate(reversed(v)):
        k = len(v) - rev_k - 1
        if last[k] or rev_k == 0:
            gen_adv[k] = rwd[k] - v[k]
            if not absorbing[k]:
                gen_adv[k] += discount * v_next[k]
        else:
            gen_adv[k] = rwd[k] + discount * v_next[k] - v[k] + discount * lam * gen_adv[k + 1]
    return gen_adv + v, gen_adv


def get_adv(v_func, x, xn,  rwd, absorbing, last, discount, lam):
    _, adv = get_targets(v_func, x, xn, rwd, absorbing, last, discount, lam)
    return adv


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, size_list, activation_list=None, preproc=None, **kwargs):
        super().__init__()
        self.size_list = [input_shape[0]] + size_list + [output_shape[0]]
        if not activation_list or activation_list is None:
            activation_list = [torch.tanh] * (len(self.size_list) - 2) + [None]
        self.activation_list = activation_list
        self.preproc = preproc
        self.layers = nn.ModuleList()
        for k, kp, activ in zip(self.size_list[:-1], self.size_list[1:], activation_list):
            self.layers.append(nn.Linear(k, kp))
            if activ is not None:
                nn.init.xavier_uniform_(self.layers[-1].weight, nn.init.calculate_gain(activ.__name__))
            else:
                nn.init.xavier_uniform_(self.layers[-1].weight)

    def forward(self, x):
        if self.preproc is not None:
            x = self.preproc(x)

        for l, a in zip(self.layers, self.activation_list):
            if a is not None:
                x = a(l(x))
            else:
                x = l(x)
        return x


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
    def __init__(self, mdp_info, std_0, lr_v, lr_p, lr_cw, max_kl, e_reduc, nb_max_clusters, nb_epochs_v,
                 nb_epochs_clus, nb_epochs_params, batch_size):

        self._nb_max_clusters = nb_max_clusters
        self._nb_epochs_clus = nb_epochs_clus
        self._nb_epochs_params = nb_epochs_params
        self._nb_epochs_v = nb_epochs_v
        self._batch_size = batch_size

        self._policy_torch = MetricPolicy(mdp_info.action_space.shape[0], std_0=std_0)

        self._cw_optim = torch.optim.Adam(self._policy_torch.cweights_list.parameters(), lr=lr_cw)
        self._p_optim = torch.optim.Adam([par for par in self._policy_torch.means_list.parameters()] +
                                         [self._policy_torch.logsigs], lr=lr_p)
        self._e_profile = TwoPhaseEntropProfile(self._policy_torch, e_reduc=e_reduc,
                                                e_thresh=self._policy_torch.entropy() / 2)

        self._max_kl = max_kl
        self._max_kl_cw = self._max_kl / 2.
        self._max_kl_cdel = 2 * self._max_kl / 3.

        self._lambda = 0.95

        h_layer_width = 64
        h_layer_length = 2
        input_shape = mdp_info.observation_space.shape
        size_list = [h_layer_width] * h_layer_length
        approximator_params = dict(network=MLP,
                                   optimizer={'class': optim.Adam,
                                              'params': {'lr': lr_v}},
                                   loss=F.smooth_l1_loss,
                                   input_shape=input_shape,
                                   output_shape=(1,),
                                   size_list=size_list, activation_list=None, preproc=None)

        self._V = Regressor(PyTorchApproximator, **approximator_params)

        self._iter = 1

        policy = TmpPolicy(self._policy_torch)

        super().__init__(policy, mdp_info, None)

    def _add_new_clusters(self, obs, act, adv, wq, old_cweights, old_cmeans):
        nb_cluster = len(self._policy_torch.cweights)
        if nb_cluster < self._nb_max_clusters:
            _, indices = torch.topk(adv, self._nb_max_clusters - nb_cluster, dim=0)
            for index in indices:
                new_mean = np.clip(act[[index]], -1, 1)
                self._policy_torch.add_cluster(obs[[index]], new_mean)
                self._cw_optim.add_param_group({"params": [self._policy_torch.cweights_list[-1]]})
                self._p_optim.add_param_group({"params": [self._policy_torch.means_list[-1]]})
                tqdm.write('--> adding new cluster. Adv {}, cluster count {}'.format(adv[index][0],
                                                                                len(self._policy_torch.cweights)))
                old_cweights = torch.cat([old_cweights, torch.tensor([0.])])
                old_cmeans = torch.cat([old_cmeans, new_mean])
                wq = torch.cat([wq, torch.zeros(wq.size()[0], 1)], dim=1)
                nb_cluster += 1

            tqdm.write('nb cluster is ' + str(nb_cluster))
        return nb_cluster, wq, old_cweights, old_cmeans

    def _update_cluster_weights(self, obs, act, wq, old_means, old_log_p, adv, old_cov_d, old_cweights):
        for epoch in range(self._nb_epochs_clus):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, adv_i in \
                    minibatch_generator(self._batch_size, obs, act, wq, old_means, old_log_p, adv):
                self._cw_optim.zero_grad()
                w = self._policy_torch.unormalized_membership(obs_i)
                means = self._policy_torch.get_weighted_means(obs_i)
                eta = cweight_mean_proj(w, means, wq_i, old_means_i, old_cov_d['prec'], self._max_kl_cw)
                self._policy_torch.cweights = eta * self._policy_torch.cweights + (1 - eta) * old_cweights
                prob_ratio = torch.exp(self._policy_torch.log_prob(obs_i, act_i) - old_log_p_i)
                loss = -torch.mean(prob_ratio * adv_i)
                loss.backward()
                self._cw_optim.step()
                self._policy_torch.update_clustering()

    def _project_cluster_weights(self, obs, old_means, old_cov_d, wq, old_cweights):
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

    def _delete_clusters(self, obs, nb_cluster, old_means, old_cov_d, old_cweights):
        deleted_clu = []
        if self._iter % 5 == 0:
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
            tqdm.write('deleted {} clusters'.format(len(self._policy_torch.cweights) - nb_cluster))
        return deleted_clu

    def _update_mean_and_covariance(self, obs, act, adv, intermediate_means, old_means, old_cov_d, old_log_p, e_lb):
        for epoch in range(self._nb_epochs_params):
            for obs_i, act_i, adv_i, intermediate_means_i, old_means_i, old_log_p_i in \
                    minibatch_generator(self._batch_size, obs, act, adv, intermediate_means, old_means, old_log_p):

                if proj.mean_diff(intermediate_means_i, old_means_i,
                                  old_cov_d['prec']) < self._max_kl - 1e-6:
                    self._p_optim.zero_grad()
                    means = self._policy_torch.get_weighted_means(obs_i)
                    chol = self._policy_torch.get_chol()
                    proj_d = proj.lin_gauss_kl_proj(means, chol, intermediate_means_i, old_means_i,
                                                    old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'],
                                                    self._max_kl, e_lb)
                    proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])
                    prob_ratio = torch.exp(proj_distrib.log_prob(act_i)[:, None] - old_log_p_i)
                    loss = -torch.mean(prob_ratio * adv_i)
                    loss.backward()
                    self._p_optim.step()
                    self._policy_torch.update_clustering()

    def _project_mean_and_covariance(self, obs, old_means, intermediate_means, old_cov_d, old_cmeans, e_lb):
        means = self._policy_torch.get_weighted_means(obs)
        chol = self._policy_torch.get_chol()
        proj_d = proj.lin_gauss_kl_proj(means, chol, intermediate_means, old_means,
                                        old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'], self._max_kl, e_lb)
        cmeans = proj_d['eta_mean'] * self._policy_torch.get_cmeans_params() + (1 - proj_d['eta_mean']) * old_cmeans

        self._policy_torch.set_cmeans_param(cmeans)
        self._policy_torch.logsigs.data = torch.log(torch.diag(proj_d['chol']))

    def fit(self, dataset):
        tqdm.write('Iteration ' + str(self._iter))
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
        rwd = torch.tensor(r, dtype=torch.float)
        avg_rwd = np.sum(r) / np.sum(last)  # TODO check this
        np_adv = get_adv(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
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

        nb_cluster, wq, old_cweights, old_cmeans = self._add_new_clusters(obs, act, adv, wq, old_cweights, old_cmeans)

        np_v_target, _ = get_targets(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        self._V.fit(x, np_v_target, n_epochs=self._nb_epochs_v)

        self._update_cluster_weights(obs, act, wq, old_means, old_log_p, adv, old_cov_d, old_cweights)

        self._project_cluster_weights(obs, old_means, old_cov_d, wq, old_cweights)

        deleted_clu = self._delete_clusters(obs, nb_cluster, old_means, old_cov_d, old_cweights)

        intermediate_means = self._policy_torch.get_weighted_means(obs).detach()
        self._update_mean_and_covariance(obs, act, adv, intermediate_means, old_means, old_cov_d, old_log_p, e_lb)

        self._project_mean_and_covariance(obs, old_means, intermediate_means, old_cov_d, old_cmeans, e_lb)

        self._policy_torch.delete_clusters(deleted_clu)

        self._iter += 1

        # Print fit information
        avgm = torch.mean(self._policy_torch.membership(obs), dim=0)
        tqdm.write('avg membership ' +
                   str(torch.sort(avgm[avgm > torch.max(avgm) / 100], descending=True)[0].detach().numpy()))
        tqdm.write('avg membership of first ten clusters ' +
                   str(torch.sum(torch.sort(avgm, descending=True)[0][:10]).detach().numpy()))
        tqdm.write('cweights' + str(self._policy_torch.cweights.detach().numpy()))
        tqdm.write('-------------------------------------------------------------------------------------------------------')

