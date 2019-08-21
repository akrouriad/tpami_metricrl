import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.minibatches import minibatch_generator

from .cluster_weight_proj import cweight_mean_proj
from .policies import PyTorchPolicy, MetricPolicy
from .gaussian_proj import mean_diff, lin_gauss_kl_proj, utils_from_chol
from .rl_shared import MLP, get_targets, get_adv, TwoPhaseEntropProfile


class ProjectionMetricRLOld(Agent):
    def __init__(self, mdp_info, std_0, lr_v, lr_p, lr_cw, max_kl, max_kl_cw, max_kl_cdel, e_reduc, n_max_clusters,
                 n_epochs_v, n_models_v, v_prediction_type, n_epochs_clus, n_epochs_params, batch_size, lam):

        self._n_max_clusters = n_max_clusters
        self._n_epochs_clus = n_epochs_clus
        self._n_epochs_params = n_epochs_params
        self._n_epochs_v = n_epochs_v
        self._batch_size = batch_size

        self._policy_torch = MetricPolicy(mdp_info.action_space.shape[0], std_0=std_0)

        self._cw_optim = torch.optim.Adam(self._policy_torch.cweights_list.parameters(), lr=lr_cw)
        self._p_optim = torch.optim.Adam([par for par in self._policy_torch.means_list.parameters()] +
                                         [self._policy_torch.logsigs], lr=lr_p)
        self._e_profile = TwoPhaseEntropProfile(self._policy_torch, e_reduc=e_reduc,
                                                e_thresh=self._policy_torch.entropy() / 2)

        self._max_kl = max_kl
        self._max_kl_cw = max_kl_cw
        self._max_kl_cdel = max_kl_cdel

        self._lambda = lam

        h_layer_width = 64
        h_layer_length = 2
        input_shape = mdp_info.observation_space.shape
        size_list = [h_layer_width] * h_layer_length
        approximator_params = dict(network=MLP,
                                   optimizer={'class': optim.Adam,
                                              'params': {'lr': lr_v}},
                                   loss=F.mse_loss,
                                   batch_size=h_layer_width,
                                   input_shape=input_shape,
                                   output_shape=(1,),
                                   size_list=size_list,
                                   activation_list=None,
                                   preproc=None,
                                   n_models=n_models_v,
                                   prediction=v_prediction_type,
                                   quiet=False)

        self._V = Regressor(TorchApproximator, **approximator_params)

        self._iter = 1

        policy = PyTorchPolicy(self._policy_torch)

        super().__init__(policy, mdp_info, None)

    def _add_new_clusters(self, obs, act, adv, wq, old_cweights, old_cmeans):
        nb_cluster = len(self._policy_torch.cweights)
        if nb_cluster < self._n_max_clusters:
            _, indices = torch.topk(adv, self._n_max_clusters - nb_cluster, dim=0)
            for index in indices:
                new_mean = np.clip(act[[index]], -1, 1)
                self._policy_torch.add_cluster(obs[[index]], new_mean)
                self._cw_optim.add_param_group({"params": [self._policy_torch.cweights_list[-1]]})
                self._p_optim.add_param_group({"params": [self._policy_torch.means_list[-1]]})
                tqdm.write('--> adding new cluster. Adv {}, cluster count {}'.format(adv[index].item(),
                                                                                     len(self._policy_torch.cweights)))
                old_cweights = torch.cat([old_cweights, torch.tensor([0.])])
                old_cmeans = torch.cat([old_cmeans, new_mean])
                wq = torch.cat([wq, torch.zeros(wq.size()[0], 1)], dim=1)
                nb_cluster += 1

            tqdm.write('nb cluster is ' + str(nb_cluster))
        return nb_cluster, wq, old_cweights, old_cmeans

    def _update_cluster_weights(self, obs, act, wq, old_means, old_log_p, adv, old_cov_d, old_cweights):
        for epoch in range(self._n_epochs_clus):
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
        init_kl = mean_diff(means, old_means, old_cov_d['prec'])
        # eta = ls_cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], kl_cluster, cmeans=old_cmeans)
        eta = cweight_mean_proj(w, means, wq, old_means, old_cov_d['prec'], self._max_kl_cw)
        weta = eta * w + (1. - eta) * wq
        weta /= torch.sum(weta, dim=1, keepdim=True) + 1  # !
        final_kl = mean_diff(weta.mm(self._policy_torch.get_cmeans_params()), old_means, old_cov_d['prec'])
        cweights = eta * torch.abs(self._policy_torch.cweights) + (1 - eta) * torch.abs(old_cweights)
        self._policy_torch.set_cweights_param(cweights)

        return init_kl, eta, final_kl

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
                    if mean_diff(means, old_means, old_cov_d['prec']) < self._max_kl_cdel and old_cweights[k] > 0.:
                        nb_cluster -= 1
                        deleted_clu.append(k)
                        self._policy_torch.zero_cweight_param(k)
                    else:
                        self._policy_torch.cweights[k] = init_weight
            tqdm.write('deleted {} clusters'.format(len(self._policy_torch.cweights) - nb_cluster))
        return deleted_clu

    def _update_mean_and_covariance(self, obs, act, adv, intermediate_means, old_means, old_cov_d, old_log_p, e_lb):
        for epoch in range(self._n_epochs_params):
            for obs_i, act_i, adv_i, intermediate_means_i, old_means_i, old_log_p_i in \
                    minibatch_generator(self._batch_size, obs, act, adv, intermediate_means, old_means, old_log_p):

                if mean_diff(intermediate_means_i, old_means_i,
                                  old_cov_d['prec']) < self._max_kl - 1e-6:
                    self._p_optim.zero_grad()
                    means = self._policy_torch.get_weighted_means(obs_i)
                    chol = self._policy_torch.get_chol()
                    proj_d = lin_gauss_kl_proj(means, chol, intermediate_means_i, old_means_i,
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
        proj_d = lin_gauss_kl_proj(means, chol, intermediate_means, old_means,
                                        old_cov_d['cov'], old_cov_d['prec'], old_cov_d['logdetcov'], self._max_kl, e_lb)
        cmeans = proj_d['eta_mean'] * self._policy_torch.get_cmeans_params() + (1 - proj_d['eta_mean']) * old_cmeans

        self._policy_torch.set_cmeans_param(cmeans)
        self._policy_torch.logsigs.data = torch.log(torch.diag(proj_d['chol']))

        return proj_d

    def fit(self, dataset):
        tqdm.write('Iteration ' + str(self._iter))
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
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
        old_cov_d = utils_from_chol(old_chol)
        e_lb = self._e_profile.get_e_lb()

        nb_cluster, wq, old_cweights, old_cmeans = self._add_new_clusters(obs, act, adv, wq, old_cweights, old_cmeans)

        np_v_target, _ = get_targets(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        self._V.fit(x, np_v_target, n_epochs=self._n_epochs_v)

        self._update_cluster_weights(obs, act, wq, old_means, old_log_p, adv, old_cov_d, old_cweights)

        init_kl, eta, final_kl = self._project_cluster_weights(obs, old_means, old_cov_d, wq, old_cweights)

        deleted_clu = self._delete_clusters(obs, nb_cluster, old_means, old_cov_d, old_cweights)

        new_pol_dist = self._policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
        tqdm.write('Cluster weights projection:\n\t init_kl {}, eta {}, proj kl {}, kl after del {}'.format(
            init_kl, eta, final_kl, logging_kl))

        intermediate_means = self._policy_torch.get_weighted_means(obs).detach()
        self._update_mean_and_covariance(obs, act, adv, intermediate_means, old_means, old_cov_d, old_log_p, e_lb)

        proj_d = self._project_mean_and_covariance(obs, old_means, intermediate_means, old_cov_d, old_cmeans, e_lb)
        tqdm.write('Mean and Covariance projection:\n\t init_kl {}, final_kl {}, eta_mean {}, eta_cov {}'.format(
            proj_d['init_kl'], proj_d['final_kl'], proj_d['eta_mean'], proj_d['eta_cov']))

        self._policy_torch.delete_clusters(deleted_clu)

        self._iter += 1

        # Print fit information
        logging_verr = []
        v_targets = torch.tensor(np_v_target, dtype=torch.float)
        for idx in range(len(self._V)):
            v_pred = torch.tensor(self._V(x, idx=idx), dtype=torch.float)
            v_err = F.mse_loss(v_pred, v_targets)
            logging_verr.append(v_err.item())

        logging_ent = self._policy_torch.entropy()
        new_pol_dist = self._policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
        avg_rwd = np.sum(r) / np.sum(last)
        tqdm.write("Iterations Results:\n\trewards {} vf_loss {}\n\tentropy {}  kl {} e_lb {}".format(
            avg_rwd, logging_verr, logging_ent, logging_kl, e_lb))
        avgm = torch.mean(self._policy_torch.membership(obs), dim=0)
        tqdm.write('avg membership ' +
                   str(torch.sort(avgm[avgm > torch.max(avgm) / 100], descending=True)[0].detach().numpy()))
        tqdm.write('avg membership of first ten clusters ' +
                   str(torch.sum(torch.sort(avgm, descending=True)[0][:10]).detach().numpy()))
        tqdm.write('cweights ' + str(self._policy_torch.cweights.detach().numpy()))
        tqdm.write('--------------------------------------------------------------------------------------------------')

