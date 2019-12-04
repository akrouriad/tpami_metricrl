import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from mushroom.algorithms.agent import Agent
from mushroom.policy import TorchPolicy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.torch import to_float_tensor
from mushroom.utils.minibatches import minibatch_generator

from .cluster_weight_proj import cweight_mean_proj
from .gaussian_proj import lin_gauss_kl_proj, utils_from_chol, mean_diff
from .rl_shared import get_targets
from .proj_metricrl import MetricPolicy


class ProjectionSwapMetricRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs_per_fit, batch_size,
                 entropy_profile, max_kl, lam, critic_fit_params=None):
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        policy = MetricPolicy(mdp_info.observation_space.shape,
                              mdp_info.action_space.shape,
                              **policy_params)

        self._use_cuda = policy_params['use_cuda'] if 'use_cuda' in policy_params else False

        self._actor_optimizers = [actor_optimizer['class']([policy._regressor._c_weights], **actor_optimizer['cw_params']),
                                  actor_optimizer['class']([policy._regressor.means], **actor_optimizer['means_params']),
                                  actor_optimizer['class']([policy._regressor._log_sigma], **actor_optimizer['log_sigma_params'])]

        self._n_epochs_per_fit = n_epochs_per_fit
        self._batch_size = batch_size

        self._critic = Regressor(TorchApproximator, **critic_params)

        self._e_profile = entropy_profile['class'](policy, e_thresh=policy.entropy() / 2,
                                                   **entropy_profile['params'])
        self._max_kl = max_kl
        self._lambda = lam

        self._iter = 0
        super().__init__(policy, mdp_info)

    def fit(self, dataset):
        # Get dataset
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        # Get tensors
        obs = to_float_tensor(x, self._use_cuda)
        act = to_float_tensor(u,  self._use_cuda)
        v, adv = get_targets(self._critic, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda,
                             prediction='min')
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        adv_t = to_float_tensor(adv, self._use_cuda)

        # Critic Update
        self._critic.fit(x, v, **self._critic_fit_params)

        # Save old data
        old_chol = self.policy.get_chol_t().detach()
        entropy_lb = self._e_profile.get_e_lb()

        # Add cluster actions for zero weighted clusters at first iter (should become obsolete later)
        if self._iter == 0:
            self._add_cluster_centers(obs, act, adv_t)

        old = dict(w=self.policy.get_unormalized_membership_t(obs).detach(),
                   cweights=self.policy.get_cweights_t().detach(),
                   cmeans=self.policy.get_cmeans_t().clone().detach(),
                   means=self.policy.get_mean_t(obs).detach(),
                   pol_dist=self.policy.distribution_t(obs),
                   log_p=self.policy.log_prob_t(obs, act).clone().detach(),
                   **utils_from_chol(old_chol))

        # optimize cw, mean and cov
        if self._iter % 2:
            self._update_all_parameters(obs, act, adv_t, old, entropy_lb)

        # swap clusters and optimize mean and cov
        else:
            self._swap_clusters(obs, act, adv_t, old)
            self._update_mean_n_cov(obs, act, adv_t, old, entropy_lb)

        # Actor Update
        self._full_batch_projection(obs, old, entropy_lb)

        # next iter
        self._iter += 1

    def _add_cluster_centers(self, obs, act, adv_t):
        # adding clusters
        sadv, oadv = torch.sort(adv_t, dim=0)
        ba_ind = 0
        for k, cw in enumerate(self.policy._regressor._c_weights):
            if cw == 0.:
                ba = oadv[ba_ind][0]
                self.policy._regressor.centers[k] = obs[ba]
                self.policy._regressor.means.data[k] = act[ba]
                ba_ind += 1
        tqdm.write('added {} clusters'.format(ba_ind))

    def _update_mean_n_cov(self, obs, act, adv_t, old, entropy_lb):
        # Compute mean projection (nu)
        intermediate_means = self.policy.get_mean_t(obs).detach()
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, adv_i, intermediate_means_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], adv_t, intermediate_means):

                if mean_diff(intermediate_means_i, old_means_i, old['prec']) < self._max_kl - 1e-6:
                    for opt in self._actor_optimizers[1:]:
                        opt.zero_grad()

                    # Get Basics stuff
                    means_i = self.policy.get_mean_t(obs_i)
                    chol = self.policy.get_chol_t()

                    proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                               old['cov'], old['prec'], old['logdetcov'],
                                               self._max_kl, entropy_lb)
                    proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                    # Compute loss
                    prob_ratio = torch.exp(proj_distrib.log_prob(act_i)[:, None] - old_log_p_i)
                    loss = -torch.mean(prob_ratio * adv_i)
                    loss.backward()

                    for opt in self._actor_optimizers[1:]:
                        opt.step()

    def _swap_clusters(self, obs, act, adv_t, old):
        base_perf = torch.mean(torch.exp(old['pol_dist'].log_prob(act)[:, None] - old['log_p']) * adv_t)  # almost zero

        high_adv_idxs = torch.argsort(adv_t, dim=0, descending=True)
        remaining_idx = [k for k in range(self.policy._regressor._n_clusters)]
        state_centers = self.policy._regressor.centers.clone().numpy()
        for idxidx, idx in enumerate(high_adv_idxs):
            # finding closest state
            dist_to_centers = -np.sum((state_centers - obs[idx].detach().numpy()) ** 2, axis=1)
            closest_idxs = np.argsort(dist_to_centers)
            for closest_idx in closest_idxs:
                closest = remaining_idx[closest_idx]
                state_backup = self.policy._regressor.centers[closest].clone()
                # swap and check KL
                self.policy._regressor.centers[closest] = obs[idx]
                kl_swap = mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec'])
                if kl_swap < self._max_kl:
                    # check increase of the objective after state swapping
                    # ... and cluster center swapping?
                    action_backup = self.policy._regressor.means[closest].clone()
                    self.policy._regressor.means.data[closest] = act[idx]
                    new_pol_dist = self.policy.distribution_t(obs)
                    new_perf = torch.mean(torch.exp(new_pol_dist.log_prob(act)[:, None] - old['log_p']) * adv_t)
                    self.policy._regressor.means.data[closest] = action_backup
                    if new_perf >= base_perf:
                    # if True:
                        # keep if kl under thresh and objective did not decrease
                        remaining_idx.pop(closest_idx)
                        state_centers = np.delete(state_centers, closest_idx, axis=0)
                        break
                    else:
                        # revert old cluster center
                        self.policy._regressor.centers[closest] = state_backup
                else:
                    # revert old cluster center
                    self.policy._regressor.centers[closest] = state_backup
            if len(remaining_idx) == 0:
                break
        new_pol_dist = self.policy.distribution_t(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old['pol_dist']))
        print('Nb swapped clusters: {}. KL: {}'.format(self.policy._regressor._n_clusters - len(remaining_idx), logging_kl))

    def _update_all_parameters(self, obs, act, adv_t, old, entropy_lb):
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, adv_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], adv_t):
                for opt in self._actor_optimizers:
                    opt.zero_grad()

                # Get Basics stuff
                w = self.policy.get_unormalized_membership_t(obs_i)
                means_i = self.policy.get_mean_t(obs_i)
                chol = self.policy.get_chol_t()

                # Compute cluster weights projection (eta)
                eta = cweight_mean_proj(w, means_i, wq_i, old_means_i, old['prec'], self._max_kl)
                cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * old['cweights']

                # Compute mean projection (nu)
                intermediate_means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'], cweights_eta).detach()

                proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                           old['cov'], old['prec'], old['logdetcov'],
                                           self._max_kl, entropy_lb)
                proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                # Compute loss
                prob_ratio = torch.exp(proj_distrib.log_prob(act_i)[:, None] - old_log_p_i)
                loss = -torch.mean(prob_ratio * adv_i)
                loss.backward()

                for opt in self._actor_optimizers:
                    opt.step()

    def _full_batch_projection(self, obs, old, entropy_lb):
        # Get Basics stuff
        w = self.policy.get_unormalized_membership_t(obs)
        means = self.policy.get_mean_t(obs)
        chol = self.policy.get_chol_t()

        # Compute cluster weights projection (eta)
        eta = cweight_mean_proj(w, means, old['w'], old['means'], old['prec'], self._max_kl)
        cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * torch.abs(old['cweights'])
        self.policy.set_cweights_t(cweights_eta)

        # Compute mean projection (nu)
        intermediate_means = self.policy.get_intermediate_mean_t(obs, old['cmeans'])

        proj_d = lin_gauss_kl_proj(means, chol, intermediate_means, old['means'],
                                   old['cov'], old['prec'], old['logdetcov'], self._max_kl, entropy_lb)
        nu = proj_d['eta_mean']
        cmeans_nu = nu * self.policy.get_cmeans_t() + (1 - nu) * old['cmeans']

        self.policy.set_cmeans_t(cmeans_nu)
        self.policy.set_chol_t(proj_d['chol'])
