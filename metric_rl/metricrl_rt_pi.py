import numpy as np
from tqdm import tqdm

import torch

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator

from .cluster_weight_proj import cweight_mean_proj
from .gaussian_proj import lin_gauss_kl_proj, utils_from_chol, mean_diff
from .rl_shared import get_targets
from .policies import MetricPolicy
from .cluster_randomized_optimization import randomized_swap_optimization


class ProjectionSwapMetricQRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs_per_fit, batch_size,
                 entropy_profile, max_kl, lam, n_samples=1000, critic_fit_params=None):
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

        critic_params['quiet'] = False
        self._critic = Regressor(TorchApproximator, **critic_params)

        self._e_profile = entropy_profile['class'](policy, e_thresh=policy.entropy() / 2,
                                                   **entropy_profile['params'])
        self._max_kl = max_kl
        self._lambda = lam

        self._n_swaps = float(policy.n_clusters)
        self._n_samples = n_samples

        self._iter = 0
        super().__init__(mdp_info, policy)

    def _get_v_from_q(self, x, nb_v_avr=10):
        og_shape = x.shape[0]
        x = np.repeat(x, nb_v_avr, axis=0)
        v = self._critic(x, self.policy.draw_action(x), prediction='min')
        v = v.reshape(og_shape, nb_v_avr)
        return np.mean(v, axis=1, keepdims=True)

    def fit(self, dataset):
        # Get dataset
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)[:, None]
        xn = xn.astype(np.float32)

        # Get tensors
        obs = to_float_tensor(x, self._use_cuda)
        act = to_float_tensor(u,  self._use_cuda)

        v_next = self._get_v_from_q(xn)
        q_targ = r + self.mdp_info.gamma * (1 - absorbing[:, None]) * v_next

        # Critic Update
        self._critic.fit(x, u, q_targ, **self._critic_fit_params)

        # Save old data
        old_chol = self.policy.get_chol_t().detach()
        entropy_lb = self._e_profile.get_e_lb()

        # Add cluster actions for zero weighted clusters at first iter (should become obsolete later)
        if self._iter == 0:
            self._add_cluster_centers(obs, act, to_float_tensor(self._critic(x, u, prediction='min') - self._get_v_from_q(x), use_cuda=self._use_cuda))

        old = dict(w=self.policy.get_unormalized_membership_t(obs).detach(),
                   cweights=self.policy.get_cweights_t().detach(),
                   cmeans=self.policy.get_cmeans_t().clone().detach(),
                   means=self.policy.get_mean_t(obs).detach(),
                   pol_dist=self.policy.distribution_t(obs),
                   log_p=self.policy.log_prob_t(obs, act).clone().detach(),
                   membership=self.policy.get_membership_t(obs).detach(),
                   **utils_from_chol(old_chol))

        v = to_float_tensor(self._get_v_from_q(x), use_cuda=self._use_cuda)

        # optimize cw, mean and cov
        if self._iter % 2:
            self._update_all_parameters(obs, act, v, old, entropy_lb)

        # swap clusters and optimize mean and cov
        else:
            self._random_swap_clusters(obs, old)
            self._update_mean_n_cov(obs, act, v, old, entropy_lb)

        # Actor Update
        self._full_batch_projection(obs, old, entropy_lb)

        # next iter
        self._iter += 1

        # logging
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(self.policy.distribution_t(obs), old['pol_dist']))
        mean_covr = torch.mean(torch.sum(self.policy.get_membership_t(obs), dim=1))
        tqdm.write('KL {} Covr {}'.format(logging_kl, mean_covr))

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

    def _update_mean_n_cov(self, obs, act, v, old, entropy_lb):
        # Compute mean projection (nu)
        intermediate_means = self.policy.get_mean_t(obs).detach()
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, v_i, intermediate_means_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], v, intermediate_means):

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
                    loss = self._loss(obs_i, proj_distrib.rsample(), v_i)
                    loss.backward()

                    for opt in self._actor_optimizers[1:]:
                        opt.step()

    def _loss(self, x, u, v):
        q_0 = self._critic(x, u, output_tensor=True, idx=0)
        q_1 = self._critic(x, u, output_tensor=True, idx=1)

        adv = torch.min(q_0, q_1)[:, None] - v
        return -adv.mean()

    def _random_swap_clusters(self, obs, old):
        candidates = obs.detach().numpy()
        c_0 = self.policy.get_cluster_centers()

        w = self.policy.get_membership_t(obs)

        cluster_h = torch.sum(w, dim=0)
        cluster_h = cluster_h.detach().numpy().squeeze()

        sample_h = torch.sum(w, dim=1)
        sample_h = sample_h.detach().numpy().squeeze()

        def bound_function(c_i):
            c_old = self.policy.get_cluster_centers()
            self.policy.set_cluster_centers(c_i)
            kl_swap = mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec'])
            self.policy.set_cluster_centers(c_old)
            return kl_swap < self._max_kl

        def evaluation_function(c_i, *args):
            c_old = self.policy.get_cluster_centers()
            self.policy.set_cluster_centers(c_i)
            w = self.policy.get_membership_t(obs)
            self.policy.set_cluster_centers(c_old)
            return torch.sum(w).item()

        swapped = False
        while True:
            c_best = randomized_swap_optimization(c_0, candidates, cluster_h, sample_h,
                                                  bound_function, evaluation_function,
                                                  int(np.ceil(self._n_swaps)), self._n_samples)

            if np.array_equal(c_best, c_0):
                self._n_swaps /= 2
                if self._n_swaps < 1.0:
                    self._n_swaps = 1.0
                    break
            else:
                swapped = True
                break

        if swapped:
            self.policy.set_cluster_centers(c_best)

            new_pol_dist = self.policy.distribution_t(obs)
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old['pol_dist']))
            print('Nb swapped clusters: {}. KL: {}'.format(int(np.ceil(self._n_swaps)), logging_kl))

            self._n_swaps *= 1.8
            self._n_swaps = np.minimum(self._n_swaps, self.policy.n_clusters)
        else:
            print('Nb swapped clusters: 0. KL: 0.0')

    def _update_all_parameters(self, obs, act, v, old, entropy_lb):
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, v_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], v):
                for opt in self._actor_optimizers:
                    opt.zero_grad()

                # Get Basics stuff
                w = self.policy.get_unormalized_membership_t(obs_i)
                means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'])
                chol = self.policy.get_chol_t()

                # Compute cluster weights projection (eta)
                eta = cweight_mean_proj(w, means_i, wq_i, old_means_i, old['prec'], self._max_kl)
                cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * old['cweights']

                # Compute mean projection (nu)
                intermediate_means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'], cweights_eta)
                means_i = self.policy.get_mean_t(obs_i)

                proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                           old['cov'], old['prec'], old['logdetcov'],
                                           self._max_kl, entropy_lb)
                proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                # Compute loss
                loss = self._loss(obs_i, proj_distrib.rsample(), v_i)
                loss.backward()

                for opt in self._actor_optimizers:
                    opt.step()

    def _full_batch_projection(self, obs, old, entropy_lb):
        # Get Basics stuff
        w = self.policy.get_unormalized_membership_t(obs)
        means = self.policy.get_intermediate_mean_t(obs, old['cmeans'])
        chol = self.policy.get_chol_t()

        # Compute cluster weights projection (eta)
        eta = cweight_mean_proj(w, means, old['w'], old['means'], old['prec'], self._max_kl)
        cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * torch.abs(old['cweights'])
        self.policy.set_cweights_t(cweights_eta)

        # Compute mean projection (nu)
        intermediate_means = self.policy.get_intermediate_mean_t(obs, old['cmeans'])
        means = self.policy.get_mean_t(obs)

        proj_d = lin_gauss_kl_proj(means, chol, intermediate_means, old['means'],
                                   old['cov'], old['prec'], old['logdetcov'], self._max_kl, entropy_lb)
        nu = proj_d['eta_mean']
        cmeans_nu = nu * self.policy.get_cmeans_t() + (1 - nu) * old['cmeans']

        self.policy.set_cmeans_t(cmeans_nu)
        self.policy.set_chol_t(proj_d['chol'])
