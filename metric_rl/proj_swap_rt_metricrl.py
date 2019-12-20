import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.replay_memory import ReplayMemory
from mushroom.utils.torch import to_float_tensor

from .cluster_weight_proj import cweight_mean_proj
from .gaussian_proj import lin_gauss_kl_proj, utils_from_chol, mean_diff
from .policies import MetricPolicy
from .cluster_randomized_optimization import randomized_swap_optimization


class ProjectionSwapRTMetricRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs_per_fit, batch_size,
                 initial_replay_size, max_replay_size, tau,
                 entropy_profile, max_kl, lam, n_samples=1000, no_swap_iteration=1000, critic_fit_params=None):
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
        self._tau = tau

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._target_critic = Regressor(TorchApproximator, **target_critic_params)
        self._critic = Regressor(TorchApproximator, **critic_params)

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        self._e_profile = entropy_profile['class'](policy, e_thresh=policy.entropy() / 2,
                                                   **entropy_profile['params'])
        self._max_kl = max_kl
        self._lambda = lam

        self._n_swaps = float(policy.n_clusters)
        self._n_samples = n_samples
        self._no_swap_iteration = no_swap_iteration


        self._iter = 0
        super().__init__(policy, mdp_info)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            self._fit_actor(state, action)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic.fit(state, action, q, **self._critic_fit_params)

    def _loss(self, state, action_new):
        q_0 = self._critic(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic(state, action_new, output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return -q.mean()

    def _next_q(self, next_state, absorbing):
        a = self.policy.draw_action(next_state)

        q = self._target_critic.predict(next_state, a, prediction='min')
        q *= 1 - absorbing

        return q

    def _fit_actor(self, x, u):
        # Get dataset
        x = x.astype(np.float32).squeeze()
        u = u.astype(np.float32).squeeze()

        # Get tensors
        obs = to_float_tensor(x, self._use_cuda)
        act = to_float_tensor(u,  self._use_cuda)

        q = self._critic.predict(x, u)
        q_t = to_float_tensor(q, self._use_cuda)

        # Save old data
        old_chol = self.policy.get_chol_t().detach()
        entropy_lb = self._e_profile.get_e_lb()

        # Add cluster actions for zero weighted clusters at first iter (should become obsolete later)
        if self._iter == 0:
            self._add_cluster_centers(obs, act, q_t)

        old = dict(w=self.policy.get_unormalized_membership_t(obs).detach(),
                   cweights=self.policy.get_cweights_t().detach(),
                   cmeans=self.policy.get_cmeans_t().clone().detach(),
                   means=self.policy.get_mean_t(obs).detach(),
                   pol_dist=self.policy.distribution_t(obs),
                   log_p=self.policy.log_prob_t(obs, act).clone().detach(),
                   membership=self.policy.get_membership_t(obs).detach(),
                   **utils_from_chol(old_chol))

        # optimize cw, mean and cov
        if self._iter % self._no_swap_iteration:
            self._update_all_parameters(obs, old, entropy_lb)

        # swap clusters and optimize mean and cov
        else:
            self._random_swap_clusters(obs, old)
            self._update_mean_n_cov(obs, old, entropy_lb)

        # logging
        if self._iter % self._no_swap_iteration == 0:
            logging_kl = torch.mean(
                torch.distributions.kl.kl_divergence(self.policy.distribution_t(obs), old['pol_dist']))
            mean_covr = torch.mean(torch.sum(self.policy.get_membership_t(obs), dim=1))
            tqdm.write('KL {} Covr {} iter {}'.format(logging_kl, mean_covr, self._iter))

        # next iter
        self._iter += 1

    def _add_cluster_centers(self, obs, act, q_t):
        # adding clusters
        sadv, oadv = torch.sort(q_t, dim=0)
        ba_ind = 0
        for k, cw in enumerate(self.policy._regressor._c_weights):
            if cw == 0.:
                ba = oadv[ba_ind].item()
                self.policy._regressor.centers[k] = obs[ba]
                self.policy._regressor.means.data[k] = act[ba]
                ba_ind += 1
        tqdm.write('added {} clusters'.format(ba_ind))

    def _update_mean_n_cov(self, obs, old, entropy_lb):
        # Compute mean projection (nu)
        intermediate_means = self.policy.get_mean_t(obs).detach()
        for epoch in range(self._n_epochs_per_fit):
            if mean_diff(intermediate_means, old['means'], old['prec']) < self._max_kl - 1e-6:
                for opt in self._actor_optimizers[1:]:
                    opt.zero_grad()

                # Get Basics stuff
                means_i = self.policy.get_mean_t(obs)
                chol = self.policy.get_chol_t()

                proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means, old['means'],
                                           old['cov'], old['prec'], old['logdetcov'],
                                           self._max_kl, entropy_lb)
                proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                # Compute loss
                new_act = proj_distrib.rsample()
                loss = self._loss(obs, new_act)
                loss.backward()

                for opt in self._actor_optimizers[1:]:
                    opt.step()

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

    def _update_all_parameters(self, obs, old, entropy_lb):
        for epoch in range(self._n_epochs_per_fit):
            for opt in self._actor_optimizers:
                opt.zero_grad()

            # Get Basics stuff
            w = self.policy.get_unormalized_membership_t(obs)
            means_i = self.policy.get_intermediate_mean_t(obs, old['cmeans'])
            chol = self.policy.get_chol_t()

            # Compute cluster weights projection (eta)
            eta = cweight_mean_proj(w, means_i, old['w'], old['means'], old['prec'], self._max_kl)
            cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * old['cweights']

            # Compute mean projection (nu)
            intermediate_means_i = self.policy.get_intermediate_mean_t(obs, old['cmeans'], cweights_eta)
            means_i = self.policy.get_mean_t(obs)

            proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old['means'],
                                       old['cov'], old['prec'], old['logdetcov'],
                                       self._max_kl, entropy_lb)
            proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

            # Compute loss
            new_act = proj_distrib.rsample()
            loss = self._loss(obs, new_act)
            loss.backward()

            for opt in self._actor_optimizers:
                opt.step()

    def _update_target(self):
        """
        Update the target networks.

        """
        for i in range(len(self._target_critic)):
            critic_weights_i = self._tau * self._critic.model[i].get_weights()
            critic_weights_i += (1 - self._tau) * self._target_critic.model[i].get_weights()
            self._target_critic.model[i].set_weights(critic_weights_i)
