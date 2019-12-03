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


class Grad1Abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.abs()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = -grad_input[input < 0]
        return grad_input


class MetricRegressor(nn.Module):
    def __init__(self, input_shape, output_shape, n_clusters, std_0, **kwargs):
        super().__init__()

        s_dim = input_shape[0]
        a_dim = output_shape[0]
        self.centers = torch.zeros(n_clusters, s_dim)
        self.means = nn.Parameter(torch.zeros(n_clusters, a_dim))
        self._c_weights = nn.Parameter(torch.zeros(n_clusters))
        self._log_sigma = nn.Parameter(torch.ones(a_dim) * np.log(std_0))

        self._log_temp = torch.tensor(0.)

        self._cluster_count = 0
        self._n_clusters = n_clusters

    def forward(self, s):
        if self._cluster_count < self._n_clusters:
            self.centers[self._cluster_count] = s
            if self._cluster_count == 0:
                self._c_weights.data[0] = 1
            self._cluster_count += 1

        return self.get_mean(s), self.get_chol()

    def get_mean(self, s):
        if len(s.size()) == 1:
            s = s[None, :]
        w = self.get_membership(s)
        return w.matmul(self.means)

    def get_chol(self):
        return torch.diag(torch.exp(self._log_sigma))

    def set_chol(self, chol):
        log_sigma = torch.log(torch.diag(chol))
        self._log_sigma.data = log_sigma

    def get_c_weights(self):
        return Grad1Abs.apply(self._c_weights)

    def set_c_weights(self, c_weights):
        self._c_weights.data = c_weights

    def get_unormalized_membership(self, s, cweights=None):
        cweights = self.get_c_weights() if cweights is None else cweights
        return cweights * self._cluster_distance(s)

    def get_membership(self, s, cweights=None):
        cweights = self.get_c_weights() if cweights is None else cweights
        w = self.get_unormalized_membership(s, cweights)
        # We add 1 to weights norm to consider also the default cluster
        w_norm = torch.sum(w, dim=-1, keepdim=True) + 1
        return w / w_norm

    def _cluster_distance(self, s):
        dist = torch.sum((s[:, None, :] - self.centers[None, :, :]) ** 2, dim=-1)
        return torch.exp(-torch.exp(self._log_temp) * dist)


class MetricPolicy(TorchPolicy):
    def __init__(self, input_shape, output_shape, n_clusters, std_0, use_cuda=False):
        self._a_dim = output_shape[0]
        self._regressor = MetricRegressor(input_shape, output_shape, n_clusters, std_0)

        super().__init__(use_cuda)

        if self._use_cuda:
            self._regressor.cuda()

    def draw_action_t(self, state):
        return self.distribution_t(state).sample()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_prob(action)[:, None]

    def entropy_t(self, state):
        log_sigma = self._regressor._log_sigma
        return self._a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(log_sigma)

    def distribution_t(self, state):
        mu, chol_sigma = self._regressor(state)
        return torch.distributions.MultivariateNormal(mu, scale_tril=chol_sigma)

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def parameters(self):
        return self._regressor.parameters()

    def get_mean_t(self, s):
        return self._regressor.get_mean(s)

    def get_intermediate_mean_t(self, s, cmeans, cweights=None):
        new_membership = self._regressor.get_membership(s, cweights)
        return new_membership.matmul(cmeans)

    def get_chol_t(self):
        return self._regressor.get_chol()

    def set_chol_t(self, chol):
        self._regressor.set_chol(chol)

    def get_cweights_t(self):
        return self._regressor.get_c_weights()

    def set_cweights_t(self, cweights):
        self._regressor.set_c_weights(cweights)

    def get_cmeans_t(self):
        return self._regressor.means

    def set_cmeans_t(self, means):
        self._regressor.means.data = means

    def get_unormalized_membership_t(self, s):
        return self._regressor.get_unormalized_membership(s)

    def get_membership_t(self, s):
        return self._regressor.get_membership(s)


class ProjectionMetricRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs_per_fit, batch_size,
                 entropy_profile, max_kl, lam, critic_fit_params=None, del_x_iter=5):
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
        self._del_x_iter = del_x_iter
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
        v, adv = get_targets(self._critic, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        adv_t = to_float_tensor(adv, self._use_cuda)

        # Critic Update
        self._critic.fit(x, v, **self._critic_fit_params)

        # Save old data
        old_chol = self.policy.get_chol_t().detach()
        entropy_lb = self._e_profile.get_e_lb()

        old = dict(w=self.policy.get_unormalized_membership_t(obs).detach(),
                   cweights=self.policy.get_cweights_t().detach(),
                   cmeans=self.policy.get_cmeans_t().clone().detach(),
                   means=self.policy.get_mean_t(obs).detach(),
                   pol_dist=self.policy.distribution_t(obs),
                   log_p=self.policy.log_prob_t(obs, act).clone().detach(),
                   **utils_from_chol(old_chol))

        # add cluster centers
        self._add_cluster_centers(obs, act, adv_t)

        # Actor Update
        self._update_parameters(obs, act, adv_t, old, entropy_lb)
        self._full_batch_projection(obs, old, entropy_lb)

        if (self._iter + 0) % self._del_x_iter == 0:
            # delete cluster centers
            self._delete_cluster_centers(obs, old)

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

    def _delete_cluster_centers(self, obs, old):
        # deleting clusters
        avgm, order = torch.sort(torch.mean(self.policy.get_membership_t(obs), dim=0))
        nb_del = 0
        for k in order:
            # trying to delete cluster k
            init_weight = self.policy._regressor._c_weights[k].clone()
            self.policy._regressor._c_weights.data[k] = to_float_tensor(0., self._use_cuda)
            means = self.policy.get_mean_t(obs)
            if mean_diff(means, old['means'], old['prec']) > self._max_kl:
                self.policy._regressor._c_weights.data[k] = init_weight
            else:
                nb_del += 1
        tqdm.write('deleted {} clusters'.format(nb_del))

    def _update_parameters(self, obs, act, adv_t, old, entropy_lb):
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
