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
from .gaussian_proj import mean_diff, lin_gauss_kl_proj, utils_from_chol
from .rl_shared import get_targets, get_adv, TwoPhaseEntropProfile


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
        s_dim = input_shape[0]
        a_dim = output_shape[0]
        self.centers = torch.zeros(n_clusters, s_dim)
        self.c_weights = nn.Parameter(torch.zeros(n_clusters))
        self.means = nn.Parameter(torch.zeros(n_clusters, a_dim))
        self.log_temp = torch.tensor(0.)

        self._log_sigma = nn.Parameter(torch.ones(a_dim) * np.log(std_0))

        self._cluster_count = 0
        self._n_clusters = n_clusters

        super().__init__()

    def forward(self, s):
        if self._cluster_count < self._n_clusters:
            self.centers[self._cluster_count] = s
            self._cluster_count += 1

        return self.get_mean(s), self.get_chol()

    def get_mean(self, s):
        if len(s.size()) == 1:
            s = s[None, :]
        w = self.get_membership(s)
        return w.matmul(self.means)

    def get_chol(self):
        return torch.diag(torch.exp(self.logsigs))

    def set_chol(self, chol):
        log_sigma = torch.log(torch.diag(chol))
        self._log_sigma.data = log_sigma

    def get_cweights(self):
        return Grad1Abs.apply(self.cweights)

    def set_cweights(self, cweights):
        self.cweights = cweights

    def get_unormalized_membership(self, s):
        return self.get_cweights() * self._cluster_distance(s)

    def get_membership(self, s):
        w = self.unormalized_membership(s)
        return w / (torch.sum(w, dim=-1, keepdim=True) + 1) #!

    def _cluster_distance(self, s):
        dist = torch.sum((s[:, None, :] - self.centers[None, :, :]) ** 2, dim=-1)
        return torch.exp(-torch.exp(self.logtemp) * dist)


class MetricPolicy(TorchPolicy):
    def __init__(self, input_shape, output_shape, use_cuda, n_clusters, std_0):
        self._a_dim = output_shape[0]
        self._regressor = MetricRegressor(input_shape, output_shape, n_clusters, std_0)

        super.__init__(use_cuda)

        if self._use_cuda:
            self._regressor.cuda()

    def draw_action_t(self, state):
        raise self.distribution_t(state).sample()

    def log_prob_t(self, state, action):
        return self.distribution_t(state).log_pdf(action)

    def entropy_t(self, state):
        log_sigma = self._regressor.log_sigma
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

    def get_intermediate_mean_t(self, obs, cmeans):
        new_membership = self.policy.get_membership_t(obs).detach()
        return new_membership.matmul(cmeans)

    def get_chol_t(self):
        return self._regressor.get_chol()

    def set_chol_t(self, chol):
        log_sigma = torch.log(torch.diag(chol))
        self._regressor.set_chol(chol)

    def get_cweights_t(self):
        return self._regressor.get_cweights()

    def set_cweights_t(self, cweights):
        self._regressor.set_cweights(cweights)

    def get_cmeans_t(self):
        return self._regressor.means

    def set_cmeans_t(self, means):
        self._regressor.means = means

    def get_unormalized_membership_t(self, s):
        return self._regressor.get_unormalized_membership(s)

    def get_membership_t(self, s):
        return self._regressor.get_membership(s)


class ProjectionMetricRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs, batch_size,
                 e_profile, max_kl=.001, lam=1.,
                 critic_fit_params=None):
        self._critic_fit_params = dict(n_epochs=3) if critic_fit_params is None else critic_fit_params

        policy = MetricPolicy(mdp_info.observation_space.shape,
                              mdp_info.action_space.shape,
                              **policy_params)

        self._use_cuda = policy_params['use_cuda'] if 'use_cuda' in policy_params else False

        self._optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])

        self._n_epochs = n_epochs
        self._batch_size = batch_size

        self._critic = Regressor(TorchApproximator, **critic_params)

        self._e_profile = e_profile
        self._max_kl = max_kl
        self._lambda = lam

        super.__init__(mdp_info, policy)

    def fit(self, dataset):
        tqdm.write('Iteration ' + str(self._iter))
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float3)

        obs = to_float_tensor(x, self._use_cuda)
        act = to_float_tensor(u,  self._use_cuda)
        v, adv = get_targets(self._critic, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        adv = - np.mean() / (np.std() + 1e-8)
        adv_t = to_float_tensor(adv, self._use_cuda)

        # Critic Update
        self._critic.fit(x, v, **self._critic_fit_params)

        # Save old data
        old_chol = self.policy.get_chol().detach()
        entropy_lb = self._e_profile.get_e_lb()

        old = dict(w=self.policy.unormalized_membership(obs).detach(),
                   cweights=self.policy.get_cweights_t().detach(),
                   cmeans=self.policy.get_cmeans_t().detach(),
                   means=self.policy.get_mean_t(obs).detach(),
                   pol_dist=self.policy.distribution_t(obs),
                   log_p=self.policy.log_prob_t(obs, act).detach(),
                   **utils_from_chol(old_chol))

        # Actor Update
        self._update_parameters(obs, act, adv_t, old, entropy_lb)
        self._full_batch_projection(self, obs, old, entropy_lb)

    def _update_parameters(self, obs, act, adv_t, old, entropy_lb):
        for epoch in range(self._n_epochs):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, adv_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], adv_t):
                self._optimizer.zero_grad()

                # Get Basics stuff
                w = self.policy.get_unormalized_membership(obs_i)
                means_i = self.policy.get_mean_t(obs_i)
                chol = self.policy.get_chol()

                # Compute cluster weights projection (eta)
                eta = cweight_mean_proj(w, means_i, wq_i, old_means_i, old['prec'], self._max_kl_cw)
                cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * old['cweights']
                self.policy.set_cweights(cweights_eta)

                # Compute mean projection (nu)
                intermediate_means_i = self.policy.get_intermediate_mean(obs_i, old['cmeans'])

                proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                           old['cov'], old['prec'], old['logdetcov'],
                                           self._max_kl, entropy_lb)
                proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                # Compute loss
                prob_ratio = torch.exp(proj_distrib.log_prob(act_i)[:, None] - old_log_p_i)
                loss = -torch.mean(prob_ratio * adv_i)
                loss.backward()

    def _full_batch_projection(self, obs, old, entropy_lb):
        # Get Basics stuff
        w = self.policy.get_unormalized_membership_t(obs)
        means = self.policy.get_mean_t(obs)
        chol = self.policy.get_chol()

        # Compute cluster weights projection (eta)
        eta = cweight_mean_proj(w, means, old['w'], old['means'], old['prec'], self._max_kl_cw)
        weta = eta * w + (1. - eta) * old['w']
        weta /= torch.sum(weta, dim=1, keepdim=True) + 1  # !
        cweights_eta = eta * self.policy.get_cweights_t + (1 - eta) * torch.abs(old['cweights'])
        self.policy.set_cweights_t(cweights_eta)

        # Compute mean projection (nu)
        intermediate_means = self.policy.get_intermediate_mean(obs, old['cmeans'])

        proj_d = lin_gauss_kl_proj(means, chol, intermediate_means, old['means'],
                                   old['cov'], old['prec'], old['logdetcov'], self._max_kl, entropy_lb)
        nu = proj_d['eta_mean']
        cmeans_nu = nu * self.policy.get_cmeans_params() + (1 - nu) * old['cmeans']

        self.policy.set_cmeans(cmeans_nu)
        self.policy.set_chol_t(proj_d['chol'])