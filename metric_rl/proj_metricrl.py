import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from mushroom.algorithms.agent import Agent
from mushroom.policy import TorchPolicy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import TorchApproximator
from mushroom.utils.dataset import parse_dataset
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
        self.log_sigma = nn.Parameter(torch.ones(a_dim)*np.log(std_0))
        self.log_temp = torch.tensor(0.)

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
        w = self.membership(s)
        return w.matmul(self.get_cmeans_params())

    def get_chol(self):
        return torch.diag(torch.exp(self.logsigs))

    def get_cweights(self):
        return Grad1Abs.apply(self.cweights)

    def _cluster_distance(self, s):
        dist = torch.sum((s[:, None, :] - self.centers[None, :, :]) ** 2, dim=-1)
        return torch.exp(-torch.exp(self.logtemp) * dist)

    def _unormalized_membership(self, s):
        return self.get_cweights() * self._cluster_distance(s)

    def _membership(self, s):
        w = self.unormalized_membership(s)
        return w / (torch.sum(w, dim=-1, keepdim=True) + 1) #!


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


class ProjectionMetricRL(Agent):
    def __init__(self, mdp_info, policy_params, policy_optimizer,
                 critic_params, n_epochs, batch_size,
                 e_reduc, max_kl=.001, lam=1.,
                 critic_fit_params=None):
        policy = MetricPolicy(mdp_info.observation_space.shape,
                              mdp_info.action_space.shape,
                              **policy_params)
        super.__init__(mdp_info, policy)

    def fit(self, dataset):
        pass
