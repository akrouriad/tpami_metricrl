import torch
import torch.nn as nn
import numpy as np

from mushroom_rl.policy import TorchPolicy
from mushroom_rl.utils.torch import to_float_tensor
from metric_rl.projections.gaussian import project_entrop


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
    def __init__(self, input_shape, output_shape, n_clusters, std_0, temp=1., w_default=1., learnable_centers=False,
                 init_cluster_noise=1e-2, **kwargs):
        super().__init__()

        s_dim = input_shape[0]
        a_dim = output_shape[0]
        if learnable_centers:
            self.centers = nn.Parameter(torch.zeros(n_clusters, s_dim))
            self._c_weights = nn.Parameter(torch.ones(n_clusters))
        else:
            self.centers = torch.zeros(n_clusters, s_dim)
            self._c_weights = nn.Parameter(torch.zeros(n_clusters))
        self.means = nn.Parameter(torch.zeros(n_clusters, a_dim))
        self._log_sigma = nn.Parameter(torch.ones(a_dim) * np.log(std_0))

        # self._log_temp = nn.Parameter(torch.log(torch.tensor(temp)) * torch.ones_like(self._c_weights))
        self._log_temp = nn.Parameter(torch.log(torch.tensor(temp)))

        self._n_clusters = n_clusters
        self._cluster_count = 0
        self.w_default = w_default
        self.e_lb = None
        self._init_cluster_noise = init_cluster_noise

    def forward(self, s):
        if self._cluster_count < self._n_clusters:
            if self.centers.requires_grad:
                for k in range(self._n_clusters):
                    print(self._init_cluster_noise)
                    self.centers.data[k] = s + torch.randn_like(s) * self._init_cluster_noise
                self._cluster_count = self._n_clusters
            else:
                self.centers[self._cluster_count] = s
                if self._cluster_count == 0:
                    self._c_weights.data[0] = 1
                self._cluster_count += 1

        return self.get_mean(s), self.get_chol()

    def get_mean(self, s):
        if len(s.size()) == 1:
            s = s[None, :]
        w = self.get_membership(s)
        return w.matmul(self.get_cmeans())

    def get_cmeans(self):
        return self.means

    def get_chol(self):
        if self.e_lb is None:
            return torch.diag(torch.exp(self._log_sigma))
        else:
            return project_entrop(torch.diag(torch.exp(self._log_sigma)), self.e_lb)

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
        w_norm = torch.sum(w, dim=-1, keepdim=True) + self.w_default
        return w / w_norm

    def _cluster_distance(self, s):
        dist = torch.sum((s[:, None, :] - self.centers[None, :, :]) ** 2, dim=-1)
        return torch.exp(-torch.exp(self._log_temp) * dist)

    def to_clust_dist(self, s):
        return torch.norm(s[:, None, :] - self.centers[None, :, :], dim=-1)

    @property
    def n_clusters(self):
        return len(self.centers)


class MetricPolicy(TorchPolicy):
    def __init__(self, input_shape, output_shape, n_clusters, std_0, temp=1., use_cuda=False, learnable_centers=False,
                 init_cluster_noise=1e-2):
        self._a_dim = output_shape[0]
        self._regressor = MetricRegressor(input_shape, output_shape, n_clusters, std_0, temp=temp,
                                          learnable_centers=learnable_centers, init_cluster_noise=init_cluster_noise)

        super().__init__(use_cuda)

        if self._use_cuda:
            self._regressor.cuda()

        self._add_save_attr(_a_dim='primitive', _regressor='torch')

    def draw_action_t(self, state):
        return self.distribution_t(state).sample()

    def log_prob_t(self, state, action, action_raw=None):
        return self.distribution_t(state).log_prob(action)[:, None]

    def entropy_t(self, state):
        log_sigma = torch.log(torch.diag(self._regressor.get_chol()))
        return self._a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(log_sigma)

    def distribution_t(self, state):
        mu, chol_sigma = self._regressor(state)
        return torch.distributions.MultivariateNormal(mu, scale_tril=chol_sigma)

    def get_weights(self):
        weights = list()
        for p in self.parameters():
            w = p.data.detach().cpu().numpy()
            weights.append(w.flatten())
        weights = np.concatenate(weights, 0)
        return weights

    def set_weights(self, weights):
        idx = 0
        for p in self.parameters():
            shape = p.data.shape
            c = 1
            for s in shape:
                c *= s
            w = np.reshape(weights[idx:idx + c], shape)
            w_tensor = torch.tensor(w).type(p.data.dtype)
            p.data = w_tensor
            idx += c
        assert idx == weights.size

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
        return self._regressor.get_cmeans()

    def set_cmeans_t(self, means):
        self._regressor.means.data = means

    def get_unormalized_membership_t(self, s):
        return self._regressor.get_unormalized_membership(s)

    def get_membership_t(self, s):
        return self._regressor.get_membership(s)

    def get_cluster_centers(self):
        return self._regressor.centers.detach().numpy()

    def set_cluster_centers(self, centers):
        self._regressor.centers.data = to_float_tensor(centers, self._use_cuda)

    @property
    def n_clusters(self):
        return self._regressor.n_clusters

    @property
    def w_default(self):
        return self._regressor.w_default
