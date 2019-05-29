import numpy as np
import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):
    def __init__(self, mean_func, a_dim, mean_mul=1.):
        super().__init__()
        self._mean_func = mean_func
        self.a_dim = a_dim
        self._mean_mul = mean_mul
        self.log_sigma = nn.Parameter(torch.zeros(a_dim))

    def get_mean(self, x):
        return self._mean_func(x) * self._mean_mul

    def forward(self, x):
        with torch.no_grad():
            return self.distribution(x).sample().detach()

    def log_prob(self, x, y):
        return self.distribution(x).log_prob(y)[:, None]

    def entropy(self):
        return self.a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self.log_sigma).detach().numpy()

    def distribution(self, x):
        cov = torch.diag(torch.exp(2 * self.log_sigma))
        return torch.distributions.MultivariateNormal(loc=self.get_mean(x), covariance_matrix=cov)


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


class MetricPolicy(nn.Module):
    def __init__(self, a_dim):
        super().__init__()
        self.a_dim = a_dim
        self.centers = None
        # self.rootweights_list = nn.ParameterList().append(nn.Parameter(torch.tensor([0.])))
        self.cweights_list = nn.ParameterList().append(nn.Parameter(torch.tensor([1.])))
        self.means_list = nn.ParameterList().append(nn.Parameter(torch.zeros(1, a_dim)))
        self.logsigs = nn.Parameter(torch.zeros(a_dim))
        # self.logtemp = torch.tensor(0.7936915159225464)
        # self.mud = torch.tensor(1.1713083982467651)
        self.logtemp = torch.tensor(0.)
        self.mud = torch.tensor(1.17)
        # self.logtemp = nn.Parameter(torch.tensor(0.))
        # self.mud = nn.Parameter(torch.tensor(1.))
        self.cweights = self.means = None
        self.update_clustering()

    def update_clustering(self):
        # to speed up computation
        self.cweights = torch.cat([*self.cweights_list.parameters()])
        self.means = torch.cat([*self.means_list.parameters()])

    def forward(self, s):
        with torch.no_grad():
            # set init state as first center
            if self.centers is None:
                self.centers = s.clone().detach()[None, :]

            return self.distribution(s).sample()

    def log_prob(self, s, a):
        return self.distribution(s).log_prob(a)[:, None]

    def membership(self, s):
        w = self.unormalized_membership(s)
        return w / torch.sum(w, dim=-1, keepdim=True)

    def unormalized_membership(self, s):
        # compute distances to cluster
        dist = torch.sum((s[:, None, :] - self.centers[None, :, :]) ** 2, dim=-1)
        # w = (self.rootweights ** 2) * torch.exp(-torch.exp(self.logtemp) * dist) + 1e-6
        # w = (torch.abs(self.rootweights) + (self.rootweights == 0.).float() * self.rootweights) * torch.exp(-torch.exp(self.logtemp) * dist) + 1e-6
        # w = Grad1Abs.apply(self.cweights) * torch.exp(-torch.exp(self.logtemp) * dist)
        # w = Grad1Abs.apply(self.cweights) * torch.exp(-torch.exp(self.logtemp) * dist)
        w = Grad1Abs.apply(self.cweights) * torch.sigmoid(-torch.exp(self.logtemp) * (dist - self.mud))
        # w = torch.exp(-torch.exp(self.logtemp) * dist + self.rootweights)
        return w

    def distribution(self, s):
        means = self.get_weighted_means(s)
        return torch.distributions.MultivariateNormal(means, scale_tril=self.get_chol())

    def get_weighted_means(self, s):
        if len(s.size()) == 1:
            s = s[None, :]
        w = self.membership(s)

        # compute weighted means
        return w.matmul(self.means)

    def get_chol(self):
        return torch.diag(torch.exp(self.logsigs))

    def entropy(self):
        return self.a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self.logsigs).detach().numpy()

    def add_cluster(self, s, a):
        self.centers = torch.cat([self.centers, s])
        self.means_list.append(nn.Parameter(a))
        self.cweights_list.append(nn.Parameter(torch.tensor([0.])))
        # self.rootweights_list.append(nn.Parameter(torch.tensor([-100.])))
        # self.rootweights_list.append(nn.Parameter(torch.tensor([-2.])))
        self.update_clustering()
