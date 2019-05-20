import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, size_list, activation_list=None, preproc=None):
        super().__init__()
        self.size_list = size_list
        if not activation_list or activation_list is None:
            activation_list = [torch.tanh] * (len(size_list) - 2) + [None]
        self.activation_list = activation_list
        self.preproc = preproc
        self.layers = nn.ModuleList()
        for k, kp, activ in zip(size_list[:-1], size_list[1:], activation_list):
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


class RunningMeanStdFilter:
    def __init__(self, s_dim, min_clamp, max_clamp, center=True):
        self.sumx = torch.zeros(s_dim, dtype=torch.float64)
        self.sumx2 = torch.zeros(s_dim, dtype=torch.float64)
        self.mean = torch.zeros(s_dim, dtype=torch.float64)
        self.std = torch.zeros(s_dim, dtype=torch.float64)
        self.count = torch.zeros(1, dtype=torch.float64)
        self.min_clamp, self.max_clamp = min_clamp, max_clamp
        self._center = center

    def update(self, x):
        self.count += x.size()[0]
        self.sumx += torch.sum(x, dim=0).to(torch.float64)
        self.sumx2 += torch.sum(x ** 2, dim=0).to(torch.float64)
        self.mean = self.sumx / self.count
        self.std = torch.clamp(torch.sqrt(self.sumx2 / self.count - self.mean ** 2), 1e-2)

    def __call__(self, x):
        dtype = x.dtype
        if self.count > 0:
            if self._center:
                return torch.clamp((x.to(torch.float64) - self.mean) / (self.std + 1e-8), self.min_clamp, self.max_clamp).to(dtype)
            else:
                return torch.clamp(x.to(torch.float64) / (self.std + 1e-8), self.min_clamp, self.max_clamp).to(dtype)
        else:
            return x


class ValueFunction(nn.Module):
    def __init__(self, approx):
        super().__init__()
        self._approx = approx

    def forward(self, x):
        return self._approx(x)


class ValueFunctionList:
    def __init__(self, value_f_list):
        self._v_list = value_f_list

    def __call__(self, x, reduce_fct=torch.min):
        list_v = [self._v_list[0](x)]
        for v_f_k in self._v_list[1:]:
            list_v.append(v_f_k(x))
        return reduce_fct(torch.cat(list_v, dim=1), dim=1, keepdim=True)[0]


def get_targets(v_func, obs, rwd, done, discount, lam):
    # computes v_update targets
    v_values = v_func(torch.tensor(obs, dtype=torch.float)).detach().numpy()
    gen_adv = np.empty_like(v_values)
    for rev_k, v in enumerate(reversed(v_values)):
        k = len(v_values) - rev_k - 1
        if done[k]:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = rwd[k] - v_values[k]
        else:
            gen_adv[k] = rwd[k] + discount * v_values[k + 1] - v_values[k] + discount * lam * gen_adv[k + 1]
    return gen_adv + v_values, gen_adv


def get_adv(v_func, obs, rwd, done, discount, lam):
    _, adv = get_targets(v_func, obs, rwd, done, discount, lam)
    return adv
