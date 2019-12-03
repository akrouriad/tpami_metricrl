import torch
import torch.nn as nn
import numpy as np


class TwoPhaseEntropProfile:
    def __init__(self, policy, e_reduc, e_thresh):
        self.init_entropy = policy.entropy()
        self._policy = policy
        self._e_reduc = e_reduc
        self._e_thresh = e_thresh
        self._phase = 1
        self._iter = 0

    def get_e_lb(self):
        if self._phase == 1 and self._policy.entropy() > self._e_thresh:
            return -10000.
        else:
            self._phase = 2
            self._iter += 1
            return self._e_thresh - self._iter * self._e_reduc


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, size_list, activation_list=None, preproc=None, **kwargs):
        super().__init__()
        self.size_list = [input_shape[0]] + size_list + [output_shape[0]]
        if not activation_list or activation_list is None:
            activation_list = [torch.tanh] * (len(self.size_list) - 2) + [None]
        self.activation_list = activation_list
        self.preproc = preproc
        self.layers = nn.ModuleList()
        for k, kp, activ in zip(self.size_list[:-1], self.size_list[1:], self.activation_list):
            self.layers.append(nn.Linear(k, kp))
        self.weights_init()

    def forward(self, x, **kwargs):
        if self.preproc is not None:
            x = self.preproc(x)

        for l, a in zip(self.layers, self.activation_list):
            if a is not None:
                x = a(l(x))
            else:
                x = l(x)
        return x

    def weights_init(self):
        for k, activ in enumerate(self.activation_list):
            if activ is not None:
                nn.init.xavier_uniform_(self.layers[k].weight, nn.init.calculate_gain(activ.__name__))
            else:
                nn.init.xavier_uniform_(self.layers[k].weight)
            nn.init.zeros_(self.layers[k].bias)


def get_targets(v_func, x, x_n, rwd, absorbing, last, discount, lam, **predict_params):
    v = v_func(x, **predict_params)
    v_next = v_func(x_n, **predict_params)
    gen_adv = np.empty_like(v)
    for rev_k, _ in enumerate(reversed(v)):
        k = len(v) - rev_k - 1
        if last[k] or rev_k == 0:
            gen_adv[k] = rwd[k] - v[k]
            if not absorbing[k]:
                gen_adv[k] += discount * v_next[k]
        else:
            gen_adv[k] = rwd[k] + discount * v_next[k] - v[k] + discount * lam * gen_adv[k + 1]
    return gen_adv + v, gen_adv


def get_adv(v_func, x, xn,  rwd, absorbing, last, discount, lam, **predict_params):
    _, adv = get_targets(v_func, x, xn, rwd, absorbing, last, discount, lam, **predict_params)
    return adv