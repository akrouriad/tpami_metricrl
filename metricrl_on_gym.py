import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import data_handling as dat


class MLP(nn.Module):
    def __init__(self, size_list, activation_list=None):
        super().__init__()
        self.size_list = size_list
        if not activation_list or activation_list is None:
            activation_list = [torch.tanh] * (len(size_list) - 2) + [None]
        self.activation_list = activation_list
        self.layers = nn.ModuleList()
        for k, kp, activ in zip(size_list[:-1], size_list[1:], activation_list):
            self.layers.append(nn.Linear(k, kp))
            if activ is not None:
                nn.init.xavier_normal_(self.layers[-1].weight, nn.init.calculate_gain(activ.__name__))
            else:
                nn.init.xavier_normal_(self.layers[-1].weight)

    def forward(self, x):
        for l, a in zip(self.layers, self.activation_list):
            if a is not None:
                x = a(l(x))
            else:
                x = l(x)
        return x


class GaussianPolicy(nn.Module):
    def __init__(self, mean_func, a_dim):
        super().__init__()
        self.a_dim = a_dim
        self.mean_func = mean_func
        self.log_sigma = nn.Parameter(torch.zeros(a_dim))

    def forward(self, x):
        cov = torch.diag(torch.exp(2 * self.log_sigma))
        return torch.distributions.MultivariateNormal(loc=self.mean_func(x), covariance_matrix=cov).sample()

    def log_prob(self, x, y):
        cov = torch.diag(torch.exp(2 * self.log_sigma))
        return torch.distributions.MultivariateNormal(loc=self.mean_func(x), covariance_matrix=cov).log_prob(y)

    def entropy(self):
        return self.a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self.log_sigma).detach().numpy()

def get_targets(mlp, obs, rwd, done, discount, lam):
    # computes v_update targets
    v_values = mlp(torch.tensor(obs, dtype=torch.float)).detach().numpy()
    gen_adv = np.empty_like(v_values)
    for rev_k, v in enumerate(reversed(v_values)):
        k = len(v_values) - rev_k - 1
        if done[k]:  # this is a new path. always true for rev_k == 0
            gen_adv[k] = rwd[k] - v_values[k]
        else:
            gen_adv[k] = rwd[k] + discount * v_values[k + 1] - v_values[k] + discount * lam * gen_adv[k + 1]
    return gen_adv + v_values, gen_adv


def get_adv(mlp, obs, rwd, done, discount, lam):
    _, adv = get_targets(mlp, obs, rwd, done, discount, lam)
    return adv


def learn(envid):
    env = gym.make(envid)
    h_layer_width = 64
    h_layer_length = 2
    lr = 3e-4
    nb_epochs = 10
    eps_ppo = .2

    input_sizes = [env.observation_space.shape[0]] + [h_layer_width] * h_layer_length
    value_mlp = MLP(input_sizes + [1])
    v_optim = torch.optim.Adam(value_mlp.parameters(), lr=lr)

    policy_mlp = MLP(input_sizes + [env.action_space.shape[0]])
    policy_torch = GaussianPolicy(policy_mlp, env.action_space.shape[0])
    policy = lambda obs: policy_torch(torch.tensor(obs, dtype=torch.float)).detach().numpy()
    p_optim = torch.optim.Adam(policy_torch.parameters(), lr=lr)

    discount = .99
    lam = .95
    max_iter = 100
    min_sample_per_iter = 3200
    for iter in range(max_iter):
        p_paths = dat.rollouts(env, policy, min_sample_per_iter, render=False)
        print("iter {}: rewards {} entropy {}".format(iter + 1, np.sum(p_paths['rwd']) / np.sum(p_paths['done']), policy_torch.entropy()))
        obs = torch.tensor(p_paths['obs'], dtype=torch.float)
        act = torch.tensor(p_paths['act'], dtype=torch.float)

        # update policy and v
        adv = get_adv(mlp=value_mlp, obs=p_paths['obs'], rwd=p_paths['rwd'], done=p_paths['done'], discount=discount, lam=lam)
        torch_adv = torch.tensor(adv, dtype=torch.float)
        old_log_p = policy_torch.log_prob(obs, act).detach()


        v_target, _ = get_targets(value_mlp, p_paths['obs'], p_paths['rwd'], p_paths['done'], discount, lam)
        torch_targets = torch.tensor(v_target, dtype=torch.float)

        # compute v_targets
        for epoch in range(nb_epochs):
            for batch_idx in dat.next_batch_idx(h_layer_width, len(p_paths['rwd'])):
                # update value network
                v_optim.zero_grad()
                mse = F.mse_loss(value_mlp(obs[batch_idx]), torch_targets[batch_idx])
                mse.backward()
                v_optim.step()

                # update policy
                p_optim.zero_grad()
                prob_ratio = torch.exp(policy_torch.log_prob(obs[batch_idx], act[batch_idx]) - old_log_p[batch_idx])
                clipped_ratio = torch.clamp(prob_ratio, 1 - eps_ppo, 1 + eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * torch_adv[batch_idx], clipped_ratio * torch_adv[batch_idx]))
                loss.backward()
                p_optim.step()

if __name__ == '__main__':
    # learn(envid='MountainCarContinuous-v0')
    learn(envid='BipedalWalker-v2')