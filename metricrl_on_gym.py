import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import roboschool
import gym
import data_handling as dat
import rllog


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
    def __init__(self, s_dim, min_clamp, max_clamp):
        self.sumx = torch.zeros(s_dim, dtype=torch.float64)
        self.sumx2 = torch.zeros(s_dim, dtype=torch.float64)
        self.mean = torch.zeros(s_dim, dtype=torch.float64)
        self.std = torch.zeros(s_dim, dtype=torch.float64)
        self.count = torch.zeros(1, dtype=torch.float64)
        self.min_clamp, self.max_clamp = min_clamp, max_clamp

    def update(self, x):
        self.count += x.size()[0]
        self.sumx += torch.sum(x, dim=0).to(torch.float64)
        self.sumx2 += torch.sum(x ** 2, dim=0).to(torch.float64)
        self.mean = self.sumx / self.count
        self.std = torch.clamp(torch.sqrt(self.sumx2 / self.count - self.mean ** 2), 1e-2)

    def __call__(self, x):
        dtype = x.dtype
        if self.count > 0:
            return torch.clamp((x.to(torch.float64) - self.mean) / self.std, self.min_clamp, self.max_clamp).to(dtype)
        else:
            return x


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
        cov = torch.diag(torch.exp(2 * self.log_sigma))
        return torch.distributions.MultivariateNormal(loc=self.get_mean(x), covariance_matrix=cov).sample().detach()

    def log_prob(self, x, y):
        cov = torch.diag(torch.exp(2 * self.log_sigma))
        return torch.distributions.MultivariateNormal(loc=self.get_mean(x), covariance_matrix=cov).log_prob(y)[:, None]

    def entropy(self):
        return self.a_dim / 2 * np.log(2 * np.pi * np.e) + torch.sum(self.log_sigma).detach().numpy()


class ValueFunction(nn.Module):
    def __init__(self, approx):
        super().__init__()
        self._approx = approx

    def forward(self, x):
        return self._approx(x)


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


def learn(envid, seed=0, max_ts=1e6, norma='None', log_name=None):
    env = gym.make(envid)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    h_layer_width = 64
    h_layer_length = 2
    lr = 3e-4
    nb_epochs = 10
    eps_ppo = .2

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    obs_filter = RunningMeanStdFilter(s_dim, min_clamp=-5, max_clamp=5)
    rwd_filter = RunningMeanStdFilter(1, min_clamp=-5, max_clamp=5)

    input_sizes = [s_dim] + [h_layer_width] * h_layer_length
    value_mlp = MLP(input_sizes + [1], preproc=obs_filter)
    value = ValueFunction(value_mlp)
    v_optim = torch.optim.Adam(value.parameters(), lr=lr)

    policy_mlp = MLP(input_sizes + [a_dim], preproc=obs_filter)
    policy_torch = GaussianPolicy(policy_mlp, a_dim)
    policy = lambda obs: policy_torch(torch.tensor(obs, dtype=torch.float)).detach().numpy()
    p_optim = torch.optim.Adam(policy_torch.parameters(), lr=lr)

    discount = .99
    lam = .95
    min_sample_per_iter = 3200

    cum_ts = 0
    iter = 0
    if log_name is not None:
        logger = rllog.PolicyIterationLogger(log_name)
    while True:
        p_paths = dat.rollouts(env, policy, min_sample_per_iter, render=False)
        iter_ts = len(p_paths['rwd'])
        cum_ts += iter_ts
        obs = torch.tensor(p_paths['obs'], dtype=torch.float)
        act = torch.tensor(p_paths['act'], dtype=torch.float)
        rwd = torch.tensor(p_paths['rwd'], dtype=torch.float)
        avg_rwd = np.sum(p_paths['rwd']) / np.sum(p_paths['done'])
        if log_name is not None:
            logger.next_iter_path(p_paths['rwd'][:, 0], p_paths['done'], policy_torch.entropy())

        rwd_filter.update(rwd)
        if norma == 'All':
            rwd = rwd_filter(rwd)

        # update policy and v
        adv = get_adv(mlp=value, obs=p_paths['obs'], rwd=rwd.numpy(), done=p_paths['done'], discount=discount, lam=lam)
        torch_adv = torch.tensor(adv, dtype=torch.float)
        old_log_p = policy_torch.log_prob(obs, act).detach()

        v_target, _ = get_targets(value, p_paths['obs'], rwd.numpy(), p_paths['done'], discount, lam)
        torch_targets = torch.tensor(v_target, dtype=torch.float)

        print("iter {}: rewards {} entropy {} vf_loss {}".format(iter + 1, avg_rwd, policy_torch.entropy(),
                                                                 F.mse_loss(value(obs), torch_targets)))
        if cum_ts > max_ts:
            break
        # compute update filter, v_values and policy
        if norma == 'All' or norma == 'Obs':
            obs_filter.update(obs)
        for epoch in range(nb_epochs):
            for batch_idx in dat.next_batch_idx(h_layer_width, iter_ts):
                # update value network
                v_optim.zero_grad()
                mse = F.mse_loss(value(obs[batch_idx]), torch_targets[batch_idx])
                mse.backward()
                v_optim.step()

                # update policy
                p_optim.zero_grad()
                prob_ratio = torch.exp(policy_torch.log_prob(obs[batch_idx], act[batch_idx]) - old_log_p[batch_idx])
                clipped_ratio = torch.clamp(prob_ratio, 1 - eps_ppo, 1 + eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * torch_adv[batch_idx], clipped_ratio * torch_adv[batch_idx]))
                loss.backward()
                p_optim.step()
        iter += 1


if __name__ == '__main__':
    # learn(envid='MountainCarContinuous-v0')
    learn(envid='BipedalWalker-v2', max_ts=1e5, seed=0, norma='All', log_name='test_log_all')
    # learn(envid='RoboschoolHalfCheetah-v1')
    # learn(envid='RoboschoolHopper-v1')
    # learn(envid='RoboschoolAnt-v1', seed=0, norma='None')
