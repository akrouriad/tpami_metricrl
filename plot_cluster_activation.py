import os
import torch
import time

import numpy as np
import matplotlib.pyplot as plt

from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from metric_rl.gym_fixed import GymFixed
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.torch import to_float_tensor


class DummyAgent(Agent):
    def __init__(self, torch_policy, dt):
        self._regressor = torch_policy
        self._dt = dt

    def draw_action(self, state):

        time.sleep(self._dt)
        with torch.no_grad():
            s = to_float_tensor(np.atleast_2d(state), False)

            mu, chol_sigma = self._regressor(s)
            dist = torch.distributions.MultivariateNormal(mu, scale_tril=chol_sigma)
            a = dist.sample()

            return torch.squeeze(a, dim=0).detach().cpu().numpy()

    def episode_start(self):
        pass

    def fit(self, dataset):
        pass

def get_cluster_activation(env_id, horizon, gamma, torch_policy, dt, n_episodes, seed):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mdp = GymFixed(env_id, horizon, gamma)
    if dt > 0:
        mdp.render()

    # Set environment seed
    mdp.env.seed(seed)

    # Set agent
    agent = DummyAgent(torch_policy, dt)

    # Set experiment
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes, quiet=True)

    s, *_ = parse_dataset(dataset)
    print(len(s))
    w = agent._regressor.get_membership(torch.tensor(s))

    return w.detach().numpy()


def plot_activations(w, idxs):
    fig, axes = plt.subplots(len(idxs) + 2, figsize=(10,40), sharex=True, sharey=True)

    lines = list()
    labels = list()
    c = plt.rcParams['axes.prop_cycle']()

    for i, idx in enumerate(idxs):
        w_idx = w[:, idx]
        lines += axes[i].plot(w_idx, color=c.__next__()['color'])
        labels.append('cluster ' + str(idx))

    w_total = np.sum(w, axis=1)
    w_others = w_total - np.sum(w[:, idxs], axis=1)
    w_default = 1 - w_total

    lines += axes[-2].plot(w_others, color=c.__next__()['color'])
    labels.append('other clusters')
    lines += axes[-1].plot(w_default, color=c.__next__()['color'])
    labels.append('default cluster')

    plt.figlegend(lines, labels, ncol=len(idxs)+2, loc='lower center',
                  shadow=False, frameon=False)
    plt.show()


def load_policy(log_name, iteration, seed):
    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


if __name__ == '__main__':
    dt = 0.
    horizon = 1000
    gamma = .99

    # env_id = 'AntBulletEnv-v0'
    # log_name = 'Results/final_medium/AntBulletEnv-v0/metricrl_c10hcovr_expdTruet0.33snone'
    # seed = 0

    env_id = 'HopperBulletEnv-v0'
    log_name = 'Results/final_medium/HopperBulletEnv-v0/metricrl_c10hcovr_expdTruet1.0snone'
    seed = 12

    n_cluster = 3


    policy = load_policy(log_name, iteration=1001, seed=12)

    w = get_cluster_activation(env_id, horizon, gamma, policy, dt, n_episodes=1, seed=12)

    w_mean = np.mean(w, axis=0)
    idxs = w_mean.argsort()[::-1][:n_cluster]
    print('w:', w_mean)
    print('idx', idxs)

    plot_activations(w, np.sort(idxs))
