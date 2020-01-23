import os
import torch
import time

import numpy as np

from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from metric_rl.gym_fixed import GymFixed
from mushroom_rl.utils.dataset import compute_J
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


def replay(env_id, horizon, gamma, torch_policy, dt, n_episodes, seed):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mdp = GymFixed(env_id, horizon, gamma)
    mdp.render()

    # Set environment seed
    mdp.env.seed(seed)

    # Set agent
    agent = DummyAgent(torch_policy, dt)

    # Set experiment
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes, render=True, quiet=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    print('J: {}, R: {}'.format(J, R))
    print('##################################################################################################')


def load_policy(log_name, iteration):
    policy_path = os.path.join(log_name, 'net/network-6-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


if __name__ == '__main__':
    # Bipedal Walker
    # env_id = 'BipedalWalker-v2'
    # horizon = 1600
    # env_id = 'AntBulletEnv-v0'
    # env_id = 'Walker2DBulletEnv-v0'
    env_id = 'HopperBulletEnv-v0'
    # env_id = 'HalfCheetahBulletEnv-v0'
    horizon = 1000
    gamma = .99

    log_name = 'Results/heurndel/' + env_id + '/metricrl_c40hcovrdTrue'

    policy = load_policy(log_name, iteration=1001)

    replay(env_id, horizon, gamma, policy, dt=1./30,
           n_episodes=10, seed=0)