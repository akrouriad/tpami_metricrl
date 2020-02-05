import os
import torch
import time

import numpy as np

from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from metric_rl.gym_fixed import GymFixed
from mushroom_rl.utils.dataset import compute_J, parse_dataset
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

    s, *_ = parse_dataset(dataset)
    w = torch.mean(agent._regressor.get_membership(torch.tensor(s)), axis=0)
    _, top_w = torch.topk(w, 5)
    c = agent._regressor.get_c_weights()
    _, top_c = torch.topk(c, 5)

    print('w: ', w.detach().numpy(), ' top: ', top_w.detach().numpy())
    print('c: ', w.detach().numpy(), ' top: ', top_c.detach().numpy())

    print('##################################################################################################')


def load_policy(log_name, iteration, seed):
    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


if __name__ == '__main__':
    dt = 0

    horizon = 1000
    gamma = .99

    # env_id = 'AntBulletEnv-v0'
    # log_name = 'Results/final_medium/AntBulletEnv-v0/metricrl_c10hcovr_expdTruet0.33snone'
    # seed = 0

    env_id = 'HopperBulletEnv-v0'
    log_name = 'Results/final_medium/HopperBulletEnv-v0/metricrl_c10hcovr_expdTruet1.0snone'
    seed = 12


    policy = load_policy(log_name, iteration=1001, seed=12)

    replay(env_id, horizon, gamma, policy, dt=dt, n_episodes=10, seed=12)

