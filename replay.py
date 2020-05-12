import os
import torch
import time
import pickle

import numpy as np

from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.torch import to_float_tensor


class DummyAgent(Agent):
    def __init__(self, torch_policy, dt, deterministic=True):
        self._regressor = torch_policy
        self._deterministic = deterministic
        self._dt = dt

    def draw_action(self, state):

        time.sleep(self._dt)
        with torch.no_grad():
            s = to_float_tensor(np.atleast_2d(state), False)

            mu, chol_sigma = self._regressor(s)

            if self._deterministic:
                return torch.squeeze(mu, dim=0).detach().cpu().numpy()
            else:
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

    mdp = Gym(env_id, horizon, gamma)

    if 'BulletEnv-v0' in env_id:
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

    if env_id == 'Pendulum-v0':
        w = agent._regressor.get_membership(torch.tensor(s)).detach().numpy()
        w_default = np.expand_dims(1 - np.sum(w, axis=1), -1)

        w_tot = np.concatenate([w, w_default], axis=1)
        for run in range(n_episodes):
            print(np.argmax(w_tot[100*run:100*(run+1), :], axis=1))


    print('##################################################################################################')

    return dataset


def load_policy(log_name, iteration, seed):
    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


if __name__ == '__main__':
    save = True
    dt = 1/60
    #dt = 0

    horizon = 100
    gamma = .99

    # env_id = 'AntBulletEnv-v0'
    # log_name = 'Results/final_medium/AntBulletEnv-v0/metricrl_c10hcovr_expdTruet0.33snone'
    # seed = 0
    # iteration = 1001

    # env_id = 'HopperBulletEnv-v0'
    # log_name = 'Results/final_medium/HopperBulletEnv-v0/metricrl_c10hcovr_expdTruet1.0snone'
    # seed = 12
    # iteration = 1001

    # env_id = 'HalfCheetahBulletEnv-v0'
    # log_name = 'Results/final_medium/HalfCheetahBulletEnv-v0/metricrl_c10hcovr_expdTruet0.33snone'
    # seed = 2
    # iteration = 1001

    env_id = 'Pendulum-v0'
    # log_name = 'Results/final_small2/Pendulum-v0/metricrl_c5hcovr_expdTruet1.0snone'
    # seed = 15
    log_name = 'Results/final_small2/Pendulum-v0/metricrl_c10hcovr_expdTruet1.0snone'
    seed = 3
    iteration = 501

    policy = load_policy(log_name, iteration=iteration, seed=seed)

    dataset = replay(env_id, horizon, gamma, policy, dt=dt, n_episodes=1, seed=seed)

    if save:
        with open('dataset.pkl', 'wb') as file:
            pickle.dump(dataset, file)

