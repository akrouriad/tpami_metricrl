import os
import torch
import argparse

import numpy as np

from pathlib import Path

from mushroom_rl.core import Core, Agent, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset


def replay(path, env_id, n_episodes, seed, save):

    logger = Logger(log_name='Metric RL', results_dir='logs' if save else None)
    logger.info(f'Replaying MetricRL agent in {path}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mdp = Gym(env_id)
    agent = Agent.load(path)

    if 'BulletEnv-v0' in env_id:
        mdp.render()
        render = False
    else:
        render = True

    # Set environment seed
    mdp.env.seed(seed)

    mdp.env.reset()

    distance = 4
    pitch = -5
    if 'BulletEnv-v0' in env_id:
        mdp.env.env._p.resetDebugVisualizerCamera(cameraTargetPosition=[4.5, 0, 1.],
                                                  cameraDistance=distance,
                                                  cameraYaw=0.,
                                                  cameraPitch=pitch)

    # Set experiment
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes, render=render, quiet=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    logger.epoch_info(0, J=J, R=R)

    s, *_ = parse_dataset(dataset)
    w = torch.mean(agent.policy._regressor.get_membership(torch.tensor(s)), axis=0)
    _, top_w = torch.topk(w, 5)
    c = agent.policy._regressor.get_c_weights()
    _, top_c = torch.topk(c, 5)

    logger.info(f'w: {w.detach().numpy()})')
    logger.info(f'top w: {top_w.detach().numpy()}')
    logger.info(f'c: {w.detach().numpy()}')
    logger.info(f'top c: {top_c.detach().numpy()}')

    if env_id == 'Pendulum-v0':
        w = agent._regressor.get_membership(torch.tensor(s)).detach().numpy()
        w_default = np.expand_dims(1 - np.sum(w, axis=1), -1)

        w_tot = np.concatenate([w, w_default], axis=1)
        for run in range(n_episodes):
            logger.info(f'w_tot: {np.argmax(w_tot[100*run:100*(run+1), :], axis=1)}')

    logger.strong_line()

    if save:
        logger.log_dataset(dataset)


def load_policy(log_name, iteration, seed):
    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', '-p', type=str,
                        default='Results/final_medium/HopperBulletEnv-v0/metricrl_c10hcovr_expdTruet1.0snone')
    parser.add_argument("--env-id", '-e', type=str,
                        default='HopperBulletEnv-v0')
    parser.add_argument("--seed", '-s', type=int, default=0)
    parser.add_argument("--n-episodes", '-n', type=int, default=1)

    args = parser.parse_args()

    save = False

    iteration = 1001

    path = Path(args.path) / f'agent-{args.seed}.msh'

    replay(path, args.env_id, n_episodes=args.n_episodes, seed=args.seed, save=save)



