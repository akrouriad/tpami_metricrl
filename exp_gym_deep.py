import os
import argparse

import torch
import numpy as np
from tqdm import tqdm

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J

from metric_rl.logger import save_parameters, Logger
from metric_rl.rl_shared import MLP

import torch.optim as optim
import torch.nn.functional as F


def experiment(alg_name, env_id, horizon, gamma,
               n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               n_models_v, seed, results_dir):
    print(alg_name)
    os.makedirs(results_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(results_dir, 'net')

    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    # Set parameters
    policy_params = dict(std_0=1.,
                         size_list=[64, 64],
                         use_cuda=False)

    critic_params = dict(network=MLP,
                         loss=F.mse_loss,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         batch_size=64,
                         n_models=n_models_v,
                         size_list=[64, 64])

    alg, alg_params = get_alg_and_parameters(alg_name)

    policy = GaussianTorchPolicy(MLP,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Save alg params
    save_parameters(results_dir, dict(alg_params=alg_params))

    # Run learning
    core = Core(agent, mdp)

    J_list = list()
    R_list = list()
    E_list = list()

    # Initial evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    J_list.append(J)
    R_list.append(R)
    E_list.append(E)

    logger.save(J=J_list, R=R_list, E=E_list, seed=seed)

    tqdm.write('EPOCH 0')
    tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
    tqdm.write('##################################################################################################')

    # Learning
    for it in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        J_list.append(J)
        R_list.append(R)
        E_list.append(E)

        logger.save(J=J_list, R=R_list, E=E_list, seed=seed)

        tqdm.write('END OF EPOCH ' + str(it + 1))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('##################################################################################################')


def get_alg_and_parameters(alg_name):
    if alg_name == 'PPO':
        alg_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': 3e-4}},
                          n_epochs_policy=10,
                          batch_size=64,
                          eps_ppo=.2,
                          lam=.95,
                          quiet=True)

        return PPO, alg_params

    elif alg_name == 'TRPO':
        alg_params = dict(ent_coeff=0.0,
                          max_kl=.01,
                          lam=.95,
                          n_epochs_line_search=10,
                          n_epochs_cg=10,
                          cg_damping=1e-2,
                          cg_residual_tol=1e-10,
                          quiet=True)

        return TRPO, alg_params

    else:
        raise RuntimeError


def default_params():
    defaults = dict(
        gamma=.99,
        n_epochs=1000,
        n_steps=3000,
        n_steps_per_fit=3000,
        n_episodes_test=5,
        n_models_v=1
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg-name', type=str)
    parser.add_argument("--env-id", type=str)
    parser.add_argument("--horizon", type=int)
    parser.add_argument('--gamma', type=float)

    parser.add_argument("--n-models-v", type=int)

    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--n-steps-per-fit", type=int)
    parser.add_argument("--n-episodes-test", type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
