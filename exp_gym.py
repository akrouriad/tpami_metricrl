import os
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J

from metric_rl.metric_rl import MetricRL
from metric_rl.logger import save_parameters, Logger
from metric_rl.rl_shared import TwoPhaseEntropProfile, MLP


def experiment(env_id, horizon, gamma,
               n_clusters, no_delete, temp,
               n_epochs, n_steps, n_steps_per_fit,
               n_episodes_test, seed, results_dir):
    print('Metric RL')
    os.makedirs(results_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(results_dir, 'net')

    params = get_parameters(n_clusters, temp)
    save_parameters(results_dir, params)

    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    # Set critic params (add input shape)
    input_shape = mdp.info.observation_space.shape
    critic_params = dict(input_shape=input_shape,
                         **params['critic_params'])
    params['critic_params'] = critic_params
    params['do_delete'] = not no_delete

    agent = MetricRL(mdp.info, **params)

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
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=True)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, quiet=True)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        J_list.append(J)
        R_list.append(R)
        E_list.append(E)

        logger.save(J=J_list, R=R_list, E=E_list, seed=seed)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('cweights {}'.format(agent.policy._regressor._c_weights))
        tqdm.write('##################################################################################################')

    logger.save(network=agent.policy._regressor, seed=seed)


def default_params():
    defaults = dict(
        gamma=.99,
        n_epochs=1000,
        n_steps=3000,
        n_steps_per_fit=3000,
        n_episodes_test=5,
        n_clusters=10,
        no_delete=True,
        temp=1.,
    )

    return defaults


def get_parameters(n_clusters, temp):

    policy_params = dict(n_clusters=n_clusters,
                         std_0=1., temp=temp)

    actor_optimizer = {'class': optim.Adam,
                       'cw_params': {'lr': .01},
                       'means_params': {'lr': .01},
                       'log_sigma_params': {'lr': .001}}
    e_profile = {'class': TwoPhaseEntropProfile,
                 'params': {'e_reduc': 0.0075, 'e_thresh_mult': .5}}

    critic_params = dict(network=MLP,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         batch_size=64,
                         output_shape=(1,),
                         size_list=[64, 64],
                         n_models=2,
                         prediction='min',
                         quiet=True)

    critic_fit_params = dict(n_epochs=10)

    params = dict(policy_params=policy_params,
                  critic_params=critic_params,
                  actor_optimizer=actor_optimizer,
                  n_epochs_per_fit=20,
                  batch_size=64,
                  entropy_profile=e_profile,
                  max_kl=.015,
                  lam=.95,
                  critic_fit_params=critic_fit_params)

    return params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str)
    parser.add_argument("--horizon", type=int)
    parser.add_argument('--gamma', type=float)

    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--no-delete", action='store_false')

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

