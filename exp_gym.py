import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J

from metric_rl.metric_rl import MetricRL
from metric_rl.utils import save_parameters
from metric_rl.rl_shared import TwoPhaseEntropProfile, MLP

from experiment_launcher import get_default_params


def experiment(env_id, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000, n_episodes_test=5, n_clusters=10,
               no_delete=True, temp=1., seed=0, results_dir=None):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name='MetricRL', results_dir=results_dir, log_console=results_dir is not None, seed=seed)

    logger.info('Running MetricRL experiment')
    logger.strong_line()

    params = get_parameters(n_clusters, temp)
    save_parameters(logger.path, params)

    mdp = Gym(env_id)

    # Set environment seed
    mdp.env.seed(seed)

    # Set critic params (add input shape)
    input_shape = mdp.info.observation_space.shape
    critic_params = dict(input_shape=input_shape,
                         **params['critic_params'])
    params['critic_params'] = critic_params
    params['do_delete'] = not no_delete

    agent = MetricRL(mdp.info, **params)

    agent.set_logger(logger)

    core = Core(agent, mdp)

    # Initial evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.log_numpy(J=J, R=R, E=E)
    logger.epoch_info(0, J=J, R=R, E=E)

    # Learning
    for it in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=True)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, quiet=True)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        logger.log_numpy(J=J, R=R, E=E)
        logger.epoch_info(it+1, J=J, R=R, E=E, cweights=agent.policy._regressor._c_weights.data.cpu().numpy())

    logger.log_agent(agent)


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

    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--no-delete", action='store_false')

    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--n-steps-per-fit", type=int)
    parser.add_argument("--n-episodes-test", type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)

    parser.set_defaults(**get_default_params(experiment))
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)

