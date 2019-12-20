import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.utils.dataset import compute_J

from metric_rl.proj_swap_rt_metricrl import ProjectionSwapRTMetricRL
from metric_rl.logger import generate_log_folder, save_parameters, Logger
from metric_rl.rl_shared import TwoPhaseEntropProfile, CriticMLP
from joblib import Parallel, delayed


def experiment(env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, seed, params,
               log_name=None):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name, 'net')

    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    # Set critic params (add input shape)
    input_shape = (mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(input_shape=input_shape,
                         **params['critic_params'])
    params['critic_params'] = critic_params

    agent = ProjectionSwapRTMetricRL(mdp.info, **params)

    core = Core(agent, mdp)

    J_list = list()
    R_list = list()
    E_list = list()

    for it in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=True)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False, quiet=True)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        J_list.append(J)
        R_list.append(R)
        E_list.append(E)

        logger.save(network=agent.policy._regressor, J=J_list, R=R_list, E=E_list, seed=seed)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('cweights {}'.format(agent.policy._regressor._c_weights))
        tqdm.write('##################################################################################################')

    # print('Press a button to visualize')
    # input()
    # core.evaluate(n_episodes=5, render=True)


def get_parameters(n_clusters):

    policy_params = dict(n_clusters=n_clusters,
                         std_0=1.0)

    actor_optimizer = {'class': optim.Adam,
                       'cw_params': {'lr': .1},
                       'means_params': {'lr': .001},
                       'log_sigma_params': {'lr': .001}}
    e_profile = {'class': TwoPhaseEntropProfile,
                 'params': {'e_reduc': .015}}

    critic_params = dict(network=CriticMLP,
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
                  initial_replay_size=1000,
                  max_replay_size=100000,
                  tau=0.005,
                  n_epochs_per_fit=10,
                  batch_size=64,
                  entropy_profile=e_profile,
                  max_kl=.015,
                  lam=.95,
                  critic_fit_params=critic_fit_params)

    return params


if __name__ == '__main__':
    n_experiments = 11
    n_jobs = n_experiments

    n_clusters = 5
    params = get_parameters(n_clusters)

    # Bipedal Walker
    env_id = 'BipedalWalker-v2'
    horizon = 1600
    # env_id = 'HopperBulletEnv-v0'
    # horizon = 1000
    gamma = .99

    log_name = generate_log_folder(env_id, 'projection_randadv_rt', str(n_clusters), True)
    save_parameters(log_name, params)
    Parallel(n_jobs=n_jobs)(delayed(experiment)(env_id=env_id, horizon=horizon, gamma=gamma, n_epochs=1000, n_steps=3000,
                                                n_steps_per_fit=1, n_episodes_test=25, seed=seed, params=params,
                                                log_name=log_name) for seed in range(n_experiments))
