import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm, trange

from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.utils.dataset import compute_J

from metric_rl.proj_metricrl import ProjectionMetricRL
from metric_rl.proj_swap_metricrl import ProjectionSwapMetricRL
from metric_rl.logger import generate_log_folder, save_parameters, Logger
from metric_rl.rl_shared import TwoPhaseEntropProfile, MLP

def experiment(env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, seed, params,
               log_name=None, swap=True):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name, 'net')

    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    # Set critic params (add input shape)
    input_shape = mdp.info.observation_space.shape
    critic_params = dict(input_shape=input_shape,
                         **params['critic_params'])
    params['critic_params'] = critic_params

    if swap:
        agent = ProjectionSwapMetricRL(mdp.info, **params)
    else:
        agent = ProjectionMetricRL(mdp.info, **params)

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

        logger.save(network=agent.policy._regressor, J=J_list, R=R_list, E=E_list)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('cweights {}'.format(agent.policy._regressor._c_weights))
        tqdm.write('##################################################################################################')

    print('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


def get_parameters(n_clusters):

    policy_params = dict(n_clusters=n_clusters,
                         std_0=1.0)

    actor_optimizer = {'class': optim.Adam,
                       'cw_params': {'lr': .1},
                       'means_params': {'lr': .001},
                       'log_sigma_params': {'lr': .001}}
    e_profile = {'class': TwoPhaseEntropProfile,
                 'params': {'e_reduc': .015}}

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


if __name__ == '__main__':
    n_clusters = 5
    params = get_parameters(n_clusters)

    # Bipedal Walker
    env_id = 'BipedalWalker-v2'
    horizon = 1600
    gamma = .99

    log_name = generate_log_folder(env_id, 'projection', str(n_clusters), True)
    save_parameters(log_name, params)
    experiment(env_id=env_id, horizon=horizon, gamma=gamma, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000,
               n_episodes_test=25, seed=0, params=params, log_name=log_name, swap=True)


    # Hopper Bullet
    # log_name = generate_log_folder('hopper_bullet', 'projection', str(n_max_clusters), True)
    # save_parameters(log_name, params)
    # experiment(env_id='HopperBulletEnv-v0', horizon=1000, gamma=.99, n_epochs=100, n_steps=30000, n_steps_per_fit=3000,
    #            n_episodes_test=10, seed=0, params=params, log_name=log_name)
