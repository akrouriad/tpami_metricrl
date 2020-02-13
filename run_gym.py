import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from mushroom_rl.core import Core
from metric_rl.gym_fixed import GymFixed
from mushroom_rl.utils.dataset import compute_J

from metric_rl.metric_rl import MetricRL
from metric_rl.logger import generate_log_folder, save_parameters, Logger
from metric_rl.rl_shared import TwoPhaseEntropProfile, MLP
from joblib import Parallel, delayed


def experiment(env_id, n_clusters, horizon, seed, gamma=.99, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000,
               n_episodes_test=5, log_name=None, do_delete=True, temp=1.):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name, 'net')

    params = get_parameters(n_clusters, temp)
    save_parameters(log_name, params)

    mdp = GymFixed(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    # Set critic params (add input shape)
    input_shape = mdp.info.observation_space.shape
    critic_params = dict(input_shape=input_shape,
                         **params['critic_params'])
    params['critic_params'] = critic_params
    params['do_delete'] = do_delete

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


if __name__ == '__main__':
    n_experiments = 1
    n_jobs = n_experiments

    # n_clusters = 40
    # n_clusters = 20
    n_clusters = 10

    gamma = .99

    # Pybullet
    env_id = 'AntBulletEnv-v0'
    # env_id = 'HalfCheetahBulletEnv-v0'
    temp = .33
    horizon = 1000

    # Bipedal Walker
    # env_id = 'BipedalWalker-v2'
    # horizon = 1600
    # temp = 1.

    do_delete = True

    log_name = generate_log_folder(env_id, 'metric', str(n_clusters), True)
    print('log name', log_name)
    Parallel(n_jobs=n_jobs)(delayed(experiment)(env_id=env_id,
                                                n_clusters=n_clusters,
                                                horizon=horizon, gamma=gamma,
                                                n_epochs=1000, n_steps=3000, n_steps_per_fit=3000,
                                                n_episodes_test=5, seed=seed, log_name=log_name,
                                                do_delete=do_delete, temp=temp)
                            for seed in range(n_experiments))
