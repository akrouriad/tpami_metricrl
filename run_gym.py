import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm, trange

from mushroom_rl.core import Core
from metric_rl.gym_fixed import GymFixed
from mushroom_rl.utils.dataset import compute_J

from metric_rl.proj_metricrl import ProjectionMetricRL
from metric_rl.proj_swap_metricrl import ProjectionSwapMetricRL
from metric_rl.proj_del_swap_metricrl import ProjectionDelSwapMetricRL
from metric_rl.logger import generate_log_folder, save_parameters, Logger
from metric_rl.rl_shared import TwoPhaseEntropProfile, MLP, SinglePhaseEntropProfile
from joblib import Parallel, delayed

from multiprocessing import Process

def experiment(env_id, n_clusters, horizon, seed, gamma=.99, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000, n_episodes_test=5, a_cost_scale=0.,
               log_name=None, swap=True, clus_sel='covr', do_delete=True, temp=1., opt_temp=False, squash='none', max_cmean=1.):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name, 'net')

    params = get_parameters(n_clusters, temp, max_cmean)
    save_parameters(log_name, params)

    mdp = GymFixed(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    # Set critic params (add input shape)
    input_shape = mdp.info.observation_space.shape
    critic_params = dict(input_shape=input_shape,
                         **params['critic_params'])
    params['critic_params'] = critic_params

    if swap:
        params['a_cost_scale'] = a_cost_scale
        params['clus_sel'] = clus_sel
        params['do_delete'] = do_delete
        params['opt_temp'] = opt_temp
        params['squash'] = squash
        agent = ProjectionDelSwapMetricRL(mdp.info, **params)
    else:
        agent = ProjectionMetricRL(mdp.info, **params)

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
    # print('Press a button to visualize')
    # input()
    # core.evaluate(n_episodes=5, render=True)


def get_parameters(n_clusters, temp, max_cmean):

    policy_params = dict(n_clusters=n_clusters,
                         std_0=1., temp=temp, max_cmean=max_cmean)

    actor_optimizer = {'class': optim.Adam,
                       'cw_params': {'lr': .01},
                       'means_params': {'lr': .01},
                       'log_sigma_params': {'lr': .001}}
    e_profile = {'class': TwoPhaseEntropProfile,
                 'params': {'e_reduc': 0.0075, 'e_thresh_mult': .5}}
    # e_profile = {'class': SinglePhaseEntropProfile,
    #              'params': {'e_reduc': 0.015}}

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

    # n_clusters = 10
    # n_clusters = 20
    n_clusters = 10

    # Bipedal Walker
    # env_id = 'BipedalWalker-v2'
    # horizon = 1600
    # env_id = 'HopperBulletEnv-v0'
    # env_id = 'HalfCheetahBulletEnv-v0'
    # env_id = 'AntBulletEnv-v0'
    # env_id = 'HumanoidBulletEnv-v0'
    env_id = 'MountainCarContinuous-v0'
    # env_id = 'InvertedDoublePendulumBulletEnv-v0'
    # env_id = 'BipedalWalker-v2'
    horizon = 1000
    gamma = .99
    temp = 1.
    # clus_sel = 'old_covr_yetnew'
    clus_sel = 'covr_exp'
    do_delete = True
    opt_temp = False
    # squash = 'tanh'
    squash = 'none'
    max_cmean = 1.

    log_name = generate_log_folder(env_id, 'lower_std', str(n_clusters), True)
    print('log name', log_name)
    Parallel(n_jobs=n_jobs)(delayed(experiment)(env_id=env_id, n_clusters=n_clusters, horizon=horizon, gamma=gamma, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000,
               n_episodes_test=1, seed=seed, log_name=log_name, swap=True, clus_sel=clus_sel, do_delete=do_delete, temp=temp, opt_temp=opt_temp, squash=squash, max_cmean=max_cmean) for seed in range(n_experiments))

    # ps = []
    # for k in range(n_experiments):
    #     p = Process(target=experiment, kwargs={'env_id': env_id, 'horizon': horizon, 'gamma': gamma, 'n_epochs': 1000, 'n_steps': 3000,
    #                                            'n_steps_per_fit': 3000, 'n_episodes_test': 25, 'seed': k, 'params': params, 'log_name': log_name, 'swap': True})
    #     p.start()
    #     ps.append(p)
    #
    # for p in ps:
    #     p.join()

    # Hopper Bullet
    # log_name = generate_log_folder(env_id, 'projection_swap1kclus5', str(n_clusters), True)
    # save_parameters(log_name, params)
    # experiment(env_id='HopperBulletEnv-v0', horizon=horizon, gamma=gamma, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000,
    #            n_episodes_test=25, seed=0, params=params, log_name=log_name, swap=True)
