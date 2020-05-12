import torch
import numpy as np
from tqdm import tqdm, trange

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J

from metric_rl.logger import generate_log_folder, save_parameters, Logger
from metric_rl.rl_shared import MLP

import torch.optim as optim
import torch.nn.functional as F


def experiment(alg_name, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               seed, n_models_v=1, log_name=None, visualize=False):
    print(alg_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name, 'net')

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
                         n_models=n_models_v,
                         size_list=[64, 64])

    alg, alg_params = get_alg_and_parameters(alg_name)

    policy = GaussianTorchPolicy(MLP,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

    # Save alg params
    save_parameters(log_name, dict(alg_params=alg_params))

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

    if visualize:
        print('Press a button to visualize')
        input()
        core.evaluate(n_episodes=5, render=True)


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


if __name__ == '__main__':
    # Algs
    algs = ['PPO', 'TRPO']

    # BipedalWalker
    env_id = 'BipedalWalker-v2'
    horizon = 1600
    gamma = .99

    # HopperBullet
    # env_id = 'HopperBulletEnv-v0'
    # horizon = 1000
    # gamma = .99

    for alg_name in algs:
        log_name = generate_log_folder(env_id, alg_name, timestamp=True)
        experiment(alg_name=alg_name, env_id=env_id, horizon=horizon, gamma=gamma,
                   n_epochs=2, n_steps=30000, n_steps_per_fit=3000, n_episodes_test=10,
                   seed=0, log_name=log_name, visualize=True)
