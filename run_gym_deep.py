import torch
import numpy as np
from tqdm import tqdm, trange

from mushroom.core import Core
from mushroom.algorithms.actor_critic import PPO, TRPO
from mushroom.environments import Gym
from mushroom.policy import GaussianTorchPolicy
from mushroom.utils.dataset import compute_J

from metric_rl.logger import generate_log_folder, save_parameters, Logger
from metric_rl.rl_shared import MLP

import torch.optim as optim
import torch.nn.functional as F


def experiment(alg, env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test,
               alg_params, critic_params, policy_params, seed, log_name=None):
    print(alg.__name__)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name, 'net')

    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    critic_params = dict(network=MLP,
                         loss=F.mse_loss,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,),
                         **critic_params)

    policy = GaussianTorchPolicy(MLP,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = alg(mdp.info, policy, critic_params, **alg_params)
    core = Core(agent, mdp)

    J_list = list()
    R_list = list()
    E_list = list()

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        J_list.append(J)
        R_list.append(R)
        E_list.append(E)

        logger.save(network=agent.policy, J=J_list, R=R_list, E=E_list)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('##################################################################################################')

    print('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    # BipedalWalker
    env_id = 'BipedalWalker-v2'
    horizon = 1600
    gamma = .99

    # HopperBullet
    # env_id = 'HopperBulletEnv-v0'
    # horizon = 1000
    # gamma = .99

    # Policy Parameters
    policy_params = dict(std_0=1.,
                         size_list=[64, 64],
                         use_cuda=False)

    # Critic Parameters
    n_models_v = 1
    critic_params = dict(optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         n_models=n_models_v,
                         size_list=[64, 64])

    # Alg Parameters
    ppo_alg_params = dict(lr_p=3e-4,
                          n_epochs_v=10,
                          n_epochs_policy=10,
                          batch_size=64,
                          eps_ppo=.2,
                          lam=.95,
                          quiet=False)

    trpo_alg_params = dict(ent_coeff=0.0,
                           max_kl=.001,
                           lam=1.,
                           n_epochs_line_search=10,
                           n_epochs_cg=10,
                           cg_damping=1e-2,
                           cg_residual_tol=1e-10,
                           quiet=False)

    algs_params = [
        (TRPO, 'trpo', trpo_alg_params),
        (PPO, 'ppo', ppo_alg_params)
     ]

    for alg, alg_name, alg_params in algs_params:
        log_name = generate_log_folder(env_id, alg_name, str(n_models_v), True)
        save_parameters(log_name, dict(alg_params=alg_params, critic_params=critic_params, policy_params=policy_params))
        experiment(alg=alg, env_id=env_id, horizon=horizon, gamma=gamma,
                   n_epochs=100, n_steps=30000, n_steps_per_fit=3000, n_episodes_test=10,
                   alg_params=alg_params, critic_params=critic_params, policy_params=policy_params,
                   seed=0, log_name=log_name)
