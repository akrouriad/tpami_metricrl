import torch
import numpy as np
from tqdm import tqdm, trange

from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.utils.dataset import compute_J

from metric_rl.ppo import PPO
from metric_rl.logger import generate_log_folder, save_parameters, Logger


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

    agent = PPO(mdp.info, **params)

    core = Core(agent, mdp)

    J_list = list()
    R_list = list()
    E_list = list()

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent._policy_torch.entropy()

        J_list.append(J)
        R_list.append(R)
        E_list.append(E)

        logger.save(network=agent._policy_torch, J=J_list, R=R_list, E=E_list)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: {}, R: {}, entropy: {}'.format(J, R, E))
        tqdm.write('##################################################################################################')

    print('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    max_kl = .015
    n_models_v = 1

    params = dict(std_0=1.0,
                  lr_v=3e-4,
                  lr_p=3e-4,
                  n_epochs_v=10,
                  n_models_v=n_models_v,
                  v_prediction_type='min',
                  lam=0.95,
                  n_epochs_policy=10,
                  batch_size=64,
                  eps_ppo=.2
                  )

    # Bipedal Walker
    log_name = generate_log_folder('bipedal_walker', 'ppo', str(n_models_v), True)
    save_parameters(log_name, params)
    experiment(env_id='BipedalWalker-v2', horizon=1600, gamma=.99, n_epochs=100, n_steps=30000, n_steps_per_fit=3000,
               n_episodes_test=10, seed=0, params=params, log_name=log_name)


    # Hopper Bullet
    # log_name = generate_log_folder('hopper_bullet', 'ppo', str(n_models_v), True)
    # save_parameters(log_name, params)
    # experiment(env_id='HopperBulletEnv-v0', horizon=1000, gamma=.99, n_epochs=100, n_steps=30000, n_steps_per_fit=3000,
    #            n_episodes_test=10, seed=0, params=params, log_name=log_name)