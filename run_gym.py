import torch
import numpy as np

from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.utils.dataset import compute_J
from proj_metricrl_msh import ProjectionMetricRL

from tqdm import tqdm, trange

def experiment(env_id, n_epochs, n_steps, n_steps_per_fit, n_steps_test, seed, params):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    horizon = 3000
    gamma = .99
    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    agent = ProjectionMetricRL(mdp.info, **params)

    core = Core(agent, mdp)

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        J = compute_J(dataset, mdp.info.gamma)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: ' + str(np.mean(J)))
        tqdm.write('#######################################################################################################')

    print('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    params = dict(std_0=1.0,
                  lr_v=3e-4,
                  lr_p=1e-3,
                  lr_cw=1e-1,
                  max_kl=.015,
                  e_reduc=.015,
                  nb_epochs_v=10,
                  nb_epochs_clus=20,
                  nb_epochs_params=20,
                  batch_size=64,
                  nb_max_clusters=10
                  )

    # experiment(env_id='HopperBulletEnv-v0', n_epochs=100, n_steps=30000, n_steps_per_fit=3000, n_steps_test=3000,
    #            seed=0, **params)
    experiment(env_id='BipedalWalker-v2', n_epochs=100, n_steps=30000, n_steps_per_fit=3000, n_steps_test=3000,
               seed=0, params=params)
