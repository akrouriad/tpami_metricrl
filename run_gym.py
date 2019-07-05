import torch
import numpy as np

from mushroom.core import Core
from mushroom.environments import Gym
from mushroom.utils.dataset import compute_J
from proj_metricrl import ProjectionMetricRL

from tqdm import tqdm, trange

def experiment(env_id, horizon, gamma, n_epochs, n_steps, n_steps_per_fit, n_episodes_test, seed, params):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mdp = Gym(env_id, horizon, gamma)

    # Set environment seed
    mdp.env.seed(seed)

    agent = ProjectionMetricRL(mdp.info, **params)

    core = Core(agent, mdp)

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        J = compute_J(dataset, mdp.info.gamma)
        R = compute_J(dataset)

        tqdm.write('END OF EPOCH ' + str(it))
        tqdm.write('J: ' + str(np.mean(J)) + ', R: ' + str(np.mean(R)))
        tqdm.write('#######################################################################################################')

    print('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    max_kl = .015
    params = dict(std_0=1.0,
                  lr_v=3e-4,
                  lr_p=1e-3,
                  lr_cw=1e-1,
                  max_kl=max_kl,
                  max_kl_cw=max_kl / 2.,
                  max_kl_cdel=2 * max_kl / 3.,
                  e_reduc=.015,
                  n_epochs_v=10,
                  n_models_v=2,
                  v_prediction_type='min',
                  lam=0.95,
                  n_epochs_clus=20,
                  n_epochs_params=20,
                  batch_size=64,
                  n_max_clusters=5
                  )

    # experiment(env_id='HopperBulletEnv-v0', horizon=1000, gamma=.99, n_epochs=100,
    #            n_steps=30000, n_steps_per_fit=3000, n_steps_test=3000, seed=0, **params)

    experiment(env_id='BipedalWalker-v2', horizon=1600, gamma=.99, n_epochs=100,
               n_steps=30000, n_steps_per_fit=3000, n_episodes_test=10, seed=0, params=params)
