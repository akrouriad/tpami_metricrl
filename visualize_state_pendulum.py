import os
import torch
import numpy as np
import pybullet_envs
import pybullet
import gym

from mushroom_rl.utils.viewer import Viewer

import pygame
from pygame.locals import *


def wait():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

            if event.type == KEYDOWN and event.key == K_RETURN:
                return


def obs_to_state(obs):
    cos_theta = obs[0]
    sin_theta = obs[1]
    theta = np.arctan2(sin_theta, cos_theta)
    theta_dot = obs[2]

    state = np.array([theta, theta_dot])

    return state


def set_state(env, cluster, action):
    state = obs_to_state(cluster)

    env.env.state = state
    env.env.last_u = action/2


def visualize_cluster(viewer, env, cluster):
    set_state(env, cluster, action)
    img = env.render(mode='rgb_array')
    transposed = np.transpose(img, (1, 0, 2))
    viewer.background_image(transposed)

    center = [5, 5]
    torque = cluster[2]
    max_torque = 8
    max_radius = 5

    viewer.torque_arrow(center, torque, max_torque,
                        max_radius, color=(255, 0, 0), width=5)

    pygame.display.flip()


if __name__ == '__main__':
    env_id = 'Pendulum-v0'
    # log_name = 'Results/final_small2/Pendulum-v0/metricrl_c5hcovr_expdTruet1.0snone'
    # seed = 15
    log_name = 'Results/final_small2/Pendulum-v0/metricrl_c10hcovr_expdTruet1.0snone'
    seed = 3
    iteration = 501

    save = True
    save_dir = os.path.join('Results', 'img', env_id)
    save_dir_gifs = os.path.join(save_dir, 'gifs')

    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    state_reconstruction_precision = 1e-7

    client = pybullet.connect(pybullet.DIRECT)
    env = gym.make(env_id)
    env._max_episode_steps = np.inf

    viewer = Viewer(10, 10)


    env.reset()
    env.step(np.zeros(1))

    if save:
        full_path = os.path.join(save_dir, 'cluster-neutral.png')
        os.makedirs(save_dir, exist_ok=True)

    for n in range(policy_torch.n_clusters):
        # get Robot object from environment
        cluster = policy_torch.centers[n].detach().numpy()
        action = policy_torch.means[n].detach().numpy()
        env.reset()

        print('- Displaying cluster ', n)
        print(obs_to_state(cluster), action)

        visualize_cluster(viewer, env, cluster)

        if save:
            full_path = os.path.join(save_dir, 'cluster-' + str(n) + '.png')
            os.makedirs(save_dir, exist_ok=True)
            pygame.image.save(viewer._screen, full_path)
        wait()
