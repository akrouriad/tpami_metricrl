import os
import torch
import numpy as np
import pybullet_envs
import pybullet
import gym

from mushroom_rl.utils.viewer import ImageViewer

import pygame
from pygame.locals import *


def set_state(env, cluster, action):
    env.env.state = cluster
    env.env.last_u = action/2


def wait():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return

            if event.type == KEYDOWN and event.key == K_RETURN:
                return


if __name__ == '__main__':
    log_name = 'Results/final_small2/Pendulum-v0/metricrl_c5hcovr_expdTruet1.0snone'
    env_id = 'Pendulum-v0'
    run_id = 15
    iteration = 501

    save = True
    save_dir = os.path.join('Results', 'img', env_id)
    save_dir_gifs = os.path.join(save_dir, 'gifs')

    policy_path = os.path.join(log_name, 'net/network-' + str(run_id) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    state_reconstruction_precision = 1e-7

    client = pybullet.connect(pybullet.DIRECT)
    env = gym.make(env_id)
    env._max_episode_steps = np.inf

    viewer = ImageViewer((500, 500), 1 / 30)

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

        print(cluster, action)

        print('- Displaying cluster ', n)
        print(cluster)

        set_state(env, cluster, action)
        img = env.render(mode='rgb_array')
        viewer.display(img)

        if save:
            full_path = os.path.join(save_dir, 'cluster-' + str(n) + '.png')
            os.makedirs(save_dir, exist_ok=True)
            pygame.image.save(viewer._screen, full_path)
        wait()
