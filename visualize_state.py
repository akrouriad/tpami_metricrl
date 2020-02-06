import os
import torch
import numpy as np
import pybullet_envs
import pybullet
import gym

from mushroom_rl.utils.viewer import ImageViewer

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


def rescale_joint(j, theta_r):
    theta_bar = 0.5 * (j.lowerLimit + j.upperLimit)
    delta = j.upperLimit - j.lowerLimit

    return delta*theta_r/2+theta_bar


def get_image(mdp, scaling, distance, pitch):
    view_matrix = mdp.env._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                               distance=distance,
                                                               yaw=mdp.env._cam_yaw,
                                                               pitch=pitch,
                                                               roll=0,
                                                               upAxisIndex=2)
    proj_matrix = mdp.env._p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(mdp.env._render_width) / mdp.env._render_height,
                                                        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = mdp.env._p.getCameraImage(width=mdp.env._render_width * scaling,
                                                 height=mdp.env._render_height * scaling,
                                                 viewMatrix=view_matrix,
                                                 projectionMatrix=proj_matrix,
                                                 renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    rgb_array = np.reshape(np.array(px), (mdp.env._render_height * scaling, mdp.env._render_width * scaling, -1))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


def set_state(robot, robot_body):
    z = cluster[0]
    v = cluster[3:6] / 0.3
    r = cluster[6]
    p = cluster[7]
    robot.feet_contact[:] = cluster[-len(robot.foot_list):]

    joint_states = cluster[8:-1]

    for i, j in enumerate(robot.ordered_joints):
        j_theta = rescale_joint(j, joint_states[2 * i])
        j_omega = joint_states[2 * i + 1] / 0.1
        j.reset_current_position(j_theta, j_omega)

    position = np.array([0, 0, z])
    orientation = pybullet.getQuaternionFromEuler([r, p, 0])
    # euler = pybullet.getEulerFromQuaternion(orientation)

    robot_body.reset_pose(position, orientation)
    current_z = robot.calc_state()[0]

    delta_z_current = z - current_z

    robot_body.reset_position([0, 0, z + delta_z_current])
    robot_body.reset_velocity(linearVelocity=v)


if __name__ == '__main__':
    #envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
    iteration = 1001
    scaling = 3
    distance = 2.5
    pitch = -5

    exp_id = 'final_medium'
    alg_name = 'metricrl'

    # env_id = 'AntBulletEnv-v0'
    # postfix = 'c10hcovr_expdTruet0.33snone'
    # run_id = 0

    # env_id = 'HopperBulletEnv-v0'
    # postfix = 'c10hcovr_expdTruet1.0snone'
    # run_id = 12
    env_id = 'HalfCheetahBulletEnv-v0'
    postfix = 'c20hcovr_expdTruet0.33snone'
    run_id = 4

    log_name = os.path.join('Results', exp_id, env_id, alg_name + '_' + postfix)
    print(log_name)

    save = True
    save_dir = os.path.join('Results', 'img', env_id)

    policy_path = os.path.join(log_name, 'net/network-' + str(run_id) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    state_reconstruction_precision = 1e-7

    client = pybullet.connect(pybullet.DIRECT)
    env = gym.make(env_id)
    env._max_episode_steps = np.inf

    viewer = ImageViewer((env.env._render_width * scaling, env.env._render_height * scaling), 1/30)

    #env.render(mode='human')
    env.reset()

    robot = env.env.robot
    robot_body = robot.robot_body

    print('- Displaying neutral position')
    px = get_image(env, scaling, distance, pitch)
    viewer.display(px)

    if save:
        full_path = os.path.join(save_dir, 'cluster-neutral.png')
        os.makedirs(save_dir, exist_ok=True)
        pygame.image.save(viewer._screen, full_path)

    wait()

    for n, cluster in enumerate(policy_torch.centers.detach().numpy()):
        # get Robot object from environment
        env.reset()
        robot = env.env.robot
        robot_body = robot.robot_body

        print('- Displaying cluster ', n)
        print(cluster)

        set_state(robot, robot_body)

        px = get_image(env, scaling, distance, pitch)
        viewer.display(px)

        if save:
            full_path = os.path.join(save_dir, 'cluster-' + str(n) + '.png')
            os.makedirs(save_dir, exist_ok=True)
            pygame.image.save(viewer._screen, full_path)

        wrong_state = np.any((cluster - robot.calc_state()) > state_reconstruction_precision)
        print('wrong state? ', wrong_state)

        if wrong_state:
            print(cluster - robot.calc_state())

        wait()
