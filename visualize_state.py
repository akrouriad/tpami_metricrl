import os
import torch
import numpy as np
import pybullet_envs
import pybullet
import gym


def rescale_joint(j, theta_r):
    theta_bar = 0.5 * (j.lowerLimit + j.upperLimit)
    delta = j.upperLimit - j.lowerLimit

    return delta*theta_r/2+theta_bar


def set_state_hopper(robot, robot_body):
    z = cluster[0]
    v = cluster[3:6] / 0.3
    r = cluster[6]
    p = cluster[7]
    robot.feet_contact[0] = cluster[-1]

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
    iteration = 1001

    exp_id = 'entropy'
    env_id = 'HopperBulletEnv-v0'
    alg_name = 'metricrl_c40'

    log_name = os.path.join('Results', exp_id, env_id, alg_name)
    print(log_name)


    policy_path = os.path.join(log_name, 'net/network-5-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    state_reconstruction_precision = 1e-7

    client = pybullet.connect(pybullet.DIRECT)
    env = gym.make(env_id)
    env._max_episode_steps = np.inf

    env.render(mode='human')
    env.reset()

    robot = env.env.robot
    robot_body = robot.robot_body

    print('- Displaying neutral position')
    input()

    for n, cluster in enumerate(policy_torch.centers.detach().numpy()):
        # get Robot object from environment
        env.reset()
        robot = env.env.robot
        robot_body = robot.robot_body

        print('- Displaying cluster ', n)
        print(cluster)

        set_state_hopper(robot, robot_body)

        wrong_state = np.any((cluster - robot.calc_state()) > state_reconstruction_precision)
        print('wrong state? ', wrong_state)

        if wrong_state:
            print(cluster - robot.calc_state())

        input()
