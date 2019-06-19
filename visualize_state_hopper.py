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


if __name__ == '__main__':
    log_name = 'Results/PyBulletHopper/projection_2019-06-07_15-37-01_5'
    iteration = 929

    state_reconstruction_precision = 1e-7

    pybullet.connect(pybullet.DIRECT)
    env = gym.make('HopperBulletEnv-v0')

    env.render(mode='human')

    policy_path = os.path.join(log_name, 'net/network-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

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

        z = cluster[0]
        v = cluster[3:6]/0.3
        r = cluster[6]
        p = cluster[7]
        robot.feet_contact[0] = cluster[-1]
        # print('height:', z)
        # print('velocity: ', v)
        # print('orientation: ', r, p, 0)


        joint_states = cluster[8:-1]
        # print(orientation, euler, [r, p, 0])

        for i, j in enumerate(robot.ordered_joints):

            j_theta = rescale_joint(j, joint_states[2*i])
            j_omega = joint_states[2*i+1]/0.1
            #print('joint: ', j_theta, j_omega)
            j.reset_current_position(j_theta, j_omega)

        #parts_xyz = np.array([p.pose().xyz() for p in robot.parts.values()]).flatten()

        position = np.array([0, 0, z])
        orientation = pybullet.getQuaternionFromEuler([r, p, 0])
        euler = pybullet.getEulerFromQuaternion(orientation)

        robot_body.reset_pose(position, orientation)
        current_z = robot.calc_state()[0]

        delta_z_current = z - current_z

        robot_body.reset_position([0, 0, z+delta_z_current])
        robot_body.reset_velocity(linearVelocity=v)

        wrong_state = np.any((cluster - robot.calc_state()) > state_reconstruction_precision)
        print('wrong state? ', wrong_state)
        if(wrong_state):
            print(cluster - robot.calc_state())

        input()
