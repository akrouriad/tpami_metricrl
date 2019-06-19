import os
import torch
import numpy as np
import pybullet_envs
import gym
import data_handling as dat
import pickle


def replay(log_name, iteration, envid='', seed=0, n_epochs=1, min_sample_per_iter=3000):
    print('Metric RL Replay')
    print('log file:', log_name)

    parameters_file = os.path.join(log_name, 'parameters.pkl')

    if not envid:
        with open(parameters_file, 'rb') as f:
            params_dict = pickle.load(f)
            envid = params_dict['envid']

    print('envid:', envid)

    if '- ' + envid in pybullet_envs.getList():
        import pybullet
        pybullet.connect(pybullet.DIRECT)
    env = gym.make(envid)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    if '- ' + envid in pybullet_envs.getList():
        env.render(mode='human')

    policy_path = os.path.join(log_name, 'net/network-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    print('cluster centers:')
    for cluster in policy_torch.centers.detach().numpy():
        if envid == 'BipedalWalker-v2':
            print(cluster[4:7:2]/np.pi*180.0, cluster[8:12:2]/np.pi*180.0)
        else:
            print(cluster)

    print('clusters means:')

    for active_cluster in policy_torch.active_cluster_list:
        print(policy_torch.means_list[active_cluster].detach().numpy())

    policy = lambda obs: torch.squeeze(policy_torch(torch.tensor(obs, dtype=torch.float)), dim=0).detach().numpy()

    for it in range(n_epochs):
        dat.rollouts(env, policy, min_sample_per_iter, render=True)


if __name__ == '__main__':
    # replay(envid='BipedalWalker-v2', log_name='clus5', iteration=194, seed=0)
    #log_name = 'log/hopper_bul/projection_2019-06-10_11-54-56_5'
    log_name = 'Results/PyBulletHopper/projection_2019-06-07_15-37-01_5'
    print('replaying from folder: ', log_name)
    replay(log_name=log_name, iteration=929, seed=0, envid='HopperBulletEnv-v0')
    # replay(log_name=log_name, iteration=30, seed=0) # add envid='HopperBulletEnv-v0' if parameters.pkl is missing
    # import roboschool
    # replay(envid='RoboschoolHopper-v1', seed=0, log_name='hopp5', iteration=85)
