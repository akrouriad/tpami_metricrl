import torch
import torch.nn.functional as F
import numpy as np
import pybullet_envs
import pybullet
import gym
import data_handling as dat
import rllog
from rl_shared import MLP, RunningMeanStdFilter, ValueFunction, ValueFunctionList
import rl_shared as rl
from policies import MetricPolicy
import gaussian_proj as proj
from cluster_weight_proj import cweight_mean_proj, ls_cweight_mean_proj
import time


def replay(envid, log_name, iteration, seed=0, min_sample_per_iter=3000):
    print('Metric RL Replay')
    print('envid:', envid)
    print('log file:', log_name)
    #print('Params: nb_vfunc {} norma {} aggreg_type {} max_ts {} seed {} log_name {}'.format(nb_vfunc, norma, aggreg_type, max_ts, seed, log_name))

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

    policy_torch = torch.load(log_name + '-' + str(iteration) + '.pth')

    print('cluster centers:')
    for cluster in  policy_torch.centers.detach().numpy():
        print(cluster[4:7:2]/np.pi*180.0, cluster[8:12:2]/np.pi*180.0)

    policy = lambda obs: torch.squeeze(policy_torch(torch.tensor(obs, dtype=torch.float)), dim=0).detach().numpy()

    while True:
        dat.rollouts(env, policy, min_sample_per_iter, render=True)


if __name__ == '__main__':
    replay(envid='BipedalWalker-v2', log_name='clus5', iteration=194, seed=0)
    #replay(envid='HopperBulletEnv-v0', log_name='exp_bip', seed=0)
