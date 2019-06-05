import torch
import gym
from policies import MetricPolicy
import data_handling as dat

env = gym.make('BipedalWalker-v2')
a_dim = env.action_space.shape[0]
pol = MetricPolicy(a_dim)

state_dict = torch.load('experiments/cluscomp/BipedalWalker-v2sampIt3000nb_v2normaNoneaggregMinmts3.0mmax_clus5run0-881.pth')
pol.load_state_dict(state_dict)
dat.rollouts(env, pol, min_trans=1, render=True)

