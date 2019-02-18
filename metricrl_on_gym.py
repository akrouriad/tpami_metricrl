import gym
import data_handling as dat

def learn(envid):
    env = gym.make(envid)
    policy = lambda obs: env.action_space.sample()
    dat.rollouts(env, policy, 1, render=True)

if __name__ == '__main__':
    learn(envid='MountainCarContinuous-v0')