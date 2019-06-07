import numpy as np
import torch
from time import sleep
sleep(0.05)


def next_batch_idx(batch_size, data_set_size):
    batch_idx_list = np.random.choice(data_set_size, data_set_size, replace=False)
    for batch_start in range(0, data_set_size, batch_size):
        yield batch_idx_list[batch_start:min(batch_start + batch_size, data_set_size)]


def rollout(env, policy, render=False):
    # Generates transitions until episode's end
    obs = env.reset()
    done = False
    while not done:
        if render:
            sleep(1/60)
            env.render()
        act = policy(obs)
        # print(act)
        nobs, rwd, done, _ = env.step(act)
        yield obs, act, rwd, done
        obs = nobs


def rollouts(env, policy, min_trans, render=False):
    # Keep calling rollout and saving the resulting path until at least min_trans transitions are collected
    keys = ['obs', 'act', 'rwd', 'done']  # must match order of the yield above
    paths = {}
    for k in keys:
        paths[k] = []
    while len(paths['rwd']) < min_trans:
        for trans_vect in rollout(env, policy, render):
            for key, val in zip(keys, trans_vect):
                paths[key].append(val)
    for key in set(keys):
        paths[key] = np.asarray(paths[key])
        if paths[key].ndim == 1:
            paths[key] = np.expand_dims(paths[key], axis=-1)
    return paths


def torch_copy_get(x):
    l = []
    for p in x.parameters():
        l.append(p.clone().detach())
    return l


def torch_copy_set(x, l):
    for p, c in zip(x.parameters(), l):
        p.data = c.data
