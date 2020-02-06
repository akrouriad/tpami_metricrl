import os
import torch
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc

from mushroom_rl.algorithms import Agent
from mushroom_rl.core import Core
from metric_rl.gym_fixed import GymFixed
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.torch import to_float_tensor


class DummyAgent(Agent):
    def __init__(self, torch_policy, dt):
        self._regressor = torch_policy
        self._dt = dt

    def draw_action(self, state):

        time.sleep(self._dt)
        with torch.no_grad():
            s = to_float_tensor(np.atleast_2d(state), False)

            mu, chol_sigma = self._regressor(s)
            dist = torch.distributions.MultivariateNormal(mu, scale_tril=chol_sigma)
            a = dist.sample()

            return torch.squeeze(a, dim=0).detach().cpu().numpy()

    def episode_start(self):
        pass

    def fit(self, dataset):
        pass


def get_cluster_activation(env_id, horizon, gamma, torch_policy, dt, n_episodes, seed):
    print('Metric RL')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mdp = GymFixed(env_id, horizon, gamma)
    if dt > 0:
        mdp.render()

    # Set environment seed
    mdp.env.seed(seed)

    # Set agent
    agent = DummyAgent(torch_policy, dt)

    # Set experiment
    core = Core(agent, mdp)
    dataset = core.evaluate(n_episodes=n_episodes, quiet=True)

    s, *_ = parse_dataset(dataset)
    print(len(s))
    w = agent._regressor.get_membership(torch.tensor(s))

    return w.detach().numpy()

def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


def plot_activations(filename, w, idxs, imgs, display):
    fig, axes = plt.subplots(len(idxs) + 2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [3.5, 1]})
    set_share_axes(axes[:, 0], sharex=True, sharey=True)
    fig.delaxes(axes[-1, 1])
    fig.delaxes(axes[-2, 1])

    lines = list()
    labels = list()
    c = plt.rcParams['axes.prop_cycle']()

    for i, idx in enumerate(idxs):
        w_idx = w[:, idx]
        lines += axes[i, 0].plot(w_idx, color=c.__next__()['color'])
        axes[i, 0].set_ylabel('$w(s_t)$')
        labels.append('cluster ' + str(idx))
        axes[i, 1].imshow(imgs[i])
        axes[i, 1].get_xaxis().set_visible(False)
        axes[i, 1].get_yaxis().set_visible(False)

    w_total = np.sum(w, axis=1)
    w_others = w_total - np.sum(w[:, idxs], axis=1)
    w_default = 1 - w_total

    lines += axes[-2, 0].plot(w_others, color=c.__next__()['color'])
    labels.append('other clusters')
    axes[-2, 0].set_ylabel('$w(s_t)$')
    lines += axes[-1, 0].plot(w_default, color=c.__next__()['color'])
    labels.append('default cluster')
    axes[-1, 0].set_ylabel('$w(s_t)$')
    axes[-1, 0].set_xlabel('$t$')

    plt.figlegend(lines, labels, ncol=len(idxs)+2, loc='lower center',
                  shadow=False, frameon=False)
    plt.subplots_adjust(left=0.08, bottom=0.10, right=0.99, top=0.99, wspace=0)

    plt.savefig(filename, bbox_inches='tight')

    if display:
        plt.show()


def load_cluster_images(env_id, idxs):
    imgs = list()
    for idx in idxs:
        filename = 'cluster-' + str(idx) + '.png'
        file_path = os.path.join('Results', 'img', env_id, filename)
        img = mpimg.imread(file_path)
        imgs.append(img)

    return imgs


def load_policy(log_name, iteration, seed):
    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


if __name__ == '__main__':
    rc('text', usetex=True)

    dt = 0.
    horizon = 1000
    gamma = .99

    # env_id = 'AntBulletEnv-v0'
    # log_name = 'Results/final_medium/AntBulletEnv-v0/metricrl_c10hcovr_expdTruet0.33snone'
    # seed = 0

    env_id = 'HopperBulletEnv-v0'
    log_name = 'Results/final_medium/HopperBulletEnv-v0/metricrl_c10hcovr_expdTruet1.0snone'
    seed = 12

    n_cluster = 3
    max_time = 350

    save_path = os.path.join('Results', 'plots', 'activations',)
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, env_id + '.png')

    policy = load_policy(log_name, iteration=1001, seed=12)

    w = get_cluster_activation(env_id, horizon, gamma, policy, dt, n_episodes=1, seed=12)

    w = w[:max_time, :]

    w_mean = np.mean(w, axis=0)
    idxs = w_mean.argsort()[::-1][:n_cluster]
    print('w:', w_mean)
    print('idx', idxs)

    sorted_idxs = np.sort(idxs)

    imgs = load_cluster_images(env_id, sorted_idxs)

    plot_activations(filename, w, sorted_idxs, imgs, True)
