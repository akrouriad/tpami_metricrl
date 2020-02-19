import os
import torch
import numpy as np
import pickle

from mayavi import mlab
from mushroom_rl.utils.dataset import parse_dataset


def load_policy(log_name, iteration, seed):
    policy_path = os.path.join(log_name, 'net/network-' + str(seed) + '-' + str(iteration) + '.pth')
    policy_torch = torch.load(policy_path)

    return policy_torch


def generate_memberships(policy, n_samples):
    theta = np.linspace(-np.pi, np.pi, n_samples)
    theta_dot = np.linspace(-8, 8, n_samples)

    theta, theta_dot = np.meshgrid(theta, theta_dot)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    s = np.stack([cos_theta, sin_theta, theta_dot], axis=-1)
    s_flat = s.reshape(-1, 3)

    w_flat = policy.get_membership(torch.tensor(s_flat)).detach().numpy()
    w = w_flat.reshape(n_samples, n_samples, policy.n_clusters)

    w_default = np.expand_dims(1-np.sum(w, axis=-1), axis=-1)
    w = np.concatenate([w,w_default], axis=-1)

    return theta, theta_dot, w


def plot_membership_mesh(theta, theta_dot, w, s, transparency=False):

    colors = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys']
    fig = mlab.figure(size=(500, 500))

    ax_ranges = [-np.pi, np.pi, -8, 8, 0, 1]
    ax_scale = [1/np.pi, 1/8, 2.0]
    ax_extent = ax_ranges * np.repeat(ax_scale, 2)

    x = s[:, 0] + np.pi
    y = s[:, 1]
    z = np.zeros_like(x)-0.023
    l = mlab.plot3d(x, y, z, tube_radius=0.025, extent=ax_extent)
    l.actor.actor.scale = ax_scale

    for i in range(w.shape[-1]):
        surf = mlab.surf(theta.T, theta_dot.T, w[:, :, i].T, colormap=colors[i], extent=ax_extent)
        surf.actor.actor.scale = ax_scale

        if transparency:
            surf.actor.property.opacity = 0.8
            fig.scene.renderer.use_depth_peeling = 1

    mlab.view(azimuth=-45, elevation=25, distance=8)
    mlab.axes(extent=ax_extent,
              ranges=ax_ranges,
              xlabel='theta', ylabel='theta_dot', zlabel='w')

    mlab.show()


def parse_states(states):
    compressed_states = np.empty((len(states), 2))
    for i in range(len(states)):
        cluster = states[i]

        cos_theta = cluster[0]
        sin_theta = cluster[1]

        theta = np.arctan2(sin_theta, cos_theta)
        theta_dot = cluster[-1]

        compressed_state = np.array([theta, theta_dot])

        compressed_states[i] = compressed_state

    return compressed_states


if __name__ == '__main__':
    log_name = 'Results/final_small2/Pendulum-v0/metricrl_c5hcovr_expdTruet1.0snone'
    seed = 15

    policy = load_policy(log_name, iteration=501, seed=seed)
    theta, theta_dot, w = generate_memberships(policy, n_samples=1000)
    centers = parse_states(policy.centers)

    with open('dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)

    s, *_ = parse_dataset(dataset)
    s_new = parse_states(s)

    plot_membership_mesh(theta, theta_dot, w, s_new, transparency=True)






