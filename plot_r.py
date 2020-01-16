import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os


def get_mean_and_confidence(data):
    mean = np.mean(data, axis=0)
    se = st.sem(data, axis=0)
    n = len(data)
    interval, _ = st.t.interval(0.95, n-1, scale=se)
    return mean, interval

res_folder = './Results/'

envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
nb_runs = 11
n_epochs = 1000
median = False

all_par = []
n_clusterss = [10, 20, 40]
a_cost_scales = [0.]#, 10.]
alg_name = 'metricrl'
xp_name = 'acost'


def plot_data(x, median):
    if median:
        plt.plot(range(n_epochs), np.median(x, axis=0).squeeze())
        plt.fill_between(range(n_epochs), np.quantile(x, axis=0, q=.25), np.quantile(x, axis=0, q=.75), alpha=.5)
    else:
        mean, interval = get_mean_and_confidence(x)
        print(mean.shape)
        print(interval.shape)
        plt.plot(range(n_epochs), mean)
        plt.fill_between(range(n_epochs), mean - interval, mean + interval, alpha=.5)



# Creating parameters tables
for env in envs:
    all_perfs = []
    all_entropy = []
    alg_labels = []
    for n_clusters in n_clusterss:
        for a_cost_scale in a_cost_scales:
            alg_label = 'c' + str(n_clusters) + 'a' + str(a_cost_scale)
            all_perfs_a = []
            all_entropy_a = []
            for run in range(nb_runs):
                postfix = '_' + alg_label
                j_filename = 'J-' + str(run) + '.npy'
                r_filename = 'R-' + str(run) + '.npy'
                e_filename = 'E-' + str(run) + '.npy'
                r_name = os.path.join(res_folder, env, alg_name + postfix, r_filename)
                e_name = os.path.join(res_folder, env, alg_name + postfix, e_filename)

                try:
                    R = np.load(r_name)
                    E = np.load(e_name)
                    if R.shape[0] != 1000:
                       print(r_name, 'wrong shape')
                    else:
                        all_perfs_a.append(R)
                        all_entropy_a.append(E)
                except:
                    print(r_name, 'bad file')

            all_perfs.append(all_perfs_a)
            all_entropy.append(all_entropy_a)
            alg_labels.append(alg_label)


    plt.figure()
    legend_lab = []
    for n, R in zip(alg_labels, all_perfs):
        R = np.array(R)
        print('shape:', R.shape)
        plot_data(R, median)
        legend_lab.append(n)

    plt.legend(legend_lab)
    plt.suptitle(env)

    fig_name = 'R_' + env + '.png'
    fig_path = os.path.join(res_folder, 'plots', xp_name)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.clf()

    plt.figure()
    legend_lab = []
    for n, E in zip(alg_labels, all_entropy):
        E = np.array(E)
        print('shape:', E.shape)
        plot_data(E, median)
        legend_lab.append(n)

    plt.legend(legend_lab)
    plt.suptitle(env)

    fig_name = 'E_' + env + '.png'
    fig_path = os.path.join(res_folder, 'plots', xp_name)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.clf()

# nb_run = 11
# suff = [''] + ['-' + str(k+1) for k in range(nb_run - 1)]
# algs = ['log/BipedalWalker-v2/projection_rand1k10_2019-12-18_17-47-04_10/', 'log/BipedalWalker-v2/projection_randadv_2019-12-18_17-54-05_5/', 'log/BipedalWalker-v2/projection_rand_lessfreq_2019-12-16_17-44-56_5/', 'log/BipedalWalker-v2/projection_rand10k_2019-12-16_17-48-57_5/', 'log/BipedalWalker-v2/projection_random1K_2019-12-13_17-55-11_5/', 'log/BipedalWalker-v2/projection_del_2019-12-13_18-06-10_5/']
# # algs = ['log/BipedalWalker-v2/projection_rand1k10_2019-12-18_17-47-04_10/', 'log/BipedalWalker-v2/projection_randadv_2019-12-18_17-54-05_5/']
# names = ['cov1k10', 'randadv', 'lessfreq', 'swap10k', 'swap', 'del']
# # names = ['cov1k10', 'randadv']
#
# all_algs = []
# for alg in algs:
#     all_iters = []
#     for k in range(nb_run):
#         all_iters.append(np.load(alg + 'R' + suff[k] + '.npy'))
#     all_algs.append(all_iters)
#
# all_algs = np.array(all_algs)
#
# for k, name in enumerate(names):
#     print(name, np.median(all_algs[k], axis=0))
#     plt.plot(range(all_algs.shape[2]), np.median(all_algs[k], axis=0))
#     plt.fill_between(range(all_algs.shape[2]), np.quantile(all_algs[k], axis=0, q=.25), np.quantile(all_algs[k], axis=0, q=.75), alpha=.5)
# plt.legend(names)
# plt.show()
