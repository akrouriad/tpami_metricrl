import numpy as np
import matplotlib.pyplot as plt
import os

res_folder = './results/'

envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
nb_runs = 11
n_epochs = 1000

all_par = []
n_clusterss = [10, 20, 40]
a_cost_scales = [0., 100.]
alg_name = 'metricrl'
xp_name = 'acost'
# Creating parameters tables
for env in envs:
    all_perfs = []
    alg_labels = []
    for n_clusters in n_clusterss:
        for a_cost_scale in a_cost_scales:
            alg_label = 'c' + str(n_clusters) + 'a' + str(a_cost_scale)
            all_perfs_a = []
            for run in range(nb_runs):
                postfix = '_' + alg_label + 'r' + str(run)
                filename = 'R'
                filename += '-' + str(run) + '.npy'
                r_name = os.path.join(res_folder, env, alg_name + postfix, filename)

                try:
                    R = np.load(r_name)
                    if R.shape[0] != 1000:
                       print(r_name, 'wrong shape')
                    else:
                        all_perfs_a.append(R)
                except:
                    print(r_name, 'bad file')

            all_perfs.append(all_perfs_a)
            alg_labels.append(alg_label)


    plt.figure()
    legend_lab = []
    for n, perf in zip(alg_labels, all_perfs):
        perf = np.array(perf)
        print('shape:', perf.shape)
        if perf.shape[0] > 0:
            plt.plot(range(n_epochs), np.median(perf, axis=0).squeeze())
            plt.fill_between(range(n_epochs), np.quantile(perf, axis=0, q=.25), np.quantile(perf, axis=0, q=.75), alpha=.5)
            legend_lab.append(n)
    plt.legend(legend_lab)
    # plt.show()
    plt.savefig(xp_name + env + '.png')
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
