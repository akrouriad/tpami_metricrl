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


def plot_data(x, use_median):
    if use_median:
        median = np.median(x, axis=0).squeeze()
        epochs = range(len(median))
        plt.plot(epochs, median)
        plt.fill_between(epochs, np.quantile(x, axis=0, q=.25), np.quantile(x, axis=0, q=.75), alpha=.5)
    else:
        mean, interval = get_mean_and_confidence(x)
        epochs = range(len(mean))
        plt.plot(epochs, mean)
        plt.fill_between(epochs, mean - interval, mean + interval, alpha=.5)


def create_figure(res_folder, subfolder, name, env, all_data, alg_labels, use_median=False,
                  legend=False, display=False):
    plt.figure()
    legend_lab = []
    for n, d in zip(alg_labels, all_data):
        d = np.array(d)
        plot_data(d, use_median)
        legend_lab.append(n)

    plt.grid(linestyle='dotted')

    if legend:
        plt.legend(legend_lab, fontsize='small', frameon=False)

    plt.xlabel('Epoch')
    plt.ylabel(name)

    fig_name = name + '_' + env
    fig_path = os.path.join(res_folder, 'plots', subfolder)
    os.makedirs(fig_path, exist_ok=True)
    full_path = os.path.join(fig_path, fig_name + '.png')
    plt.savefig(full_path,  bbox_inches='tight')
    if display:
        plt.suptitle(env)
        plt.show()

    plt.clf()


def load_data_base(res_folder, env, subfolder, nb_runs, nb_epochs, entropy=True, add_prop_id=False):
    all_r_a = []
    all_j_a = []
    all_entropy_a = []

    for run in range(nb_runs):
        j_filename = 'J-' + str(run) + '.npy'
        r_filename = 'R-' + str(run) + '.npy'
        e_filename = 'E-' + str(run) + '.npy'

        print(res_folder, env, subfolder, j_filename)
        prop_id = ''
        if add_prop_id:
            prop_id = 'env_id_'
        j_name = os.path.join(res_folder, prop_id+env, subfolder, j_filename)
        r_name = os.path.join(res_folder, prop_id+env, subfolder, r_filename)
        e_name = os.path.join(res_folder, prop_id+env, subfolder, e_filename)

        try:
            J = np.load(j_name)
            R = np.load(r_name)
            do_add = True
            if entropy:
                E = np.load(e_name)
            if R.shape[0] != nb_epochs + 1:
                print(r_name, 'wrong shape')
                print(R.shape, J.shape)
                if R.shape[0] < nb_epochs + 1:
                    print('discarding run')
                    do_add = False
                else:
                    J = J[:nb_epochs+1]
                    R = R[:nb_epochs+1]
            if do_add:
                all_j_a.append(J)
                all_r_a.append(R)
                if entropy:
                    all_entropy_a.append(E)
        except:
            print(r_name, 'bad file')

    if entropy:
        return all_j_a, all_r_a, all_entropy_a
    else:
        return all_j_a, all_r_a


def load_data_baselines(res_folder, env, alg_names, nb_runs, nb_epochs, entropy=False, add_prop_id=False):
    all_j = []
    all_r = []

    for alg_name in alg_names:
        if add_prop_id:
            subfolder = 'alg_name_' + alg_name
        else:
            subfolder = alg_name

        all_j_a, all_r_a = load_data_base(res_folder, env, subfolder, nb_runs, nb_epochs, entropy=entropy, add_prop_id=add_prop_id)

        all_j.append(all_j_a)
        all_r.append(all_r_a)

    return all_j, all_r


def load_data_fixedtemp(res_folder, env, alg_name, n_clusterss, nb_runs, nb_epochs, clus_sels, clus_dels, temps, add_prop_id=False):
    all_j = []
    all_r = []
    all_entropy = []
    alg_labels = []

    for n_clusters in n_clusterss:
        for clus_sel in clus_sels:
            for clus_del in clus_dels:
                for temp in temps:
                    alg_label = 'c' + str(n_clusters) + 'h' + clus_sel + 'd' + str(clus_del) + 't' + str(temp) + 'snone'
                    subfolder = alg_name + '_' + alg_label
                    print(subfolder)
                    all_j_a, all_r_a, all_entropy_a = load_data_base(res_folder, env, subfolder, nb_runs, nb_epochs, add_prop_id=add_prop_id)

                    all_j.append(all_j_a)
                    all_r.append(all_r_a)
                    all_entropy.append(all_entropy_a)
                    alg_labels.append(alg_label)

    return all_j, all_r, all_entropy, alg_labels


if __name__ == '__main__':

    plot_baselines = True
    plot_diffproto = True

    # xp_name = 'final_small2'
    # envs = ['BipedalWalker-v2', 'Pendulum-v0', 'InvertedPendulumSwingupBulletEnv-v0', 'InvertedPendulumBulletEnv-v0']
    # temps = [1., 1., 1., 1.]
    # n_clusterss = [5, 10, 20]
    # nb_epochs = 500
    # alg_labels = ['Metric-5', 'Metric-10', 'Metric-20', 'PPO', 'TRPO (MLP)', 'TRPO (Linear)']

    xp_name = 'final_medium'
    # xp_name = 'med_diff'
    # envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
    envs = ['AntBulletEnv-v0']
    # envs = ['HalfCheetahBulletEnv-v0']
    temps = [1., 1., .33, .33]
    # n_clusterss = [10, 20, 40]
    n_clusterss = [10]
    nb_epochs = 1000
    # alg_labels = ['Metric-10', 'Metric-20', 'Metric-40', 'PPO', 'TRPO (MLP)', 'TRPO (Linear)']
    # alg_labels = ['Metric-10', 'PPO-DiffMetric10', 'TRPO-DiffMetric10', 'PPO', 'TRPO (MLP)']
    alg_labels = ['Metric-10', 'TRPO-DiffProto', 'PPO-DiffProto', 'TRPO-DiffProto+Entrop', 'PPO-DiffProto+Entrop', 'TRPO (MLP)']
    # alg_labels = ['Metric-10', 'PPO-DiffMetric10', 'PPO', 'TRPO (MLP)']

    res_folder = './Results/'
    exp_folder = os.path.join(res_folder, xp_name)
    baseline_folder = os.path.join(res_folder, 'baselines')
    diffproto_folder = os.path.join(res_folder, 'diffproto')
    # diffproto_folder = os.path.join(res_folder, 'diffproto_temp')
    diffproto_folder_entrop = os.path.join(res_folder, 'diffentrop')
    # diffproto_folder = os.path.join(res_folder, 'diffnoise')

    nb_runs = 25
    clus_sels = ['covr_exp']
    clus_dels = [True]
    alg_name = 'metricrl'
    # baselines_algs = ['ppo2', 'trpo_mpi', 'trpo_linear']
    baselines_algs = ['trpo_mpi']
    # diffproto_algs = ['PPO', 'TRPO']
    # diffproto_algs = ['PPO']
    diffproto_algs = ['TRPO', 'PPO']
    diffproto_algs_entrop = ['TRPO', 'PPO']

    diffproto_algs = [dpalg + '/nb_centers_' + str(n_clusterss[0]) for dpalg in diffproto_algs]
    diffproto_algs_entrop = [dpalg + '/init_cluster_noise_' + str(noise) for dpalg in diffproto_algs_entrop for noise in [0.1, 1.]]

    count = 0
    for temp, env in zip(temps, envs):
        all_j, all_r, all_e, _ = load_data_fixedtemp(exp_folder, env, alg_name, n_clusterss, nb_runs, nb_epochs, clus_sels, clus_dels, [temp], add_prop_id=False)

        if plot_diffproto:
            b_j, b_r = load_data_baselines(diffproto_folder, env, diffproto_algs, nb_runs, nb_epochs, add_prop_id=True)
            # for k_r in range(len(b_r)):
            #     new_r = [rr for rr in b_r[k_r] if len(rr) > nb_epochs]
            #     b_r[k_r] = new_r
            all_j += b_j
            all_r += b_r
            b_j, b_r = load_data_baselines(diffproto_folder_entrop, env, diffproto_algs, nb_runs, nb_epochs, add_prop_id=True)
            # for k_r in range(len(b_r)):
            #     new_r = [rr for rr in b_r[k_r] if len(rr) > nb_epochs]
            #     b_r[k_r] = new_r
            all_j += b_j
            all_r += b_r

        if plot_baselines:
            b_j, b_r = load_data_baselines(baseline_folder, env, baselines_algs, nb_runs, nb_epochs, add_prop_id=False)
            all_j += b_j
            all_r += b_r

        print('creating figure in folder {}, subfolder plots/{}, for env {}'.format(res_folder, xp_name, env))
        create_figure(res_folder, xp_name, 'R', env, all_r, alg_labels,
                      legend=(count + 1 == len(envs)))
        count += 1
