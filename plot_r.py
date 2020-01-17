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
xp_name = 'entropy'


def plot_data(x, median):
    if median:
        plt.plot(range(n_epochs), np.median(x, axis=0).squeeze())
        plt.fill_between(range(n_epochs), np.quantile(x, axis=0, q=.25), np.quantile(x, axis=0, q=.75), alpha=.5)
    else:
        mean, interval = get_mean_and_confidence(x)
        plt.plot(range(n_epochs), mean)
        plt.fill_between(range(n_epochs), mean - interval, mean + interval, alpha=.5)


def create_figure(name, env, all_data, alg_labels, median=False):
    plt.figure()
    legend_lab = []
    for n, d in zip(alg_labels, all_data):
        d = np.array(d)
        plot_data(d, median)
        legend_lab.append(n)

    plt.legend(legend_lab)
    plt.suptitle(env)

    fig_name = name + '_' + env + '.png'
    fig_path = os.path.join(res_folder, 'plots', xp_name)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(os.path.join(fig_path, fig_name))
    plt.clf()


def load_data_base(env, subfolder):
    all_r_a = []
    all_j_a = []
    all_entropy_a = []

    for run in range(nb_runs):
        j_filename = 'J-' + str(run) + '.npy'
        r_filename = 'R-' + str(run) + '.npy'
        e_filename = 'E-' + str(run) + '.npy'

        j_name = os.path.join(res_folder, env, subfolder, j_filename)
        r_name = os.path.join(res_folder, env, subfolder, r_filename)
        e_name = os.path.join(res_folder, env, subfolder, e_filename)

        try:
            J = np.load(j_name)
            R = np.load(r_name)
            E = np.load(e_name)
            if R.shape[0] != 1000:
                print(r_name, 'wrong shape')
            else:
                all_j_a.append(J)
                all_r_a.append(R)
                all_entropy_a.append(E)
        except:
            print(r_name, 'bad file')

    return all_j_a, all_r_a, all_entropy_a


def load_data_acost(alg_name):
    all_j = []
    all_r = []
    all_entropy = []
    alg_labels = []

    for n_clusters in n_clusterss:
        for a_cost_scale in a_cost_scales:
            alg_label = 'c' + str(n_clusters) + 'a' + str(a_cost_scale)
            subfolder = alg_name + '_' + alg_label

            all_j_a, all_r_a, all_entropy_a = load_data_base(env, subfolder)

            all_j.append(all_j_a)
            all_r.append(all_r_a)
            all_entropy.append(all_entropy_a)
            alg_labels.append(alg_label)

    return all_j, all_r, all_entropy, alg_labels


def load_data_entropy(alg_name):
    all_j = []
    all_r = []
    all_entropy = []
    alg_labels = []

    for n_clusters in n_clusterss:
        alg_label = 'c' + str(n_clusters)
        subfolder = alg_name + '_' + alg_label

        all_j_a, all_r_a, all_entropy_a = load_data_base(env, subfolder)

        all_j.append(all_j_a)
        all_r.append(all_r_a)
        all_entropy.append(all_entropy_a)
        alg_labels.append(alg_label)

    return all_j, all_r, all_entropy, alg_labels


if __name__ == '__main__':
    # Creating parameters tables
    for env in envs:
        all_j, all_r, all_entropy, alg_labels = load_data_entropy(alg_name)

        create_figure('R', env, all_r, alg_labels)
        create_figure('E', env, all_entropy, alg_labels)
