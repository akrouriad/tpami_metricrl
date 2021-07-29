import argparse
from pathlib import Path
import numpy as np
from metric_rl.utils import plot_data
import matplotlib.pyplot as plt
from contextlib import contextmanager


def create_figures(results_dir, env, data_dict, subfolder=None, use_median=False, display=False, legend_plot='R'):
    for name in ['J', 'R', 'E']:
        plot_legend = name == legend_plot
        create_figure(results_dir, name, env, data_dict, subfolder=subfolder, use_median=use_median,
                      legend=plot_legend, display=display)


def create_figure(results_dir, name, env, data_dict, subfolder=None, use_median=False, legend=False, display=False):
    fig, ax = plt.subplots()
    legend_lab = []
    for n, d in data_dict.items():
        plot_data(ax, d[name], use_median)
        legend_lab.append(n)

    plt.grid(linestyle='dotted')

    if legend:
        plt.legend(legend_lab, fontsize='small', frameon=False)

    plt.xlabel('Epoch')
    plt.ylabel(name)

    fig_name = name + '_' + env
    fig_path = results_dir / 'plots'
    if subfolder:
        fig_path = fig_path / subfolder

    fig_path.mkdir(exist_ok=True)
    full_path = fig_path / (fig_name + '.png')
    plt.savefig(full_path,  bbox_inches='tight')
    if display:
        plt.suptitle(env)
        plt.show()

    plt.close(fig)


def load_data_metricrl(results_dir, env, n_seeds):
    env_subfolder = 'env_id_' + env
    env_dir = results_dir / env_subfolder

    results_dict = dict()

    for clusters_dir in sorted(env_dir.iterdir()):

        n_clusters = clusters_dir.name.split('_')[-1]
        data_dir = clusters_dir / 'MetricRL'

        J, R, E = load_data(data_dir, n_seeds)

        results_dict[f'MetricRL-{n_clusters}'] = dict(J=J, R=R, E=E)

    return results_dict

@contextmanager
def ignore_missing_file():
  try:
    yield
  except FileNotFoundError as e:
      print(e)
      return dict()


def load_data_metricrl_diff(results_dir, env, n_seeds):
    env_subfolder = 'env_id_' + env
    env_dir = results_dir / env_subfolder

    results_dict = dict()

    for alg_dir in sorted(env_dir.iterdir()):
        alg_name = alg_dir.name.split('_')[-1]

        for nb_centers_dir in sorted(alg_dir.iterdir()):
            nb_centers = nb_centers_dir.name.split('_')[-1]

            for init_cluster_noise_dir in sorted(nb_centers_dir.iterdir()):
                init_cluster_noise = init_cluster_noise_dir.name.split('_')[-1]
                data_dir = init_cluster_noise_dir / f'MetricRLDiff'

                J, R, E = load_data(data_dir, n_seeds)

                results_dict[f'MetricRLDiff-{alg_name}-{nb_centers}-{init_cluster_noise}'] = dict(J=J, R=R, E=E)

    return results_dict


def load_data(data_dir, n_seeds):
    J_list = list()
    R_list = list()
    E_list = list()

    for seed in range(n_seeds):
        try:
            J = np.load(data_dir / f'J-{seed}.npy')
            R = np.load(data_dir / f'R-{seed}.npy')
            E = np.load(data_dir / f'E-{seed}.npy')

            J_list.append(J)
            R_list.append(R)
            E_list.append(E)
        except FileNotFoundError as e:
            print(e)
            pass

    return np.array(J_list), np.array(R_list), np.array(E_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=str, nargs='+')
    parser.add_argument('--results-dir', type=str, default='logs')
    parser.add_argument('--n-seeds', type=int, default=25)
    parser.add_argument('--display', action='store_true')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    bullet_envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']

    envs = args.envs

    if 'bullet' in envs:
        envs.remove('bullet')
        envs += bullet_envs

    for env in envs:
        data_dict = dict()
        with ignore_missing_file():
            data_dict.update(**load_data_metricrl(results_dir / 'metricrl', env, args.n_seeds))
        with ignore_missing_file():
            data_dict.update(**load_data_metricrl_diff(results_dir / 'metricrl_diff', env, args.n_seeds))

        create_figures(results_dir, env, data_dict, subfolder=None, display=args.display)
