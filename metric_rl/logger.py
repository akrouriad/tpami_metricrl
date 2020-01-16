import os
import pathlib
import datetime
import torch
import pickle
import numpy as np


def generate_log_folder(name, algorithm_name='', postfix='', timestamp=False, base_folder='./log'):

    if timestamp:
        if algorithm_name:
            algorithm_name += '_'
        algorithm_name += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if algorithm_name and postfix:
        algorithm_name += '_' + postfix

    folder_name = os.path.join(base_folder, name, algorithm_name)

    network_folder_name = os.path.join(folder_name, 'net')

    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    pathlib.Path(network_folder_name).mkdir(parents=True, exist_ok=True)

    return folder_name


def save_parameters(log_name, params):
    file_name = os.path.join(log_name, 'parameters.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)


class Logger:
    def __init__(self, directory, subdir=''):
        if directory is None:
            print('!!! No log dir specified, no data will be saved')

        self._directory = directory
        self._sub_directory = os.path.join(directory, subdir) if directory is not None else None
        self._iter_n = 0

    def save(self, network=None, seed=None, **kwargs):
        if self._directory is not None:
            if network is not None:
                self.save_network(network, self._iter_n, seed)

            for name, array in kwargs.items():
                self.save_numpy(name, array, seed)
            self._iter_n += 1

    def save_network(self, network, n_it, seed=None):
        filename = 'network-'
        if seed is not None:
            filename += str(seed) + '-'
        filename += str(n_it) + '.pth'
        file_path = os.path.join(self._sub_directory, filename)
        torch.save(network, file_path)

    def save_numpy(self, name, array, seed=None):

        if seed is not None:
            name += '-' + str(seed)
        file_path = os.path.join(self._directory, name)
        np.save(file_path, array)
