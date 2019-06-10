import os
import pathlib
import datetime
import torch
import numpy as np


def generate_log_folder(name, algorithm_name='', postfix='', timestamp=False, base_folder='./log'):

    if timestamp:
        if algorithm_name:
            algorithm_name += '_'
        algorithm_name += datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S')

    if algorithm_name and postfix:
        algorithm_name += '_' + postfix

    folder_name = os.path.join(base_folder, name, algorithm_name)

    pathlib.Path(folder_name).mkdir(parents=True)

    return folder_name


class FixedIterSaver:
    def __init__(self, network, directory, verbose=True, iter_to_save=[]):
        self.network = network
        self.directory = directory
        self.verbose = verbose
        self.nb_iter = 0
        self.iter_to_save = np.asarray(iter_to_save)

    def next_iter(self):
        self.nb_iter += 1
        if len(self.iter_to_save) == 0 or np.any((self.iter_to_save - self.nb_iter) == 0):
            if self.verbose:
                print('--> saving iteration {}'.format(self.nb_iter))
            filename = 'network-' + str(self.nb_iter) + '.pth'
            file_path = os.path.join(self.directory, filename)
            torch.save(self.network, file_path)


class PolicyIterationLogger:
    def __init__(self, file_base_name):
        self.rwd_fname = os.path.join(file_base_name, 'reward.txt')
        self.update_fname = os.path.join(file_base_name, 'update.txt')
        self.iter = 0
        self.nb_trans = 0
        self.partial_return_prev_iter = 0
        with open(self.rwd_fname, "w") as file:
            print("# reward.txt contains number of transitions:cumulative reward", file=file)
        with open(self.update_fname, "w") as file:
            print("# update.txt contains iter and entropy", file=file)

    def next_iter_path(self, rwd, done, entropy):
        for k, r in enumerate(rwd):
            self.partial_return_prev_iter += r
            self.nb_trans += 1
            if done[k]:
                with open(self.rwd_fname, "a") as file:
                    print("{} {}".format(self.nb_trans, self.partial_return_prev_iter), file=file)
                self.partial_return_prev_iter = 0
        with open(self.update_fname, "a") as file:
            print("{} {}".format(self.iter, entropy), file=file)
        self.iter += 1

    def next_iter_ret(self, rets, lens, entropy):
        for k, ret in enumerate(rets):
            self.nb_trans += lens[k]
            with open(self.rwd_fname, "a") as file:
                print("{} {}".format(self.nb_trans, ret), file=file)
        with open(self.update_fname, "a") as file:
            print("{} {}".format(self.iter, entropy), file=file)
        self.iter += 1
