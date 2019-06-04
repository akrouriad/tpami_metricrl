import torch
import numpy as np


class FixedIterSaver:
    def __init__(self, network, file_base_name, verbose=True, iter_to_save=[]):
        self.network = network
        self.file_base_name = file_base_name
        self.verbose = verbose
        self.nb_iter = 0
        self.iter_to_save = np.asarray(iter_to_save)

    def next_iter(self):
        self.nb_iter += 1
        if np.any((self.iter_to_save - self.nb_iter) == 0):
            if self.verbose:
                print('--> saving iteration {}'.format(self.nb_iter))
            torch.save(self.network.state_dict(), self.file_base_name)


class PolicyIterationLogger:
    def __init__(self, file_base_name):
        self.rwd_fname = file_base_name + '.rwd'
        self.update_fname = file_base_name + '.up'
        self.iter = 0
        self.nb_trans = 0
        self.partial_return_prev_iter = 0
        with open(self.rwd_fname, "w") as file:
            print("# .rwd contains number of transitions:cumulative reward", file=file)
        with open(self.update_fname, "w") as file:
            print("# .up contains iter and entropy", file=file)

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
