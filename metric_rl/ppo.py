import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.utils.dataset import parse_dataset
from mushroom.utils.minibatches import minibatch_generator

from .policies import PyTorchPolicy, GaussianPolicy
from .rl_shared import MLP, get_targets, get_adv


class PPO(Agent):
    def __init__(self, mdp_info, std_0, lr_v, lr_p, n_epochs_v, n_models_v, v_prediction_type,
                 n_epochs_policy, batch_size, eps_ppo, lam):

        self._n_epochs_policy = n_epochs_policy
        self._n_epochs_v = n_epochs_v
        self._batch_size = batch_size
        self._eps_ppo = eps_ppo

        h_layer_width = 64
        h_layer_length = 2
        input_shape = mdp_info.observation_space.shape
        size_list = [h_layer_width] * h_layer_length

        mean_network = MLP(mdp_info.observation_space.shape, mdp_info.action_space.shape, size_list)
        self._policy_torch = GaussianPolicy(mean_network, mdp_info.action_space.shape[0], std_0)

        self._p_optim = torch.optim.Adam(self._policy_torch.parameters(), lr=lr_p)

        self._lambda = lam

        approximator_params = dict(network=MLP,
                                   optimizer={'class': optim.Adam,
                                              'params': {'lr': lr_v}},
                                   loss=F.mse_loss,
                                   batch_size=h_layer_width,
                                   input_shape=input_shape,
                                   output_shape=(1,),
                                   size_list=size_list,
                                   activation_list=None,
                                   preproc=None,
                                   n_models=n_models_v,
                                   prediction=v_prediction_type,
                                   quiet=False)

        self._V = Regressor(PyTorchApproximator, **approximator_params)

        self._iter = 1

        policy = PyTorchPolicy(self._policy_torch)

        super().__init__(policy, mdp_info, None)

    def fit(self, dataset):
        tqdm.write('Iteration ' + str(self._iter))

        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
        np_adv = get_adv(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = torch.tensor(np_adv, dtype=torch.float)

        old_pol_dist = self._policy_torch.distribution(obs)
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()

        np_v_target, _ = get_targets(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        self._V.fit(x, np_v_target, n_epochs=self._n_epochs_v)

        for epoch in range(self._n_epochs_policy):
            for obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(self._batch_size, obs, act, adv, old_log_p):
                self._p_optim.zero_grad()
                prob_ratio = torch.exp(self._policy_torch.log_prob(obs_i, act_i) - old_log_p_i)
                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo, 1 + self._eps_ppo)
                loss = -torch.mean(torch.min(prob_ratio * adv_i, clipped_ratio * adv_i))
                loss.backward()
                self._p_optim.step()

        self._iter += 1

        # Print fit information
        logging_verr = []
        v_targets = torch.tensor(np_v_target, dtype=torch.float)
        for idx in range(len(self._V)):
            v_pred = torch.tensor(self._V(x, idx=idx), dtype=torch.float)
            v_err = F.mse_loss(v_pred, v_targets)
            logging_verr.append(v_err.item())

        logging_ent = self._policy_torch.entropy()
        new_pol_dist = self._policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
        avg_rwd = np.sum(r) / np.sum(last)
        tqdm.write("Iterations Results:\n\trewards {} vf_loss {}\n\tentropy {}  kl {}".format(
            avg_rwd, logging_verr, logging_ent, logging_kl))
        tqdm.write('--------------------------------------------------------------------------------------------------')