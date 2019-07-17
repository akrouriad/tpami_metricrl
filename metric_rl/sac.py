from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from mushroom.algorithms.agent import Agent
from mushroom.policy import Policy
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.utils.replay_memory import ReplayMemory


class ActorLossSAC(nn.Module):
    """
    Class used to implement the loss function of the actor.

    """

    def __init__(self, critic_0, critic_1, alpha):
        super().__init__()

        self._critic_0 = critic_0
        self._critic_1 = critic_1
        self._alpha = alpha

    def forward(self, action, log_prob, state):
        q_0 = self._critic_0(state, action)
        q_1 = self._critic_1(state, action)

        q = torch.min(q_0, q_1)

        return (self._alpha() * log_prob - q).mean()


class SACGaussianPolicy(Policy):
    def __init__(self, approximator, use_cuda=False):
        """
        Constructor.

        Args:
            approximator (Regressor): a regressor computing mean and variance given a state
        """
        self._approximator = approximator
        self._use_cuda = use_cuda

    def __call__(self, state, action, use_log=False):
        raise NotImplementedError

    def draw_action(self, state):
        mu, sigma = self._approximator.predict(state)

        a = mu + sigma*np.random.randn(len(sigma))
        return a

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    def reset(self):
        pass


class SAC(Agent):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019
    """
    def __init__(self, mdp_info,
                 batch_size, initial_replay_size, max_replay_size,
                 warmup_transitions, tau, lr_alpha,
                 actor_params, critic_params,
                 actor_fit_params=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions (int): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau (float): value of coefficient for soft updates;
            lr_alpha (float): Learning rate for the entropy coefficient;
            actor_params (dict): parameters of the actor approximator to
                build;
            critic_params (dict): parameters of the critic approximator to
                build;
            actor_fit_params (dict, None): parameters of the fitting algorithm
                of the actor approximator;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator;

        """

        self._actor_fit_params = dict() if actor_fit_params is None else actor_fit_params
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._warmup_transitions = warmup_transitions
        self._tau = tau
        self._target_entropy = - mdp_info.action_space.shape[0]

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        if 'prediction' in critic_params.keys():
            assert critic_params['prediction'] == 'min'
        else:
            critic_params['prediction'] = 'min'

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(PyTorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(PyTorchApproximator,
                                                     **target_critic_params)

        self._log_alpha = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        self._alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)

        if 'loss' not in actor_params:
            actor_params['loss'] = ActorLossSAC(self._critic_approximator.model[0].network,
                                                self._critic_approximator.model[0].network,
                                                self._alpha)
        else:
            actor_params['loss'] = actor_params['loss'](self._critic_approximator.model[0].network,
                                                        self._critic_approximator.model[0].network,
                                                        self._alpha)

        self._actor_approximator = Regressor(PyTorchApproximator,
                                             **actor_params)
        policy = SACGaussianPolicy(self._actor_approximator)

        self._init_target()

        super().__init__(policy, mdp_info)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            if self._replay_memory.size() > self._warmup_transitions:
                eps = np.random.randn(action.shape)
                self._actor_approximator.fit(state, eps, state)
                log_prob = self._actor_approximator.predict(state, eps)
                self._update_alpha(log_prob)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target()

    def _init_target(self):
        """
        Init weights for target approximators

        """
        for i in range(len(self._critic_approximator)):
            self._target_critic_approximator.model[i].set_weights(
                self._critic_approximator.model[i].get_weights())

    def _update_target(self):
        """
        Update the target networks.

        """
        for i in range(len(self._target_critic_approximator)):
            critic_weights_i = self._tau * self._critic_approximator.model[i].get_weights()
            critic_weights_i += (1 - self._tau) * self._target_critic_approximator.model[i].get_weights()
            self._target_critic_approximator.model[i].set_weights(critic_weights_i)

    def _alpha(self):
        return self.log_alpha.exp()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a = self.policy(next_state)
        log_prob_next = self.policy(next_state, a)

        q = self._target_critic_approximator.predict(next_state, a) - self._alpha() * log_prob_next
        q *= 1 - absorbing

        return q