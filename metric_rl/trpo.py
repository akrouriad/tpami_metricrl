import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import PyTorchApproximator
from mushroom.utils.dataset import parse_dataset, compute_J

from .policies import PyTorchPolicy, GaussianPolicy
from .rl_shared import MLP, get_targets


def gather_flat_grad(params):
    views = []
    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        else:
            view = p.grad.view(-1)
        views.append(view)
    return torch.cat(views, 0)


class TRPO(Agent):
    def __init__(self, mdp_info, std_0, lr_v=3e-4, n_epochs_v=3, n_models_v=1, v_prediction_type='min',
                 n_epochs_line_search=10, n_epochs_cg=10, cg_damping=1e-2, cg_residual_tol=1e-10, batch_size=64,
                 ent_coeff=0.0, max_kl=0.001, lam=1.0):
        self._n_epochs_line_search = n_epochs_line_search
        self._n_epochs_v = n_epochs_v
        self._n_epochs_cg = n_epochs_cg
        self._cg_damping = cg_damping
        self._cg_residual_tol = cg_residual_tol

        self._max_kl = max_kl
        self._ent_coeff = ent_coeff

        self._lambda = lam

        h_layer_width = 64
        h_layer_length = 2
        input_shape = mdp_info.observation_space.shape
        size_list = [h_layer_width] * h_layer_length

        mean_network = MLP(mdp_info.observation_space.shape, mdp_info.action_space.shape, size_list)
        self._policy_torch = GaussianPolicy(mean_network, mdp_info.action_space.shape[0], std_0)
        self._old_policy_torch = GaussianPolicy(mean_network, mdp_info.action_space.shape[0], std_0)

        approximator_params = dict(network=MLP,
                                   optimizer={'class': optim.Adam,
                                              'params': {'lr': lr_v}},
                                   loss=F.mse_loss,
                                   batch_size=batch_size,
                                   input_shape=input_shape,
                                   output_shape=(1,),
                                   size_list=size_list,
                                   activation_list=None,
                                   preproc=None,
                                   n_models=n_models_v,
                                   prediction=v_prediction_type,
                                   quiet=True)

        self._V = Regressor(PyTorchApproximator, **approximator_params)

        self._iter = 1

        policy = PyTorchPolicy(self._policy_torch)

        super().__init__(policy, mdp_info, None)

    def _zero_grad(self):
        for p in self._policy_torch.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _conjugate_gradient(self, b, obs, old_pol_dist):
        p = b.detach().numpy()
        r = b.detach().numpy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)

        for i in range(self._n_epochs_cg):
            z = self._fisher_vector_product(torch.from_numpy(p), obs, old_pol_dist).detach().numpy()
            v = rdotr / p.dot(z)
            x += v*p
            r -= v*z
            newrdotr = r.dot(r)
            mu = newrdotr/rdotr
            p = r + mu*p

            rdotr = newrdotr
            if rdotr < self._cg_residual_tol:
                break
        return x

    def _fisher_vector_product(self, p, obs, old_pol_dist):
        self._zero_grad()
        kl = self._compute_kl(obs, old_pol_dist)
        grads = torch.autograd.grad(kl, self._policy_torch.parameters(), create_graph=True, retain_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * torch.autograd.Variable(p)).sum()
        grads = torch.autograd.grad(kl_v, self._policy_torch.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + p * self._cg_damping

    def _compute_kl(self, obs, old_pol_dist):
        new_pol_dist = self._policy_torch.distribution(obs)
        return torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))

    def _compute_loss(self, obs, act, adv):
        ratio = torch.exp(self._policy_torch.log_prob(obs, act) - self._old_policy_torch.log_prob(obs, act))
        J = torch.mean(ratio * adv)

        return J + self._ent_coeff * self._policy_torch.entropy()  # FIXME use distributions

    def fit(self, dataset):
        tqdm.write('Iteration ' + str(self._iter))

        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        obs = torch.tensor(x, dtype=torch.float)
        act = torch.tensor(u, dtype=torch.float)
        v_target, np_adv = get_targets(self._V, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda)
        np_adv = (np_adv - np.mean(np_adv)) / (np.std(np_adv) + 1e-8)
        adv = torch.tensor(np_adv, dtype=torch.float)

        ## Policy update

        # Set old policy weights
        self._old_policy_torch.load_state_dict(self._policy_torch.state_dict())

        # Compute loss
        self._zero_grad()
        loss = self._compute_loss(obs, act, adv)

        prev_loss = loss.item()

        # Compute Gradient
        loss.backward(retain_graph=True)
        g = gather_flat_grad(self._policy_torch.parameters())

        old_pol_dist = self._old_policy_torch.distribution(obs)

        # Compute direction trought conjugate gradient
        stepdir = self._conjugate_gradient(g, obs, old_pol_dist)

        # Line search
        shs = .5 * stepdir.dot(self._fisher_vector_product(torch.from_numpy(stepdir), obs, old_pol_dist))
        lm = np.sqrt(shs / self._max_kl)
        fullstep = stepdir / lm
        stepsize = 1.0

        theta_old = self.policy.get_weights()

        violation = True

        for _ in range(self._n_epochs_line_search):
            theta_new = theta_old + fullstep * stepsize
            self.policy.set_weights(theta_new)

            new_loss = self._compute_loss(obs, act, adv)
            kl = self._compute_kl(obs, old_pol_dist)
            improve = new_loss - prev_loss
            if kl <= self._max_kl * 1.5 or improve >= 0:
                violation = False
                break
            stepsize *= .5

        if violation:
            self.policy.set_weights(theta_old)

        # VF update
        self._V.fit(x, v_target, n_epochs=self._n_epochs_v)

        self._iter += 1

        # Print fit information
        logging_verr = []
        torch_v_targets = torch.tensor(v_target, dtype=torch.float)
        for idx in range(len(self._V)):
            v_pred = torch.tensor(self._V(x, idx=idx), dtype=torch.float)
            v_err = F.mse_loss(v_pred, torch_v_targets)
            logging_verr.append(v_err.item())

        logging_ent = self._policy_torch.entropy()
        new_pol_dist = self._policy_torch.distribution(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old_pol_dist))
        avg_rwd = np.mean(compute_J(dataset))
        tqdm.write("Iterations Results:\n\trewards {} vf_loss {}\n\tentropy {}  kl {}".format(
            avg_rwd, logging_verr, logging_ent, logging_kl))
        tqdm.write('--------------------------------------------------------------------------------------------------')


