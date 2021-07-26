import numpy as np
from tqdm import tqdm

import torch

from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator

from metric_rl.projections.cluster_weights import cweight_mean_proj
from metric_rl.projections.gaussian import lin_gauss_kl_proj, utils_from_chol, mean_diff
from metric_rl.rl_shared import get_targets
from metric_rl.policies import MetricPolicy
from metric_rl.cluster_randomized_optimization import randomized_swap_optimization


class MetricRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs_per_fit, batch_size,
                 entropy_profile, max_kl, lam, n_samples=2000, critic_fit_params=None, do_delete=False):
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        policy = MetricPolicy(mdp_info.observation_space.shape,
                              mdp_info.action_space.shape,
                              **policy_params)

        self._use_cuda = policy_params['use_cuda'] if 'use_cuda' in policy_params else False

        self._actor_optimizers = [actor_optimizer['class']([policy._regressor._c_weights], **actor_optimizer['cw_params']),
                                  actor_optimizer['class']([policy._regressor.means], **actor_optimizer['means_params']),
                                  actor_optimizer['class']([policy._regressor._log_sigma], **actor_optimizer['log_sigma_params'])]

        self._temp_lr = self._temp_base_lr = .01
        self._temp_optimizer = actor_optimizer['class']([policy._regressor._log_temp], lr=self._temp_lr)

        self._n_epochs_per_fit = n_epochs_per_fit
        self._batch_size = batch_size

        self._critic = Regressor(TorchApproximator, **critic_params)
        self._do_delete = do_delete
        self._e_profile = entropy_profile['class'](policy, **entropy_profile['params'])
        self._max_kl = max_kl
        self._kl_temp = max_kl * .25
        self._kl_del = max_kl * .5
        self._kl_cw_after_del = max_kl * .66
        self._lambda = lam

        self._n_swaps = float(policy.n_clusters)
        self._n_samples = n_samples

        self._iter = 0
        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        # Get dataset
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        assert((self.mdp_info.action_space.high == -self.mdp_info.action_space.low).all())
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)

        # Get tensors
        obs = to_float_tensor(x, self._use_cuda)
        act = to_float_tensor(u,  self._use_cuda)
        v, adv = get_targets(self._critic, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda,
                             prediction='min')
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
        adv_t = to_float_tensor(adv, self._use_cuda)

        # Critic Update
        self._critic.fit(x, v, **self._critic_fit_params)

        # Save old data
        old_chol = self.policy.get_chol_t().detach()

        # Add cluster actions for zero weighted clusters at first iter (should become obsolete later)
        if self._iter == 0:
            self._add_cluster_centers(obs, act, adv_t)

        old = dict(w=self.policy.get_unormalized_membership_t(obs).detach(),
                   cweights=self.policy.get_cweights_t().detach(),
                   cmeans=self.policy.get_cmeans_t().clone().detach(),
                   means=self.policy.get_mean_t(obs).detach(),
                   pol_dist=self.policy.distribution_t(obs),
                   log_p=self.policy.log_prob_t(obs, act).clone().detach(),
                   membership=self.policy.get_membership_t(obs).detach(),
                   **utils_from_chol(old_chol))

        entropy_lb = self._e_profile.get_e_lb()

        full_batch_proj = True

        # Optimize cw, mean and cov
        if self._iter % 2 == 0:
            self._update_all_parameters(obs, act, adv_t, old, entropy_lb)
        else:
            # Swap clusters
            swapped = self._random_swap_clusters(obs, old, act, adv_t)

            # Compress policy
            deleted = []
            if self._do_delete and mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) < self._kl_del:
                deleted = self._cleanup(self._kl_del, obs, old, adv_t, act)
                if deleted:
                    self._increase_cw(deleted, self._kl_cw_after_del, obs, old)

            if swapped or deleted:
                # Optimize mean and covariance
                print('doing partial update')
                old['intermediate_cmeans'] = self.policy.get_cmeans_t().clone().detach()
                self._update_mean_n_cov(obs, act, adv_t, old, entropy_lb)
                full_batch_proj = False
            else:
                # Optimize cw, mean and cov
                print('doing full update')
                self._update_all_parameters(obs, act, adv_t, old, entropy_lb)

        # Actor Update
        if full_batch_proj:
            self._full_batch_projection_all_par(obs, old, entropy_lb)
        else:
            self._full_batch_projection_partial(obs, old, entropy_lb)

        # next iter
        self._iter += 1

        # logging
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(self.policy.distribution_t(obs), old['pol_dist']))
        mean_covr = torch.mean(torch.sum(self.policy.get_membership_t(obs), dim=1))
        tqdm.write('KL {} Covr {}'.format(logging_kl, mean_covr))

    def _add_cluster_centers(self, obs, act, adv_t):
        # adding clusters
        sadv, oadv = torch.sort(adv_t, dim=0)
        ba_ind = 0
        for k, cw in enumerate(self.policy._regressor._c_weights):
            if cw == 0.:
                ba = oadv[ba_ind][0]
                self.policy._regressor.centers[k] = obs[ba]
                self.policy._regressor.means.data[k] = act[ba]
                self.policy.set_cmeans_t(self.policy._regressor.means)

                ba_ind += 1
        tqdm.write('added {} clusters'.format(ba_ind))

    def _update_mean_n_cov(self, obs, act, adv_t, old, entropy_lb):
        # Compute mean projection (nu)
        intermediate_means = self.policy.get_mean_t(obs).detach()
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, adv_i, intermediate_means_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], adv_t, intermediate_means):

                if mean_diff(intermediate_means_i, old_means_i, old['prec']) < self._max_kl - 1e-6:
                    for opt in self._actor_optimizers[1:]:
                        opt.zero_grad()

                    # Get Basics stuff
                    means_i = self.policy.get_mean_t(obs_i)
                    chol = self.policy.get_chol_t()

                    proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                               old['cov'], old['prec'], old['logdetcov'],
                                               self._max_kl, entropy_lb)
                    proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                    # Compute loss
                    prob_ratio = torch.exp(proj_distrib.log_prob(act_i)[:, None] - old_log_p_i)
                    loss = -torch.mean(prob_ratio * adv_i)
                    loss.backward()

                    for opt in self._actor_optimizers[1:]:
                        opt.step()

    def _cleanup(self, kl_ub, obs, old_pol_data, adv, act):
        cws = torch.sort(self.policy.get_cweights_t().clone())
        deleted = []
        for cw, k in zip(cws[0], cws[1]):
            self.policy._regressor._c_weights.data[k] = 0.
            if mean_diff(self.policy.get_mean_t(obs), old_pol_data['means'], old_pol_data['prec']) < kl_ub:
                deleted.append(k)
            else:
                self.policy._regressor._c_weights.data[k] = cw
        print('deleted {} clusters'.format(len(deleted)))
        if deleted:
            high_adv_idxs = torch.topk(adv, k=len(deleted), dim=0)[1]
            for k, idx in zip(deleted, high_adv_idxs):
                self.policy._regressor.centers[k] = obs[idx]
                self.policy._regressor.means.data[k] = act[idx]
                self.policy.set_cmeans_t(self.policy._regressor.means)

        return deleted

    def _increase_cw(self, deleted, kl_ub, obs, old_pol_data):
        inc_lb = 0.
        inc_ub = 10.
        for k in range(2000):
            inc = .5 * (inc_ub + inc_lb)
            for cwi in deleted:
                self.policy._regressor._c_weights.data[cwi] = inc
            if mean_diff(self.policy.get_mean_t(obs), old_pol_data['means'], old_pol_data['prec']) < kl_ub:
                inc_lb = inc
            else:
                inc_ub = inc
        for cwi in deleted:
            self.policy._regressor._c_weights.data[cwi] = inc_lb
        print('increased cluster weights to {}'.format(inc_lb))

    def _swap_clusters_covr(self, obs, act, old):
        base_perf = torch.mean(old['membership'])
        print('init covr', base_perf)

        high_heur_idxs = torch.argsort(torch.sum(old['membership'], dim=1), dim=0, descending=True)
        remaining_idx = [k for k in range(self.policy._regressor._n_clusters)]
        state_centers = self.policy._regressor.centers.clone().numpy()
        for idxidx, idx in enumerate(high_heur_idxs):
            # finding closest state
            dist_to_centers = -np.sum((state_centers - obs[idx].detach().numpy()) ** 2, axis=1)
            closest_idxs = np.argsort(dist_to_centers)
            for closest_idx in closest_idxs:
                closest = remaining_idx[closest_idx]
                state_backup = self.policy._regressor.centers[closest].clone()
                # swap and check KL
                self.policy._regressor.centers[closest] = obs[idx]
                kl_swap = mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec'])
                if kl_swap < self._max_kl:
                    # check increase of the objective after state swapping
                    # ... and cluster center swapping?
                    new_perf = torch.mean(self.policy.get_membership_t(obs))
                    if new_perf >= base_perf:
                        # keep if kl under thresh and objective did not decrease
                        base_perf = new_perf
                        remaining_idx.pop(closest_idx)
                        state_centers = np.delete(state_centers, closest_idx, axis=0)
                        break
                    else:
                        # revert old cluster center
                        self.policy._regressor.centers[closest] = state_backup
                else:
                    # revert old cluster center
                    self.policy._regressor.centers[closest] = state_backup
            if len(remaining_idx) == 0:
                break
        new_pol_dist = self.policy.distribution_t(obs)
        logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old['pol_dist']))
        print('Nb swapped clusters: {}. KL: {}'.format(self.policy._regressor._n_clusters - len(remaining_idx), logging_kl))

    def _random_swap_clusters(self, obs, old, act, adv):
        candidates = obs.detach().numpy()
        c_0 = self.policy.get_cluster_centers()

        w = self.policy.get_membership_t(obs)

        cluster_h = -torch.sum(w, dim=0)
        cluster_h = cluster_h.detach().numpy().squeeze()

        sample_h = torch.mean(self.policy._regressor.to_clust_dist(obs), dim=1)
        sample_h = sample_h.detach().numpy().squeeze()

        def bound_function(c_i):
            c_old = self.policy.get_cluster_centers()
            self.policy.set_cluster_centers(c_i)
            kl_swap = mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec'])
            self.policy.set_cluster_centers(c_old)
            return kl_swap < self._max_kl

        def evaluation_function(c_i):
            c_old = self.policy.get_cluster_centers()
            self.policy.set_cluster_centers(c_i)
            w = self.policy._regressor._cluster_distance(obs)
            self.policy.set_cluster_centers(c_old)
            return torch.mean(torch.max(w, dim=1)[0]).item() - torch.mean(torch.topk(w, k=3, dim=1)[0]).item()

        swapped = False
        while True:
            c_best = randomized_swap_optimization(c_0, candidates, cluster_h, sample_h,
                                                  bound_function, evaluation_function,
                                                  int(np.ceil(self._n_swaps)), self._n_samples)

            if np.array_equal(c_best, c_0):
                self._n_swaps /= 2
                if self._n_swaps < 1.0:
                    self._n_swaps = 1.0
                    break
            else:
                swapped = True
                break

        if swapped:
            self.policy.set_cluster_centers(c_best)

            new_pol_dist = self.policy.distribution_t(obs)
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old['pol_dist']))
            print('Nb swapped clusters: {}. KL: {}'.format(int(np.ceil(self._n_swaps)), logging_kl))

            self._n_swaps *= 1.8
            self._n_swaps = np.minimum(self._n_swaps, self.policy.n_clusters)
        else:
            print('Nb swapped clusters: 0. KL: 0.0')
        return swapped

    def _update_all_parameters(self, obs, act, adv_t, old, entropy_lb):
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, wq_i, old_means_i, old_log_p_i, adv_i in \
                    minibatch_generator(self._batch_size, obs, act, old['w'], old['means'], old['log_p'], adv_t):
                for opt in self._actor_optimizers:
                    opt.zero_grad()

                # Get Basics stuff
                w = self.policy.get_unormalized_membership_t(obs_i)
                means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'])
                chol = self.policy.get_chol_t()

                # Compute cluster weights projection (eta)
                eta = cweight_mean_proj(self.policy.w_default, w, means_i, wq_i, old_means_i, old['prec'], self._max_kl)
                cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * old['cweights']

                # Compute mean projection (nu)
                intermediate_means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'], cweights_eta)
                means_i = self.policy.get_mean_t(obs_i)

                proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                           old['cov'], old['prec'], old['logdetcov'],
                                           self._max_kl, entropy_lb)
                proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                # Compute loss
                prob_ratio = torch.exp(proj_distrib.log_prob(act_i)[:, None] - old_log_p_i)
                loss = -torch.mean(prob_ratio * adv_i)
                loss.backward()

                for opt in self._actor_optimizers:
                    opt.step()

    def _full_batch_projection_all_par(self, obs, old, entropy_lb):
        # Get Basics stuff
        w = self.policy.get_unormalized_membership_t(obs)
        means = self.policy.get_intermediate_mean_t(obs, old['cmeans'])
        chol = self.policy.get_chol_t()

        # Compute cluster weights projection (eta)
        eta = cweight_mean_proj(self.policy.w_default, w, means, old['w'], old['means'], old['prec'], self._max_kl)
        cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * torch.abs(old['cweights'])
        self.policy.set_cweights_t(cweights_eta)

        # Compute mean projection (nu)
        intermediate_means = self.policy.get_intermediate_mean_t(obs, old['cmeans'])
        means = self.policy.get_mean_t(obs)

        proj_d = lin_gauss_kl_proj(means, chol, intermediate_means, old['means'],
                                   old['cov'], old['prec'], old['logdetcov'], self._max_kl, entropy_lb)
        nu = proj_d['eta_mean']
        cmeans_nu = nu * self.policy.get_cmeans_t() + (1 - nu) * old['cmeans']

        self.policy.set_cmeans_t(cmeans_nu)
        self.policy.set_chol_t(proj_d['chol'])

    def _full_batch_projection_partial(self, obs, old, entropy_lb):
        means = self.policy.get_mean_t(obs)
        chol = self.policy.get_chol_t()

        intermediate_means = self.policy.get_intermediate_mean_t(obs, old['intermediate_cmeans'])
        proj_d = lin_gauss_kl_proj(means, chol, intermediate_means, old['means'],
                                   old['cov'], old['prec'], old['logdetcov'], self._max_kl, entropy_lb)
        nu = proj_d['eta_mean']
        cmeans_nu = nu * self.policy.get_cmeans_t() + (1 - nu) * old['intermediate_cmeans']

        self.policy.set_cmeans_t(cmeans_nu)
        self.policy.set_chol_t(proj_d['chol'])

