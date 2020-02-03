import numpy as np
from tqdm import tqdm

import torch

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.dataset import parse_dataset, compute_J
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator

from .cluster_weight_proj import cweight_mean_proj
from .gaussian_proj import lin_gauss_kl_proj, utils_from_chol, mean_diff
from .rl_shared import get_targets
from .policies import MetricPolicy
from .cluster_randomized_optimization import randomized_swap_optimization
from scipy.spatial.distance import pdist


class ProjectionDelSwapMetricRL(Agent):
    def __init__(self, mdp_info, policy_params, critic_params,
                 actor_optimizer, n_epochs_per_fit, batch_size,
                 entropy_profile, max_kl, lam, n_samples=2000, critic_fit_params=None, a_cost_scale=0., clus_sel='covr',
                 do_delete=False, opt_temp=True, squash='none'):
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        policy = MetricPolicy(mdp_info.observation_space.shape,
                              mdp_info.action_space.shape, squash=squash,
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

        self._do_opt_temp = opt_temp
        self._a_cost_scale = a_cost_scale
        self._clus_sel = clus_sel
        self._n_swaps = float(policy.n_clusters)
        self._n_samples = n_samples
        self._squash = squash

        self._iter = 0
        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        # Get dataset
        x, u, r, xn, absorbing, last = parse_dataset(dataset)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        assert((self.mdp_info.action_space.high == -self.mdp_info.action_space.low).all())
        r = r.astype(np.float32) - self._a_cost_scale * np.squeeze(np.sum(np.logical_or(u > self.mdp_info.action_space.high, u < -self.mdp_info.action_space.high) * (u**2 - self.mdp_info.action_space.high), axis=1))
        xn = xn.astype(np.float32)

        # Get tensors
        obs = to_float_tensor(x, self._use_cuda)
        act = to_float_tensor(u,  self._use_cuda)
        act_raw = to_float_tensor(np.arctanh(u), self._use_cuda)
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
                   membership=self.policy.get_membership_t(obs).detach(),
                   **utils_from_chol(old_chol))
        if self._squash == 'tanh':
            old['log_p'] = self.policy.log_prob_t(obs, act, act_raw).clone().detach()
        else:
            old['log_p'] = self.policy.log_prob_t(obs, act).clone().detach()

        entropy_lb = self._e_profile.get_e_lb()

        full_batch_proj = True

        # optimize cw, mean and cov
        if self._iter % 2 == 0:
            self._update_all_parameters(obs, act, act_raw, adv_t, old, entropy_lb)

        # swap clusters and optimize mean and cov
        else:
            # deleted = self._cleanup(self._kl_del, obs, old, adv_t, act)
            # if deleted:
            #     self._increase_cw(deleted, self._kl_cw_after_del, obs, old)
            # else:
            #     swapped = self._random_swap_clusters(obs, old, act, adv_t)
            # if deleted or swapped:
            #     print('doing partial update')
            #     self._update_mean_n_cov(obs, act, adv_t, old, entropy_lb)
            # else:
            #     print('doing full update')
            #     self._update_all_parameters(obs, act, adv_t, old, entropy_lb)
            temp_change = False
            if self._do_opt_temp:
                temp_change = self._opt_temp(self._kl_temp, obs, act, adv_t, old)
            swapped = self._random_swap_clusters(obs, old, act, adv_t)
            deleted = []
            if self._do_delete and mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) < self._kl_del:
                deleted = self._cleanup(self._kl_del, obs, old, adv_t, act)
                if deleted:
                    self._increase_cw(deleted, self._kl_cw_after_del, obs, old)
            # if self._do_lsearch_temp and mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) < self._kl_del:
            #     temp_change = self._lsearch_temp(self._kl_del, obs, act, adv_t, old)
            if swapped or deleted or temp_change:
                print('doing partial update')
                old['intermediate_cmeans'] = self.policy.get_cmeans_t().clone().detach()
                self._update_mean_n_cov(obs, act, act_raw, adv_t, old, entropy_lb)
                full_batch_proj = False
            else:
                print('doing full update')
                self._update_all_parameters(obs, act, act_raw, adv_t, old, entropy_lb)

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
                # self.policy._regressor.means.data[k] = to_float_tensor(np.arctanh(np.clip(act[ba].detach().numpy(), -.99, .99)), self._use_cuda)
                self.policy._regressor.means.data[k] = act[ba]
                self.policy.set_cmeans_t(self.policy._regressor.means)

                ba_ind += 1
        tqdm.write('added {} clusters'.format(ba_ind))

    def _update_mean_n_cov(self, obs, act, act_raw, adv_t, old, entropy_lb):
        # Compute mean projection (nu)
        intermediate_means = self.policy.get_mean_t(obs).detach()
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, act_raw_i, wq_i, old_means_i, old_log_p_i, adv_i, intermediate_means_i in \
                    minibatch_generator(self._batch_size, obs, act, act_raw, old['w'], old['means'], old['log_p'], adv_t, intermediate_means):

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
                    if self._squash == 'tanh':
                        prob_ratio = torch.exp(MetricPolicy.log_prob_from_distrib(proj_distrib, act_i, act_raw_i) - old_log_p_i)
                    else:
                        prob_ratio = torch.exp(MetricPolicy.log_prob_from_distrib(proj_distrib, act_i) - old_log_p_i)
                    loss = -torch.mean(prob_ratio * adv_i)
                    loss.backward()

                    for opt in self._actor_optimizers[1:]:
                        opt.step()

                    if self._squash == 'tanh' or self._squash == 'clip':
                        self.policy.set_cmeans_t(self.policy.get_cmeans_t())

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

    def _opt_temp(self, kl_ub, obs, act, adv, old):
        temp_change = False
        temp_backup = self.policy._regressor._log_temp.clone()
        nb_success = 0
        self._temp_lr = self._temp_base_lr
        for param_group in self._temp_optimizer.param_groups:
            param_group['lr'] = self._temp_lr
        for k in range(500):
            self._temp_optimizer.zero_grad()

            w = self.policy.get_unormalized_membership_t(obs)
            loss = -torch.mean(torch.max(w, dim=1)[0]) + torch.mean(torch.topk(w, k=3, dim=1)[0])
            # topk = torch.topk(w, k=3, dim=1)[0]
            # loss = -torch.mean(topk[:, 0] - torch.mean(topk[:, 1:]))

            # new_pol_dist = self.policy.distribution_t(obs)
            # loss = -torch.mean(torch.exp(new_pol_dist.log_prob(act)[:, None] - old['log_p']) * adv)

            loss.backward()
            self._temp_optimizer.step()
            if mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) > kl_ub:
                self.policy._regressor._log_temp.data = temp_backup
                self._temp_lr *= .8
                for param_group in self._temp_optimizer.param_groups:
                    param_group['lr'] = self._temp_lr
                nb_success = 0
            else:
                temp_backup = self.policy._regressor._log_temp.clone()
                temp_change = True
                nb_success += 1
                if nb_success > 100:
                    nb_success = 0
                    self._temp_lr *= 1.25
                    for param_group in self._temp_optimizer.param_groups:
                        param_group['lr'] = self._temp_lr
        print('lr:', self._temp_lr)
        if temp_change:
            print('temp changed to {}'.format(self.policy._regressor._log_temp))
        else:
            print('temp unchanged')
        return temp_change

    def _lsearch_temp(self, kl_ub, obs, act, adv, old):
        print('dead code, lsearch temp')
        exit()
        # changed_temp = False
        # eval_budget = 2000
        # best_perf = base_perf = torch.mean(torch.exp(old['pol_dist'].log_prob(act)[:, None] - old['log_p']) * adv)  # almost zero
        # best_temp = base_temp = self.policy._regressor._log_temp.clone()
        # delta = .5
        #
        # # searching ub
        # budget_up = eval_budget / 2
        # ub = base_temp.clone()
        # while eval_budget > budget_up:
        #     ub += delta
        #     self.policy._regressor._log_temp = ub
        #     new_pol_dist = self.policy.distribution_t(obs)
        #     new_perf = torch.mean(torch.exp(new_pol_dist.log_prob(act)[:, None] - old['log_p']) * adv)
        #     if new_perf > best_perf and mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) < kl_ub:
        #         best_perf = new_perf
        #         best_temp = ub
        #         changed_temp = True
        #     else:
        #         break
        #     eval_budget -= 1
        #
        # # searching down
        # if not changed_temp:
        #     lb = base_temp.clone()
        #     while eval_budget:
        #         lb -= delta
        #         self.policy._regressor._log_temp = lb
        #         new_pol_dist = self.policy.distribution_t(obs)
        #         new_perf = torch.mean(torch.exp(new_pol_dist.log_prob(act)[:, None] - old['log_p']) * adv)
        #         if new_perf > best_perf and mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) < kl_ub:
        #             best_perf = new_perf
        #             best_temp = lb
        #             changed_temp = True
        #         elif changed_temp:
        #             break
        #         eval_budget -= 1
        # else:
        #     lb = base_temp
        #
        # for k in range(eval_budget):
        #     t = (lb + ub) / 2
        #     self.policy._regressor._log_temp = t
        #     new_pol_dist = self.policy.distribution_t(obs)
        #     new_perf = torch.mean(torch.exp(new_pol_dist.log_prob(act)[:, None] - old['log_p']) * adv)
        #     if new_perf > best_perf and mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec']) < kl_ub:
        #         best_perf = new_perf
        #         best_temp = t
        #         changed_temp = True
        #         ub = t
        #     else:
        #         lb = t
        # self.policy._regressor._log_temp = best_temp
        # if changed_temp:
        #     print('log temp changed from {} to {}, perf inc'.format(base_temp, best_temp, best_perf - base_perf))
        # else:
        #     print('temp unchanged')
        # return changed_temp

    def _swap_clusters_adv(self, obs, act, adv_t, old):
        print('dead code, swap clusters')
        exit()
        # base_perf = torch.mean(torch.exp(old['pol_dist'].log_prob(act)[:, None] - old['log_p']) * adv_t)  # almost zero
        #
        # high_adv_idxs = torch.argsort(adv_t, dim=0, descending=True)
        # remaining_idx = [k for k in range(self.policy._regressor._n_clusters)]
        # state_centers = self.policy._regressor.centers.clone().numpy()
        # for idxidx, idx in enumerate(high_adv_idxs):
        #     # finding closest state
        #     dist_to_centers = -np.sum((state_centers - obs[idx].detach().numpy()) ** 2, axis=1)
        #     closest_idxs = np.argsort(dist_to_centers)
        #     for closest_idx in closest_idxs:
        #         closest = remaining_idx[closest_idx]
        #         state_backup = self.policy._regressor.centers[closest].clone()
        #         # swap and check KL
        #         self.policy._regressor.centers[closest] = obs[idx]
        #         kl_swap = mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec'])
        #         if kl_swap < self._max_kl:
        #             # check increase of the objective after state swapping
        #             # ... and cluster center swapping?
        #             action_backup = self.policy._regressor.means[closest].clone()
        #             self.policy._regressor.means.data[closest] = act[idx]
        #             new_pol_dist = self.policy.distribution_t(obs)
        #             new_perf = torch.mean(torch.exp(new_pol_dist.log_prob(act)[:, None] - old['log_p']) * adv_t)
        #             self.policy._regressor.means.data[closest] = action_backup
        #             if new_perf >= base_perf:
        #             # if True:
        #                 # keep if kl under thresh and objective did not decrease
        #                 remaining_idx.pop(closest_idx)
        #                 state_centers = np.delete(state_centers, closest_idx, axis=0)
        #                 break
        #             else:
        #                 # revert old cluster center
        #                 self.policy._regressor.centers[closest] = state_backup
        #         else:
        #             # revert old cluster center
        #             self.policy._regressor.centers[closest] = state_backup
        #     if len(remaining_idx) == 0:
        #         break
        # new_pol_dist = self.policy.distribution_t(obs)
        # logging_kl = torch.mean(torch.distributions.kl.kl_divergence(new_pol_dist, old['pol_dist']))
        # print('Nb swapped clusters: {}. KL: {}'.format(self.policy._regressor._n_clusters - len(remaining_idx), logging_kl))

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

        if self._clus_sel == 'adv':
            sample_h = adv.detach().numpy().squeeze()
        else:
            if self._clus_sel.startswith('old'):
                sample_h = -torch.sum(w, dim=1)
                sample_h = sample_h.detach().numpy().squeeze()
            else:
                sample_h = torch.mean(self.policy._regressor.to_clust_dist(obs), dim=1)
                sample_h = sample_h.detach().numpy().squeeze()

        def bound_function(c_i):
            c_old = self.policy.get_cluster_centers()
            self.policy.set_cluster_centers(c_i)
            kl_swap = mean_diff(self.policy.get_mean_t(obs), old['means'], old['prec'])
            self.policy.set_cluster_centers(c_old)
            return kl_swap < self._max_kl

        def evaluation_function_old(c_i):
                c_old = self.policy.get_cluster_centers()
                self.policy.set_cluster_centers(c_i)
                w = self.policy.get_membership_t(obs)
                self.policy.set_cluster_centers(c_old)
                if self._clus_sel == 'old_min':
                    return torch.min(torch.sum(w, dim=1)).item()
                elif self._clus_sel == 'old_covr':
                    return torch.mean(torch.max(w, dim=1)[0]).item()
                else:
                    return torch.mean(torch.max(w, dim=1)[0]).item() - torch.mean(torch.topk(w, k=3, dim=1)[0]).item()
                    # topk = torch.topk(w, k=3, dim=1)[0]
                    # return torch.mean(topk[:, 0] - torch.mean(topk[:, 1:]))


        def evaluation_function(c_i, clust_idxs=None, samp_idxs=None):
            if self._clus_sel.startswith('old'):
                return evaluation_function_old(c_i)
            if self._clus_sel == 'adv':
                print('dead code, adv evaluation_function')
                exit()
                # c_old = self.policy.get_cluster_centers()
                # cmeas_backup = self.policy._regressor.means.clone()
                #
                # self.policy.set_cluster_centers(c_i)
                # if clust_idxs is not None:
                #     for k, si in zip(clust_idxs, samp_idxs):
                #         self.policy._regressor.means.data[k] = act[si]
                # prob_ratio = torch.exp(self.policy.log_prob_t(obs, act) - old['log_p'])
                # fitness = torch.mean(prob_ratio * adv).item()
                #
                # self.policy.set_cluster_centers(c_old)
                # self.policy._regressor.means.data = cmeas_backup
                # return fitness
            elif self._clus_sel.startswith('covr_exp'):
                c_old = self.policy.get_cluster_centers()
                self.policy.set_cluster_centers(c_i)
                w = self.policy._regressor._cluster_distance(obs)
                self.policy.set_cluster_centers(c_old)
                if self._clus_sel == 'covr_exp':
                    return torch.mean(torch.max(w, dim=1)[0]).item() - torch.mean(torch.topk(w, k=3, dim=1)[0]).item()
                else:
                    topk = torch.topk(w, k=3, dim=1)[0]
                    return torch.mean(topk[:, 0] - torch.mean(topk[:, 1:])).item()
            else:
                c_old = self.policy.get_cluster_centers()
                self.policy.set_cluster_centers(c_i)
                samp_to_clu_dist = -self.policy._regressor.to_clust_dist(obs)
                self.policy.set_cluster_centers(c_old)
                if self._clus_sel == 'min':
                    return torch.min(torch.mean(samp_to_clu_dist, dim=1)).item()
                elif self._clus_sel == 'covr':
                    return torch.mean(torch.max(samp_to_clu_dist, dim=1)[0]).item()
                # return torch.mean(samp_to_clu_dist).item() + np.mean(pdist(c_i))
                else:
                    return torch.mean(samp_to_clu_dist).item() + np.min(pdist(c_i))

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

    def _update_all_parameters(self, obs, act, act_raw, adv_t, old, entropy_lb):
        for epoch in range(self._n_epochs_per_fit):
            for obs_i, act_i, act_raw_i, wq_i, old_means_i, old_log_p_i, adv_i in \
                    minibatch_generator(self._batch_size, obs, act, act_raw, old['w'], old['means'], old['log_p'], adv_t):
                for opt in self._actor_optimizers:
                    opt.zero_grad()

                # Get Basics stuff
                w = self.policy.get_unormalized_membership_t(obs_i)
                means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'])
                chol = self.policy.get_chol_t()

                # Compute cluster weights projection (eta)
                eta = cweight_mean_proj(w, means_i, wq_i, old_means_i, old['prec'], self._max_kl)
                cweights_eta = eta * self.policy.get_cweights_t() + (1 - eta) * old['cweights']

                # Compute mean projection (nu)
                intermediate_means_i = self.policy.get_intermediate_mean_t(obs_i, old['cmeans'], cweights_eta)
                means_i = self.policy.get_mean_t(obs_i)

                proj_d = lin_gauss_kl_proj(means_i, chol, intermediate_means_i, old_means_i,
                                           old['cov'], old['prec'], old['logdetcov'],
                                           self._max_kl, entropy_lb)
                proj_distrib = torch.distributions.MultivariateNormal(proj_d['means'], scale_tril=proj_d['chol'])

                # Compute loss
                if self._squash == 'tanh':
                    prob_ratio = torch.exp(MetricPolicy.log_prob_from_distrib(proj_distrib, act_i, act_raw_i) - old_log_p_i)
                else:
                    prob_ratio = torch.exp(MetricPolicy.log_prob_from_distrib(proj_distrib, act_i) - old_log_p_i)
                loss = -torch.mean(prob_ratio * adv_i)
                loss.backward()

                for opt in self._actor_optimizers:
                    opt.step()

                if self._squash == 'tanh' or self._squash == 'clip':
                    self.policy.set_cmeans_t(self.policy.get_cmeans_t())

    def _full_batch_projection_all_par(self, obs, old, entropy_lb):
        # Get Basics stuff
        w = self.policy.get_unormalized_membership_t(obs)
        means = self.policy.get_intermediate_mean_t(obs, old['cmeans'])
        chol = self.policy.get_chol_t()

        # Compute cluster weights projection (eta)
        eta = cweight_mean_proj(w, means, old['w'], old['means'], old['prec'], self._max_kl)
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

