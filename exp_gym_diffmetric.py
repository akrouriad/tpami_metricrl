import argparse

import torch
import numpy as np

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.utils.dataset import compute_J

from metric_rl.utils import save_parameters
from metric_rl.rl_shared import MLP, TwoPhaseEntropProfile

import torch.optim as optim
import torch.nn.functional as F

from metric_rl.policies import MetricPolicy

from experiment_launcher import get_default_params


def experiment(alg_name, env_id, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000, n_episodes_test=5, n_models_v=1,
               nb_centers=10, init_cluster_noise=1e-2, seed=0, results_dir='logs'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name='MetricRLDiff', results_dir=results_dir, seed=seed)

    mdp = Gym(env_id)

    # Set environment seed
    mdp.env.seed(seed)

    # Set parameters
    critic_params = dict(network=MLP,
                         loss=F.mse_loss,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         batch_size=64,
                         n_models=n_models_v,
                         size_list=[64, 64])

    alg, alg_params = get_alg_and_parameters(alg_name)

    policy = MetricPolicy(mdp.info.observation_space.shape, mdp.info.action_space.shape, nb_centers, std_0=1.,
                          learnable_centers=True, init_cluster_noise=init_cluster_noise)
    entropy_profile = TwoPhaseEntropProfile(policy, e_reduc=0.0075, e_thresh_mult=.5)

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)
    agent.set_logger(logger)

    # Save alg params
    save_parameters(results_dir, dict(alg_params=alg_params))

    # Run learning
    core = Core(agent, mdp)

    # Initial evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.log_numpy(J=J, R=R, E=E)
    logger.epoch_info(0, J=J, R=R, E=E)

    # Learning
    for it in range(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        logger.log_numpy(J=J, R=R, E=E)
        logger.epoch_info(it + 1, J=J, R=R, E=E)

        # update entropy lb
        policy.set_chol_t(policy.get_chol_t())
        policy._regressor.e_lb = entropy_profile.get_e_lb()

    logger.log_agent(agent)


def get_alg_and_parameters(alg_name):
    if alg_name == 'PPO':
        alg_params = dict(actor_optimizer={'class': optim.Adam,
                                           'params': {'lr': 3e-4}},
                          n_epochs_policy=10,
                          batch_size=64,
                          eps_ppo=.2,
                          lam=.95)

        return PPO, alg_params

    elif alg_name == 'TRPO':
        alg_params = dict(ent_coeff=0.0,
                          max_kl=.01,
                          lam=.95,
                          n_epochs_line_search=10,
                          n_epochs_cg=10,
                          cg_damping=1e-2,
                          cg_residual_tol=1e-10)

        return TRPO, alg_params

    else:
        raise RuntimeError


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg-name', type=str)
    parser.add_argument("--env-id", type=str)

    parser.add_argument("--n-models-v", type=int)

    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--n-steps-per-fit", type=int)
    parser.add_argument("--n-episodes-test", type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--nb-centers', type=int)
    parser.add_argument('--init-cluster-noise', type=float)

    parser.set_defaults(**get_default_params(experiment))
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    experiment(**args)
