import argparse

import torch
import numpy as np

from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gym
from mushroom_rl.algorithms.actor_critic import PPO, TRPO
from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J

from metric_rl.utils import save_parameters
from metric_rl.rl_shared import MLP

import torch.optim as optim
import torch.nn.functional as F

from experiment_launcher import get_default_params, add_launcher_base_args, run_experiment


def experiment(alg_name, env_id, n_epochs=1000, n_steps=3000, n_steps_per_fit=3000, n_episodes_test=5, n_models_v=1,
               seed=0, results_dir='logs'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    logger = Logger(log_name=alg_name, results_dir=results_dir, seed=seed)

    mdp = Gym(env_id)

    # Set environment seed
    mdp.env.seed(seed)

    # Set parameters
    policy_params = dict(std_0=1.,
                         size_list=[64, 64],
                         use_cuda=False)

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

    policy = GaussianTorchPolicy(MLP,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    agent = alg(mdp.info, policy, critic_params=critic_params, **alg_params)

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
        logger.epoch_info(it+1, J=J, R=R, E=E)


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

    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_default_params(experiment))
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    run_experiment(experiment, args)
