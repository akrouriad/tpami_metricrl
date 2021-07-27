from experiment_launcher import Launcher

if __name__ == '__main__':
    local = False
    test = False
    
    exp = 'big'
    # exp = 'small'
    
    exp_name = 'metricrl_diff'
    python_file = 'exp_gym_diffmetric'
    # exp_name = 'deep_baselines'
    # python_file = 'exp_gym_deep'

    launcher = Launcher(exp_name=exp_name,
                        python_file=python_file,
                        # n_exp=25,
                        n_exp=25,
                        memory=2000,
                        hours=24,
                        minutes=0,
                        seconds=0,
                        n_jobs=-1,
                        partition='test24',
                        use_timestamp=True)

    algs = ['PPO', 'TRPO']
    # algs = ['PPO']
    # nb_centers_list = [10, 20, 40]
    nb_centers_list = [10]
    init_cluster_noises = [0.1, 1.]

    if exp == 'big':
        envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
        n_epochs = 1000

    elif exp == 'small':
        envs = ['MountainCarContinuous-v0', 'BipedalWalker-v3', 'Pendulum-v0', 'InvertedPendulumBulletEnv-v0',
                'InvertedPendulumSwingupBulletEnv-v0', 'InvertedDoublePendulumBulletEnv-v0']
        n_epochs = 500

    else:
        raise RuntimeError

    launcher.add_default_params(
        n_epochs=n_epochs,
        n_steps=3008,
        n_steps_per_fit=3008,
        n_episodes_test=5,
    )

    for env in envs:
        for alg in algs:
            for nb_centers in nb_centers_list:
                for init_cluster_noise in init_cluster_noises:
                    launcher.add_experiment(env_id=env, alg_name=alg, nb_centers=nb_centers,
                                            init_cluster_noise=init_cluster_noise)

        launcher.run(local, test)
