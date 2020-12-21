from experiment_launcher import Launcher

if __name__ == '__main__':
    local = False
    test = False
    exp = 'bigdiff'
    #exp = 'smalldiff'

    launcher = Launcher(exp_name='baselines_diff_mushroom',
                        # python_file='exp_gym_deep',
                        python_file='exp_gym_diffmetric',
                        n_exp=25,
                        memory=2000,
                        hours=24,
                        minutes=0,
                        seconds=0,
                        n_jobs=-1,
                        use_timestamp=True)

    algs = ['PPO', 'TRPO']
    nb_centers_list = [10]

    if exp == 'bigdiff':
        envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
        horizons = [1000, 1000, 1000, 1000]
        n_epochs = 1000

    elif exp == 'smalldiff':
        envs = ['MountainCarContinuous-v0', 'BipedalWalker-v3', 'Pendulum-v0', 'InvertedPendulumBulletEnv-v0',
                'InvertedPendulumSwingupBulletEnv-v0', 'InvertedDoublePendulumBulletEnv-v0']
        horizons = [1000, 1600, 200, 1000, 1000, 1000]
        n_epochs = 500

    else:
        raise RuntimeError

    launcher.add_default_params(
        n_epochs=n_epochs,
        n_steps=3008,
        n_steps_per_fit=3008,
        n_episodes_test=5,
    )

    for env, horizon in zip(envs, horizons):
        launcher.add_default_params(horizon=horizon)
        for alg in algs:
            for nb_centers in nb_centers_list:
                launcher.add_experiment(env_id=env,
                                        alg_name=alg, nb_centers=nb_centers)

        launcher.run(local, test)
