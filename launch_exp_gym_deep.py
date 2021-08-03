from experiment_launcher import Launcher

if __name__ == '__main__':
    local = False
    test = False

    exp = 'big'
    # exp = 'small'

    launcher = Launcher(exp_name='deep_baselines',
                        python_file='exp_gym_deep',
                        n_exps=25,
                        memory=2000,
                        hours=24,
                        minutes=0,
                        seconds=0,
                        joblib_n_jobs=5,
                        partition='test24',
                        use_timestamp=True)

    algs = ['PPO', 'TRPO']

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
            launcher.add_experiment(env_id=env, alg_name=alg)

        launcher.run(local, test)
