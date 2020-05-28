from experiment_launcher import Launcher

if __name__ == '__main__':
    local = True
    test = False
    exp = 'big'
    #exp = 'small'

    launcher = Launcher(exp_name='baselines_mushroom',
                        python_file='exp_gym_deep',
                        n_exp=25,
                        memory=2000,
                        hours=24,
                        minutes=0,
                        seconds=0,
                        n_jobs=-1,
                        use_timestamp=True)

    algs = ['PPO', 'TRPO']

    if exp == 'big':
        envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
        horizons = [1000, 1000, 1000, 1000]
        n_epochs = 1000

    elif exp == 'small':
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

    for alg in algs:
        for env, horizon in zip(envs, horizons):
            launcher.add_default_params(horizon=horizon)
            launcher.add_experiment(env_id=env,
                                    alg_name=alg)

    launcher.run(local, test)
