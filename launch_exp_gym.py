from experiment_launcher import Launcher

if __name__ == '__main__':
    local = True
    test = False
    exp = 'big'
    #exp = 'small'

    launcher = Launcher(exp_name='metricrl',
                        python_file='exp_gym',
                        n_exp=25,
                        memory=2000,
                        hours=24,
                        minutes=0,
                        seconds=0,
                        n_jobs=-1,
                        partition='test24',
                        use_timestamp=True)

    if exp == 'big':
        envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
        temp_per_envs = [1., 1., .33, .33]
        n_epochs = 1000

        n_clusterss = [10, 20, 40]

    elif exp == 'small':
        envs = ['MountainCarContinuous-v0', 'BipedalWalker-v3', 'Pendulum-v0', 'InvertedPendulumBulletEnv-v0',
                'InvertedPendulumSwingupBulletEnv-v0', 'InvertedDoublePendulumBulletEnv-v0']
        temp_per_envs = [1., 1., 1., 1., 1., 1., 1.]
        n_epochs = 500

        n_clusterss = [5, 10, 20]

    else:
        raise RuntimeError

    launcher.add_default_params(
        n_epochs=n_epochs,
        n_steps=3008,
        n_steps_per_fit=3008,
        n_episodes_test=5,
    )

    for env, temp in zip(envs, temp_per_envs):
        launcher.add_default_params(temp=temp)
        for n_clusters in n_clusterss:
            launcher.add_experiment(env_id=env, n_clusters=n_clusters)

        launcher.run(local, test)
