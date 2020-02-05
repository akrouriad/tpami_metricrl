import sys
sys.path.append('../')
from metric_rl.logger import generate_log_folder

test = False
local = False

tu_id = 'dt11kypo'
home = '~/'
# tu_id = 'ra61casa'
# home = '~/src/'

#!
# experiment_name = 'final_big'
# experiment_name = 'final_medium_h2'
# experiment_name = 'final_small'
experiment_name = 'final_small2'

cluster_log_dir = '/work/scratch/' + tu_id + '/logs/' + experiment_name + '/'
cluster_script_dir = home + 'metricrl/experiments'
cluster_python_cmd = 'python'

local_python_cmd = 'python'
local_log_dir = experiment_name + '/'

#!
# base = '_fbig'
# base = '_fmed_h2'
# base = '_fsml'
base = '_fsml2'

#!
# envs = ['HumanoidBulletEnv-v0']
# envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']
# envs = ['MountainCarContinuous-v0', 'BipedalWalker-v2', 'InvertedDoublePendulumBulletEnv-v0']
envs = ['BipedalWalker-v2', 'InvertedPendulumBulletEnv-v0', 'Pendulum-v0', 'InvertedPendulumSwingupBulletEnv-v0']

#!
# temp_per_envs = [.33]
# temp_per_envs = [1., 1., .33, .33]
# temp_per_envs = [1., 1., 1.]
temp_per_envs = [1., 1., 1., 1.]

horizons = [1600, 1000, 200, 1000]
nb_runs = 25

#!
# n_epochs = 3500
# n_epochs = 1000
n_epochs = 500

n_steps = 3008
n_steps_per_fit = 3008
n_episodes_test = 5

all_par = []

#!
# n_clusterss = [20, 40, 80]
# n_clusterss = [10, 20, 40]
n_clusterss = [5, 10, 20]

opt_temp = False
# n_clusterss = [10]
# clus_sels = ['adv', 'old_covr', 'covr', 'covr_minpen']
# clus_sels = ['old_covr', 'covr']
# clus_sels = ['old_covr', 'old_covr_yetnew', 'covr']
# clus_sels = ['old_covr_yetnew']
clus_sels = ['covr_exp']
# clus_dels = [True, False]
clus_dels = [True]
squash = 'none'
max_cmean = 1.
alg_name = 'metricrl'

# Creating parameters tables
for env, temp_per_env, horizon in zip(envs, temp_per_envs, horizons):
    for n_clusters in n_clusterss:
        for clus_sel in clus_sels:
            for clus_del in clus_dels:
                postfix = 'c' + str(n_clusters) + 'h' + clus_sel + 'd' + str(clus_del) + 't' + str(temp_per_env) + 's' + squash
                log_name = generate_log_folder(name=env, algorithm_name=alg_name, postfix=postfix,
                                               timestamp=False, base_folder=local_log_dir if local else cluster_log_dir)

                for run in range(nb_runs):
                    params = {'env_id': env, 'n_clusters': n_clusters, 'horizon': horizon, 'seed': run, 'log_name': log_name,
                              'n_epochs': n_epochs, 'n_steps': n_steps, 'n_steps_per_fit': n_steps_per_fit,
                              'n_episodes_test': n_episodes_test, 'clus_sel': clus_sel, 'do_delete': clus_del,
                              'opt_temp': opt_temp, 'temp': temp_per_env, 'squash': squash, 'max_cmean': max_cmean}
                    all_par.append(params)

# Creating launch scripts
slurms = []
nb_proc_per_act = 1
nb_act = 16
nb_proc = nb_act * nb_proc_per_act
for k, i in enumerate(range(0, len(all_par), nb_act)):
    # create python script
    script_name = base + '_script' + str(k) + '.py'
    with open(script_name, "w") as file:
        code = """\
import sys
import os
from multiprocessing import Process
def learn_process(dict):
  sys.stdout = open(os.path.join(dict['log_name'], 'console' + str(dict['seed']) + '.out'), "w")
  sys.stderr = open(os.path.join(dict['log_name'], 'console' + str(dict['seed']) + '.err'), "w")
  import run_gym
  run_gym.experiment(**dict)
     
ps = []
params = {}
for par in params:
  p = Process(target=learn_process, args=(par, ))
  p.start()
  ps.append(p)    

for p in ps:
  p.join()
""".format(all_par[i:i+nb_act])
        for c in code.splitlines():
            print(c, file=file)
    
    if not local:
        # create slurm script
        slurms.append(base + '_slurm' + str(k))
        with open(slurms[-1], "w") as file:
            code = """\
#!/bin/bash
# Job name"""
            for c in code.splitlines():
                print(c, file=file)
            print("#SBATCH -J " + slurms[-1], file=file)
            print("#SBATCH -o " + cluster_log_dir + slurms[-1] + ".stdout", file=file)
            print("#SBATCH -e " + cluster_log_dir + slurms[-1] + ".stderr", file=file)

            code = """\
# request computation time hh:mm:ss
#SBATCH -t 24:00:00

# request virtual memory in MB per core
#SBATCH --mem-per-cpu=2000

#nodes for a single job
#SBATCH -n 1"""
            for c in code.splitlines():
                print(c, file=file)
            # print("#SBATCH -C avx2", file=file)
            print("#SBATCH -c " + str(nb_proc), file=file)
            print("cd \nsource .bash_profile", file=file)
            print("cd " + cluster_script_dir, file=file)
            print("PYTHONPATH=$(cd ..;pwd) " + cluster_python_cmd + " " + script_name, file=file)
        
if not test:
    import os, time
    time.sleep(2)
    if local:
        for k, i in enumerate(range(0, len(all_par), nb_act)):
            script_name = base + '_script' + str(k) + '.py'
            os.system("PYTHONPATH=$(cd ..;pwd) " + local_python_cmd + " " + script_name)
            # os.system(local_python_cmd + " " + script_name)
    else:
        for slurm in slurms:
            os.system("sbatch " + slurm)
