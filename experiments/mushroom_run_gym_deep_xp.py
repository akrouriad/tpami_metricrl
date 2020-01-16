import sys
sys.path.append('../')
from metric_rl.logger import generate_log_folder

test = False
local = False

tu_id = 'dt11kypo'
home = '~/'
#tu_id = 'ra61casa'
#home = '~/src'

cluster_log_dir = '/work/scratch/' + tu_id + '/logs/'
cluster_script_dir = home + 'metricrl/experiments'
cluster_python_cmd = 'python'

local_python_cmd = 'python'

base = '_deep'
# envs = ['BipedalWalker-v2', 'RoboschoolHopper-v1', 'RoboschoolInvertedDoublePendulum-v1', 'RoboschoolWalker2d-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1', 'MountainCarContinuous-v0']
envs = ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'AntBulletEnv-v0']

horizon = 1000
gamma = .99
nb_runs = 11
n_epochs = 1000
n_steps = 3000
n_steps_per_fit = 3000
n_episodes_test = 5

all_par = []
algs = ['PPO', 'TRPO']

# Creating parameters tables
for env in envs:
    for alg_name in algs:
        log_name = generate_log_folder(name=env, algorithm_name=alg_name,
                                       timestamp=False, base_folder=cluster_log_dir)

        for run in range(nb_runs):
            params = {'alg_name': alg_name, 'env_id': env, 'horizon': horizon, 'gamma': gamma,
                      'seed': run, 'log_name': log_name,
                      'n_epochs': n_epochs, 'n_steps': n_steps, 'n_steps_per_fit': n_steps_per_fit,
                      'n_episodes_test': n_episodes_test}
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
  import run_gym_deep
  run_gym_deep.experiment(**dict)
     
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
            print("#SBATCH -C avx2", file=file)
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
