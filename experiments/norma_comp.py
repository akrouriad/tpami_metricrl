test = False
local = False
dir_out = '/home/ra61casa/normacomp/'
# dir_out = './normacomp/'

base = '_norma'
envs = ['BipedalWalker-v2', 'RoboschoolHopper-v1', 'RoboschoolInvertedDoublePendulum-v1', 'RoboschoolWalker2d-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1', 'MountainCarContinuous-v0']
nb_runs = 5
all_par = []
max_tss = [1e3]
normas = ['None', 'Obs', 'All']
# Creating parameters tables
for max_ts in max_tss:
    for run in range(nb_runs):
        for env in envs:
            for norma in normas:
                run_name = env + 'norma' + norma + 'mts' + str(max_ts / 1e6) + 'm' + 'run' + str(run)
                params = {'envid': env, 'seed': run, 'log_name': dir_out+run_name, 'max_ts': max_ts, 'norma': norma}
                all_par.append(params)

# Creating launch scripts
slurms = []
nb_par = 16
for k, i in enumerate(range(0, len(all_par), nb_par)):
    # create python script
    script_name = base + '_script' + str(k) + '.py'
    with open(script_name, "w") as file:
        code = """\
import sys
from multiprocessing import Process
def learn_process(dict):
  sys.stdout = open(dict['log_name'] + ".out", "w")
  import metricrl_on_gym
  metricrl_on_gym.learn(**dict)
     
ps = []
params = {}
for par in params:
  p = Process(target=learn_process, args=(par, ))
  p.start()
  ps.append(p)    

for p in ps:
  p.join()
""".format(all_par[i:i+nb_par])
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
            print("#SBATCH -o /work/scratch/ra61casa/env/logs/" + slurms[-1] + ".stdout", file=file)
            print("#SBATCH -e /work/scratch/ra61casa/env/logs/" + slurms[-1] + ".stderr", file=file)

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
            print("#SBATCH -c " + str(nb_par), file=file)
            print("cd \nsource .bash_profile", file=file)
            print("cd /home/ra61casa/src/baselines/experiments", file=file)
            print("PYTHONPATH=$(cd ..;pwd) python " + script_name, file=file)
        
if not test:
    import os, time
    time.sleep(2)
    if local:
        for k, i in enumerate(range(0, len(all_par), nb_par)):
            script_name = base + '_script' + str(k) + '.py'
            os.system("PYTHONPATH=$(cd ..;pwd) python3 " + script_name)
    else:
        for slurm in slurms:
            os.system("sbatch " + slurm)
