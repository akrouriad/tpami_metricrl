test = False
local = True
# dir_out = '/home/ra61casa/cluscomp/'
dir_out = './cluscomp/'

base = '_cc'
# envs = ['BipedalWalker-v2', 'RoboschoolHopper-v1', 'RoboschoolInvertedDoublePendulum-v1', 'RoboschoolWalker2d-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1', 'MountainCarContinuous-v0']
# envs = ['RoboschoolHopper-v1', 'RoboschoolWalker2d-v1', 'RoboschoolHalfCheetah-v1', 'RoboschoolAnt-v1']
# envs = ['RoboschoolHopper-v1', 'RoboschoolWalker2d-v1', 'BipedalWalker-v2']
envs = ['BipedalWalker-v2', 'RoboschoolHopper-v1']
nb_runs = 2
all_par = []
max_tss = [3e6]
# normas = ['None', 'Obs', 'All']
# normas = ['None']
normas = ['None']
# aggreg_types = ['None', 'Max', 'Min']
aggreg_types = ['Min']
# max_cluss = [5, 10, 20, 40]
max_cluss = [5, 10]
min_sam_iter = 3000
nb_v = 2
# Creating parameters tables
for max_ts in max_tss:
    for run in range(nb_runs):
        for env in envs:
            for norma in normas:
                for aggreg_type in aggreg_types:
                    for max_clus in max_cluss:
                        run_name = env + 'sampIt' + str(min_sam_iter) + 'nb_v' + str(nb_v) + 'norma' + norma + 'aggreg' + aggreg_type + 'mts' + str(max_ts / 1e6) + 'm' + 'max_clus' + str(max_clus) + 'run' + str(run)
                        params = {'envid': env, 'nb_max_clusters': max_clus, 'seed': run, 'log_name': dir_out+run_name, 'max_ts': max_ts, 'norma': norma, 'aggreg_type': aggreg_type, 'nb_vfunc': nb_v, 'min_sample_per_iter': min_sam_iter}
                        all_par.append(params)


# Creating launch scripts
slurms = []
nb_proc_per_act = 1
nb_act = 6
nb_proc = nb_act * nb_proc_per_act
for k, i in enumerate(range(0, len(all_par), nb_act)):
    # create python script
    script_name = base + '_script' + str(k) + '.py'
    with open(script_name, "w") as file:
        code = """\
import sys
from multiprocessing import Process
def learn_process(dict):
  sys.stdout = open(dict['log_name'] + ".out", "w")
  import proj_metricrl
  proj_metricrl.learn(**dict)
     
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
            print("#SBATCH -c " + str(nb_proc), file=file)
            print("cd \nsource .bash_profile", file=file)
            print("cd /home/ra61casa/src/metricrl/experiments", file=file)
            print("PYTHONPATH=$(cd ..;pwd) python " + script_name, file=file)
        
if not test:
    import os, time
    time.sleep(2)
    if local:
        for k, i in enumerate(range(0, len(all_par), nb_act)):
            script_name = base + '_script' + str(k) + '.py'
            os.system("PYTHONPATH=$(cd ..;pwd) python3 " + script_name)
    else:
        for slurm in slurms:
            os.system("sbatch " + slurm)