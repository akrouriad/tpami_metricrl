import numpy as np
import matplotlib.pyplot as plt

nb_run = 11
suff = [''] + ['-' + str(k+1) for k in range(nb_run - 1)]
algs = ['log/BipedalWalker-v2/projection_rand1k10_2019-12-18_17-47-04_10/', 'log/BipedalWalker-v2/projection_randadv_2019-12-18_17-54-05_5/', 'log/BipedalWalker-v2/projection_rand_lessfreq_2019-12-16_17-44-56_5/', 'log/BipedalWalker-v2/projection_rand10k_2019-12-16_17-48-57_5/', 'log/BipedalWalker-v2/projection_random1K_2019-12-13_17-55-11_5/', 'log/BipedalWalker-v2/projection_del_2019-12-13_18-06-10_5/']
names = ['cov1k10', 'randadv', 'lessfreq', 'swap10k', 'swap', 'del']

all_algs = []
for alg in algs:
    all_iters = []
    for k in range(nb_run):
        all_iters.append(np.load(alg + 'R' + suff[k] + '.npy'))
    all_algs.append(all_iters)

all_algs = np.array(all_algs)

for k, name in enumerate(names):
    print(name, np.median(all_algs[k], axis=0))
    plt.plot(np.median(all_algs[k], axis=0))
plt.legend(names)
plt.show()
