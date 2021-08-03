# Metric RL
Code for the paper "Continuous Action Reinforcement Learning from a Mixture of Interpretable Experts", 
Riad Akrour, Davide Tateo, and Jan Peters.

## Installation
To install the package you can run this command in the top-level folder:

    pip install -e .
    
This library uses the MushroomRL reinforcement learning library and the experiment-launcher package.
    
## Running experiments
You can run a single experiment by launching one of the three experiment files (`exp_gym.py`, `exp_gym_deep.py`, 
`exp_gym_diffmetric.py`). Use --help to know which are the available arguments. Important arguments to set  are the 
`--results-dir` and `--seed` command line arguments.
If you want to launch a batch of seeds, you can use the launch files (`launch_exp_gym.py`, `launch_exp_gym_deep.py`, 
`launch_exp_gym_diffmetric.py`). These files can launch a batch of experiments in a local machine using Joblib, or on a
Slurm cluster. You need to modify the launch file to fit it to your hardware requirements. In particular, the variable
`local` can be set to true to run with Joblib, while if it is set to false, it will submit the jobs to a cluster.

The code for the Metric RL algorithm can be launched using the `exp_gym.py` python file or its associate launcher file.
The differentiable version of the Metric RL algorithm can be launched using the `exp_gym_diffmetric.py` python file or 
its associate launcher file.
differently from the paper, in this code, we don't provide the stable-baselines implementation of deep reinforcement 
learning algorithms. The `exp_gym_deep.py` file and the associated launch file will use the Mushroom RL implementation 
of these algorithms.