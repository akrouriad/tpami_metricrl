import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import data_handling as dat
import rllog


class MetricPolicy(nn.Module):
    def __init__(self, a_dim):
        super().__init__()
        self.centers = []
        self.weights = nn.Parameter(torch.ones(1))
        self.means = nn.Parameter(torch.zeros(1, a_dim))
        self.cov = nn.Parameter(torch.eye(a_dim))
