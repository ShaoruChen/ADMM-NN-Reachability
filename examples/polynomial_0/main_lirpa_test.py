
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("always")
import cvxpy as cp
import numpy as np

from pympc.geometry.polyhedron import Polyhedron
import nn_reachability.utilities as ut
import matplotlib.pyplot as plt
from nn_reachability.nn_models import SequentialModel, SystemDataSet, train_nn_torch

from nn_reachability.LP_relaxation import pre_activation_bounds_dual

import sys
sys.path.append(r'C:\GithubRepository\auto_LiRPA')
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def system_dynamics(x):
    y_1 = x[0] + 0.2*(x[0] - x[0]**3/3 - x[1] + 0.875)
    y_2 = x[1] + 0.2*(0.08*(x[0] + 0.7 - 0.8*x[1]))
    y = np.array([y_1, y_2])
    return y


def nn_dynamics(n=2):
    # neural network dynamics with randomly generated weights
    model = nn.Sequential(
        nn.Linear(n, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, n)
    )
    return model

def new_nn():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Linear(2, 1)
    )
    return model

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # nn_system = torch.load('trained_nn_model_adaptive_layer.pt')
    # horizon = 1
    # seq_nn_system = SequentialModel(nn_system, horizon)
    seq_nn_system = new_nn()

    my_input = torch.tensor([[-1.75, -1.75, 0.2]])
    output = seq_nn_system(my_input)

    eps_input = torch.tensor([[0.1]])
    eps_input = eps_input.repeat(my_input.size(0), 1)

    x_lb = torch.rand(4, 3)
    x_ub = x_lb + torch.rand(4, 3).abs()
    center = (x_lb + x_ub)/2
    radius = (x_ub - x_lb)/2

    # Wrap the model with auto_LiRPA
    model = BoundedModule(seq_nn_system, my_input)
    ptb = PerturbationLpNorm(norm=np.inf, eps=radius)
    # Make the input a BoundedTensor with perturbation
    my_input = BoundedTensor(center, ptb)
    # Regular forward propagation using BoundedTensor works as usual.
    prediction = model(my_input)
    # Compute LiRPA bounds
    lb_0, ub_0 = model.compute_bounds(x=(my_input,), method='CROWN')
    lb, ub, A_dict = model.compute_bounds(x = (my_input,), method = 'IBP+backward', return_A = True)





