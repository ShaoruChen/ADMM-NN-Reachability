
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
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, n)
    )
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    nn_system = torch.load('trained_nn_model_8_layer.pt')
    horizon = 2
    seq_nn_system = SequentialModel(nn_system, horizon)

    seq_layers = seq_nn_system.layer_list
    nn_model_relaxation = nn.Sequential(*seq_layers)

    x0 = torch.tensor([-1.75, -1.75])
    x0 = x0.unsqueeze(0)
    epsilon = torch.tensor([0.75, 0.75])
    epsilon = epsilon.unsqueeze(0)

    nx = 2
    lb, ub = pre_activation_bounds_dual(nn_model_relaxation, x0, epsilon)