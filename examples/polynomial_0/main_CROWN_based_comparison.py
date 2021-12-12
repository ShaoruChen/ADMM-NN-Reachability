
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
from nn_reachability.nn_models import iterative_output_Lp_bounds_LiRPA, output_Lp_bounds_LiRPA

''' FitzHugh-Nagumo Neuron Model from 
Semidefinite Approximations of Reachable Sets for Discrete-time Polynomial Systems, Morgan et al., 2019'''


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
        nn.Linear(10, n)
    )
    return model


if __name__ == '__main__':
    is_train = False

    if is_train:
        '''train a neural network to approximate the given dynamics'''
        # domain of the system
        x_min = np.array([-5.0, -5.0])
        x_max = np.array([5.0, 5.0])
        domain = Polyhedron.from_bounds(x_min, x_max)

        # uniformly sample from the domain to generate training data
        resol = 50
        x_train_samples, y_train_samples = ut.generate_training_data(system_dynamics, domain, resol, None)

        sample_data = {'train_data': x_train_samples, 'label_data': y_train_samples}
        ut.pickle_file(sample_data, 'training_data_set')

        train_data_set = SystemDataSet(x_train_samples, y_train_samples)

        train_batch_size = 10
        train_loader = DataLoader(train_data_set, batch_size=train_batch_size, shuffle=True)

        nn_structure = nn_dynamics(2)
        nn_model = train_nn_torch(train_loader, nn_structure, num_epochs=100, l1=None,
                                  pred_step=0, lr=1e-4, decay_rate=1.0, clr=None, path='torch_nn_model')

        torch.save(nn_model, 'trained_nn_modeL_adaptive_layer.pt')


    '''
    finite step reachability analysis
    '''
    # torch.set_grad_enabled(False)

    nn_system = torch.load('trained_nn_model_4_layer.pt')
    # nn_system.train()

    x0 = torch.zeros(2)
    nx = 2

    lb_0 = torch.tensor([[-1.85, -1.85]])
    ub_0 = torch.tensor([[-1.65, -1.65]])

    # lb_0 = torch.tensor([[-1.25, -1.25]])
    # ub_0 = torch.tensor([[-1.0, -1.0]])
    domain = Polyhedron.from_bounds(lb_0[0].numpy(), ub_0[0].numpy())

    # view the sampled trajectories
    # plt.figure()
    # init_states = ut.unif_sample_from_Polyhedron(domain, 5)
    # traj_list = ut.simulate_NN_system(nn_system, init_states, step=15)
    #
    # ut.plot_multiple_traj_tensor_to_numpy(traj_list)
    # domain.plot(fill=False, ec='r', linestyle='-.', linewidth=2)
    # plt.show()

    horizon = 3
    # reachability analysis through sequential nn
    print('Sequential reachability analysis \n')
    # method = 'CROWN-Optimized'
    method = 'forward'
    box = {'lb': lb_0[0].numpy(), 'ub': ub_0[0].numpy()}
    seq_bounds_list = [box]
    for i in range(horizon):
        nn_model = SequentialModel(nn_system, i+1)
        output_lb, output_ub = output_Lp_bounds_LiRPA(nn_model, lb_0, ub_0, method= method)
        # fixme: we only extract the bounds for the first batch
        box = {'lb': output_lb[0].detach().numpy(), 'ub': output_ub[0].detach().numpy()}
        seq_bounds_list.append(box)

    # reachability analysis through iterative methods
    print('Iterative reachability analysis \n')
    base_nn_system = SequentialModel(nn_system, 1)
    iter_bounds_list = iterative_output_Lp_bounds_LiRPA(base_nn_system, lb_0, ub_0, horizon, method= method)
    iter_bounds_list = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in iter_bounds_list]

    # add IBP baseline
    base_nn_system = SequentialModel(nn_system, 1)
    iter_bounds_IBP = iterative_output_Lp_bounds_LiRPA(base_nn_system, lb_0, ub_0, horizon, method= 'IBP')
    iter_bounds_IBP = [{'lb': item['lb'][0].detach().numpy(), 'ub': item['ub'][0].detach().numpy()} for item in iter_bounds_IBP]
    iter_poly_IBP = ut.bounds_list_to_polyhedron_list(iter_bounds_IBP)

    seq_poly_list = ut.bounds_list_to_polyhedron_list(seq_bounds_list)
    iter_poly_list = ut.bounds_list_to_polyhedron_list(iter_bounds_list)

    for i in range(horizon):
        plt.figure()
        seq_poly_list[i+1].plot(fill=False, ec='r', linestyle='-', linewidth=2)
        iter_poly_list[i+1].plot(fill=False, ec='b', linestyle='-.', linewidth=2)

        # add IBP baseline
        iter_poly_IBP[i+1].plot(fill =False, ec = 'g', linestyle = '-.', linewidth = 2)

        init_states = ut.unif_sample_from_Polyhedron(domain, 8)
        traj_list = ut.simulate_NN_system(nn_system, init_states, step=i)

        ut.plot_multiple_traj_tensor_to_numpy(traj_list)
        domain.plot(fill=False, ec='k', linestyle='--', linewidth=2)
        plt.title('Reachable set at step {}'.format(i+1))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')

    plt.show()