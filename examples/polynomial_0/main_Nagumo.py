
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


''' FitzHugh-Nagumo Neuron Model from 
Semidefinite Approximations of Reachable Sets for Discrete-time Polynomial Systems, Morgan et al., 2019'''


def system_dynamics(x):
    y_1 = x[0] + 0.2*(x[0] - x[0]**3/3 - x[1] + 0.875)
    y_2 = x[1] + 0.2*(0.08*(x[0] + 0.7 - 0.8*x[1]))
    y = np.array([y_1, y_2])
    return y


# def nn_dynamics(n=2):
#     # neural network dynamics with randomly generated weights
#     model = nn.Sequential(
#         nn.Linear(n, 900),
#         nn.ReLU(),
#         nn.Linear(900, 500),
#         nn.ReLU(),
#         nn.Linear(500, 200),
#         nn.ReLU(),
#         nn.Linear(200, 50),
#         nn.ReLU(),
#         nn.Linear(50, n)
#     )
#     return model

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

        torch.save(nn_model, 'trained_nn_modeL_large_layer.pt')


    '''
    finite step reachability analysis
    '''
    torch.set_grad_enabled(False)

    nn_system = torch.load('trained_nn_model_4_layer.pt')

    x0 = torch.zeros(2)
    nx = 2

    A_input = np.vstack((np.eye(nx), -np.eye(nx)))
    # b_input = np.array([0.0, 0.0, 0.5, 0.5])
    b_input = np.array([-1.65, -1.65, 1.85, 1.85])
    c_output = ut.unif_normal_vecs(nx, n = 4)

    # # view the sampled trajectories
    # plt.figure()
    # domain = Polyhedron(A_input, b_input)
    # init_states = ut.unif_sample_from_Polyhedron(domain, 5)
    # traj_list = ut.simulate_NN_system(nn_system, init_states, step=15)
    #
    # ut.plot_multiple_traj_tensor_to_numpy(traj_list)
    # domain.plot(fill=False, ec='r', linestyle='-.', linewidth=2)
    # plt.show()

    horizon = 5
    # reachability analysis through sequential nn
    print('Sequential reachability analysis \n')
    output_bds_list_seq = []
    solver_time_seq_list = []
    seq_nn_system = SequentialModel(nn_system, 1)
    for i in range(horizon):
        print('step {} \n'.format(i+1))
        seq_nn_system.reset_horizon(i+1)
        bounds_list, solver_time_seq = seq_nn_system.output_inf_bounds_LP(A_input, b_input, c_output, file_name=None)
        output_bds = bounds_list[-1]
        output_bds_list_seq.append(output_bds)
        solver_time_seq_list.append(solver_time_seq)

    pre_act_bds_list_seq = bounds_list[:-1]

    # reachability analysis through iterative methods
    print('Iterative reachability analysis \n')
    output_bds_list_iter = []
    solver_time_iter_list = []
    A_input_iter, b_input_iter = A_input, b_input
    pre_act_bds_list_iter = []
    for i in range(horizon):
        base_nn_system = SequentialModel(nn_system, 1)
        bounds_list, solver_time_iter = base_nn_system.output_inf_bounds_LP(A_input_iter, b_input_iter, c_output, file_name=None)
        pre_act_bds_list_iter = pre_act_bds_list_iter + bounds_list[:-1]
        output_bds = bounds_list[-1]
        output_bds_list_iter.append(output_bds)
        A_input_iter, b_input_iter = output_bds['A'], output_bds['b']
        solver_time_iter_list.append(solver_time_iter)

    print('sequential:', output_bds_list_seq, 'solver time:', solver_time_seq)
    print('iterative:', output_bds_list_iter, 'solver_time:', solver_time_iter)
    # print('solver time comparison: sequentail:', str(sum(solver_time_seq_list)), ' iterative:', str(sum(solver_time_iter_list)))

    plt.figure()
    solver_time_seq_accumulated_iter = [sum(solver_time_iter_list[:i+1]) for i in range(horizon)]
    plt.semilogy(solver_time_seq_list, 'ro-', label = 'sequential')
    plt.semilogy(solver_time_seq_accumulated_iter,'bs-.', label = 'iterative')
    plt.title('solver time comparison')
    plt.xlabel('step')
    plt.ylabel('solver time [sec]')

    seq_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_seq)
    iter_poly_list = ut.bounds_list_to_polyhedron_list(output_bds_list_iter)

    for i in range(horizon):
        plt.figure()
        seq_poly_list[i].plot(fill=False, ec='r', linestyle='-', linewidth=2)
        iter_poly_list[i].plot(fill=False, ec='b', linestyle='-.', linewidth=2)

        domain = Polyhedron(A_input, b_input)
        init_states = ut.unif_sample_from_Polyhedron(domain, 8)
        traj_list = ut.simulate_NN_system(nn_system, init_states, step=i)

        ut.plot_multiple_traj_tensor_to_numpy(traj_list)
        domain.plot(fill=False, ec='k', linestyle='--', linewidth=2)
        plt.title('Reachable set at step {}'.format(i+1))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')

    # compare layerwise bounds
    ut.compare_layerwise_bounds(pre_act_bds_list_iter, pre_act_bds_list_seq)

    plt.figure()
    ut.plot_poly_list(seq_poly_list, fill=False, ec='r', linestyle='-', linewidth=2)
    ut.plot_poly_list(iter_poly_list, fill=False, ec='b', linestyle='-.', linewidth=2)
    ut.plot_multiple_traj_tensor_to_numpy(traj_list)
    domain.plot(fill=False, ec='k', linestyle='--', linewidth=2)
    plt.title('Reachable sets'.format(i + 1))
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.show()
