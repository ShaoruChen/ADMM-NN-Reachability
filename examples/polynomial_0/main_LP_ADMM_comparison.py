
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
from nn_reachability.ADMM import init_sequential_admm_session, run_ADMM, intermediate_bounds_from_ADMM, InitModule, ADMM_Session

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        nn.Linear(n, 900),
        nn.ReLU(),
        nn.Linear(900, 500),
        nn.ReLU(),
        nn.Linear(500, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, n)
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

    nn_system = torch.load('trained_nn_modeL_large_layer.pt')
    horizon = 2

    example = 'LP'
    if example == 'LP':
        x0 = torch.zeros(2)
        nx = 2

        A_input = np.vstack((np.eye(nx), -np.eye(nx)))
        # b_input = np.array([0.0, 0.0, 0.5, 0.5])
        b_input = np.array([-1.65, -1.65, 1.85, 1.85])
        c_output = ut.unif_normal_vecs(nx, n = 4)

        # view the sampled trajectories
        # plt.figure()
        # domain = Polyhedron(A_input, b_input)
        # init_states = ut.unif_sample_from_Polyhedron(domain, 5)
        # traj_list = ut.simulate_NN_system(nn_system, init_states, step=15)
        #
        # ut.plot_multiple_traj_tensor_to_numpy(traj_list)
        # domain.plot(fill=False, ec='r', linestyle='-.', linewidth=2)
        # plt.show()

        seq_nn_system = SequentialModel(nn_system, horizon)
        file_name = 'LP_intermediate_bounds_new'
        bounds_list, solver_time_seq = seq_nn_system.output_inf_bounds_LP(A_input, b_input, c_output, file_name)

        LP_result = {'bounds_list': bounds_list, 'solver_time_seq': solver_time_seq, 'output_bds': bounds_list[-1]}

        torch.save(LP_result, 'LP_result_nn_large_new.pt')

    elif example == 'ADMM':
        # ADMM method
        # fixme: need to test if codes work on GPUs
        alg_options = {'rho': 1.0}
        nn_system.to(device)
        base_nn_model_list = list(nn_system)
        nn_layers_list = base_nn_model_list * horizon

        nx = 2
        x0 = torch.tensor([[-1.75, -1.75]]).to(device)

        epsilon = 0.1
        x0_lb = x0 - epsilon
        x0_ub = x0 + epsilon

        file_name = 'ADMM_intermediate_bounds_large_high_precision_adaptive.pt'
        load_file = 0
        if load_file == 1:
            data = torch.load(file_name)
            pre_act_bds_admm = data
        else:
            pre_act_bds_admm, runtime = intermediate_bounds_from_ADMM(nn_layers_list, x0_lb, x0_ub, alg_options, file_name)
            print('runtime: {}'.format(runtime))
            temp_result = {'pre_adt_bds_admm':pre_act_bds_admm, 'runtime': runtime}
            torch.save(temp_result,'ADMM_pre_act_bounds_result.pt')

        # find the output overapproximation
        c_output = ut.unif_normal_vecs(nx, n=4)
        c_output = torch.from_numpy(c_output).type(torch.float32)
        c_output = -c_output.to(device)

        rho = 1.0

        num_batches = c_output.size(0)
        x_input = x0.repeat(num_batches, 1)
        lb_input = x0_lb.repeat(num_batches, 1)
        ub_input = x0_ub.repeat(num_batches, 1)

        init_module = InitModule(nn_layers_list, x_input, lb_input, ub_input, pre_act_bds_list=None)
        admm_module = init_module.init_ADMM_module()
        admm_sess = ADMM_Session([admm_module], lb_input, ub_input, c_output, rho)

        pre_act_bds = [{'lb': item['lb'].repeat(num_batches, 1), 'ub': item['ub'].repeat(num_batches, 1)} for item in
                       pre_act_bds_admm]
        admm_sess.assign_pre_activation_bounds(pre_act_bds)

        objective, running_time, result, termination_example_id = run_ADMM(admm_sess, eps_abs=1e-5, eps_rel=1e-4,
                                                                           residual_balancing=True, max_iter=7000,
                                                                           record=True, verbose=True)

        ADMM_result = {'pre_act_bds': pre_act_bds_admm, 'pre_act_bds_runtime': runtime,
                       'output_admm_objective': objective, 'c': c_output.to(torch.device('cpu')),
                       'output_admm_runtime': running_time, 'output_admm_result': result}
        torch.save(ADMM_result, 'ADMM_result.pt')

        solver_time_list = ADMM_result['pre_act_bds_runtime']
        solver_time_list.append(ADMM_result['output_admm_runtime'])
        plt.figure()
        plt.plot(solver_time_list, '-o')
        plt.xlabel('layer numer')
        plt.ylabel('solver time [s]')
        print('ADMM total solver time: {}'.format(sum(solver_time_list)))
        plt.show()

        # compare runtime
        LP_result = torch.load('LP_result_nn_large_new.pt')
        solver_time_LP = LP_result['solver_time_seq']

        ADMM_result_low = torch.load('ADMM_result_nn_large_low_precision.pt')
        solver_time_admm_low = ADMM_result_low['pre_act_bds_runtime']
        solver_time_admm_low.append(ADMM_result_low['output_admm_runtime'])

        ADMM_result_high = torch.load('ADMM_result_nn_large.pt')
        solver_time_admm_high = ADMM_result_high['pre_act_bds_runtime']
        solver_time_admm_high.append(ADMM_result_high['output_admm_runtime'])

        plt.figure()
        plt.plot(solver_time_LP,'-o', label = 'Gurobi')
        plt.plot(solver_time_admm_high, '-s', label = 'ADMM high prec.')
        plt.plot(solver_time_admm_low, '-^', label = 'ADMM low prec.')
        plt.xlabel('activation layer numer')
        plt.ylabel('solver time [s]')
        plt.legend()
