
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
from nn_reachability.nn_models import iterative_output_Lp_bounds_LiRPA, output_Lp_bounds_LiRPA

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

''' FitzHugh-Nagumo Neuron Model from 
Semidefinite Approximations of Reachable Sets for Discrete-time Polynomial Systems, Morgan et al., 2019'''


def system_dynamics(x):
    y_1 = x[0] + 0.2*(x[0] - x[0]**3/3 - x[1] + 0.875)
    y_2 = x[1] + 0.2*(0.08*(x[0] + 0.7 - 0.8*x[1]))
    y = np.array([y_1, y_2])
    return y


def nn_dynamics(n=2, m = 100):
    # neural network dynamics with randomly generated weights
    model = nn.Sequential(
        nn.Linear(n, m),
        nn.ReLU(),
        nn.Linear(m, m),
        nn.ReLU(),
        nn.Linear(m, n)
    )
    return model

if __name__ == '__main__':
    is_train = True
    nn_width = 50

    nn_file_name = 'nn_model_2_layer_' + str(nn_width) + '_neuron.pt'

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

        nn_structure = nn_dynamics(2, nn_width)
        nn_model = train_nn_torch(train_loader, nn_structure, num_epochs=100, l1=None,
                                  pred_step=0, lr=1e-4, decay_rate=1.0, clr=None, path='torch_nn_model')

        torch.save(nn_model, nn_file_name)


    '''
    finite step reachability analysis
    '''
    # torch.set_grad_enabled(False)

    nn_system = torch.load(nn_file_name)

    nx = 2
    # x0 = torch.tensor([[1.0, 1.0]]).to(device)
    # x0 = torch.tensor([[0.0,0.0]]).to(device)

    x0 = torch.tensor([[0.0, 0.0]]).to(device)

    epsilon = 1.0
    x0_lb = x0 - epsilon
    x0_ub = x0 + epsilon

    horizon = 5

    skip_LP = True

    #######################################
    # LP method
    ######################################
    if not skip_LP:
        nx = 2

        A_input = np.vstack((np.eye(nx), -np.eye(nx)))

        input_lb = x0_lb.to(torch.device('cpu')).numpy()
        input_ub = x0_ub.to(torch.device('cpu')).numpy()

        b_input = np.concatenate((input_ub, -input_lb)).flatten()
        c_output = ut.unif_normal_vecs(nx, n = 4)

        # view the sampled trajectories
        # plt.figure()
        # domain = Polyhedron(A_input, b_input)
        # init_states = ut.unif_sample_from_Polyhedron(domain, 8)
        # traj_list = ut.simulate_NN_system(nn_system, init_states, step=100)
        #
        # ut.plot_multiple_traj_tensor_to_numpy(traj_list)
        # domain.plot(fill=False, ec='r', linestyle='-.', linewidth=2)
        # plt.show()

        seq_nn_system = SequentialModel(nn_system, horizon)
        LP_suffix = '_horizon_' + str(horizon) + '_radius_' + str(epsilon) + '.pt'
        file_name = 'LP_intermediate_bounds' + LP_suffix
        bounds_list, solver_time_seq = seq_nn_system.output_inf_bounds_LP(A_input, b_input, c_output, file_name)

        LP_result = {'bounds_list': bounds_list, 'solver_time_seq': solver_time_seq, 'output_bds': bounds_list[-1]}

        LP_result_file = 'LP_result' + '_width_' + str(nn_width)  + LP_suffix
        torch.save(LP_result, LP_result_file)
        print('LP solver time {}'.format(sum(solver_time_seq)))

    #######################################################
    # ADMM method
    ######################################################
    # fixme: need to test if codes work on GPUs

    alg_options = {'rho': 0.1, 'eps_abs': 1e-4, 'eps_rel': 1e-3, 'residual_balancing': False, 'max_iter': 20000,
                   'record': False, 'verbose': True, 'alpha': 1.6}

    nn_system.to(device)
    base_nn_model_list = list(nn_system)
    nn_layers_list = base_nn_model_list * horizon

    ADMM_suffix = '_horizon_' + str(horizon) + '_radius_' + str(epsilon) + '_eps_abs_' + str(alg_options['eps_abs']) + '_new.pt'

    file_name = 'ADMM_intermediate_bounds' + '_width_' + str(nn_width) + ADMM_suffix
    load_file = 0

    if load_file == 1:
        data = torch.load(file_name)
        pre_act_bds_admm = data

        ADMM_pre_act_bds_result = torch.load('ADMM_pre_act_bounds_result' + ADMM_suffix)
        runtime = ADMM_pre_act_bds_result['runtime']
    else:
        pre_act_bds_admm, runtime = intermediate_bounds_from_ADMM(nn_layers_list, x0_lb, x0_ub, alg_options, file_name)
        print('runtime: {}'.format(runtime))
        temp_result = {'pre_adt_bds_admm':pre_act_bds_admm, 'runtime': runtime, 'alg_options': alg_options}
        torch.save(temp_result,'ADMM_pre_act_bounds_result' + ADMM_suffix)

    # find the output overapproximation
    c_output = ut.unif_normal_vecs(nx, n=4)
    c_output = torch.from_numpy(c_output).type(torch.float32)
    c_output = -c_output.to(device)

    rho = alg_options['rho']

    num_batches = c_output.size(0)
    x_input = x0.repeat(num_batches, 1).to(device)
    lb_input = x0_lb.repeat(num_batches, 1).to(device)
    ub_input = x0_ub.repeat(num_batches, 1).to(device)

    init_module = InitModule(nn_layers_list, x_input, lb_input, ub_input, pre_act_bds_list=None)
    admm_module = init_module.init_ADMM_module()
    admm_sess = ADMM_Session([admm_module], lb_input, ub_input, c_output, rho)

    pre_act_bds = [{'lb': item['lb'].repeat(num_batches, 1), 'ub': item['ub'].repeat(num_batches, 1)} for item in
                   pre_act_bds_admm]
    admm_sess.assign_pre_activation_bounds(pre_act_bds)

    objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)

    ADMM_result = {'pre_act_bds': pre_act_bds_admm, 'pre_act_bds_runtime': runtime,
                   'output_admm_objective': objective.to(torch.device('cpu')), 'c': c_output.to(torch.device('cpu')),
                   'output_admm_runtime': running_time, 'output_admm_result': result,
                   'alg_options': alg_options,
                   'x0_lb': x0_lb.to(torch.device('cpu')), 'x0_ub': x0_ub.to(torch.device('cpu')), 'horizon': horizon}

    ADMM_result_file = 'ADMM_result' + '_width_' + str(nn_width) + ADMM_suffix
    torch.save(ADMM_result, ADMM_result_file)


    # plot solver time
    solver_time_list = ADMM_result['pre_act_bds_runtime']
    solver_time_list.append(ADMM_result['output_admm_runtime'])
    plt.figure()
    plt.plot(solver_time_list, '-o')
    plt.xlabel('layer numer')
    plt.ylabel('solver time [s]')
    plt.title('ADMM solver time')
    print('ADMM total solver time: {}'.format(sum(solver_time_list)))
    plt.show()

    # # compare runtime
    # LP_result = torch.load('LP_result_nn_large_new.pt')
    # solver_time_LP = LP_result['solver_time_seq']
    #
    # ADMM_result_low = torch.load('ADMM_result_nn_large_low_precision.pt')
    # solver_time_admm_low = ADMM_result_low['pre_act_bds_runtime']
    # solver_time_admm_low.append(ADMM_result_low['output_admm_runtime'])
    #
    # ADMM_result_high = torch.load('ADMM_result_nn_large.pt')
    # solver_time_admm_high = ADMM_result_high['pre_act_bds_runtime']
    # solver_time_admm_high.append(ADMM_result_high['output_admm_runtime'])
    #
    # plt.figure()
    # plt.plot(solver_time_LP,'-o', label = 'Gurobi')
    # plt.plot(solver_time_admm_high, '-s', label = 'ADMM high prec.')
    # plt.plot(solver_time_admm_low, '-^', label = 'ADMM low prec.')
    # plt.xlabel('activation layer numer')
    # plt.ylabel('solver time [s]')
    # plt.legend()


    # load results
    width = 100
    eps = 1e-5
    admm_file_high = 'ADMM_result_width_' + str(width) + '_horizon_5_radius_1.0_eps_abs_' + str(eps) +'_new.pt'
    admm_result_high = torch.load(admm_file_high)
    solver_time_high = sum(admm_result_high['pre_act_bds_runtime']) + admm_result_high['output_admm_runtime']
    output_bounds_high = admm_result_high['output_admm_objective']

    eps = 1e-4
    admm_file_low = 'ADMM_result_width_' + str(width) + '_horizon_5_radius_1.0_eps_abs_' + str(eps) + '_new.pt'
    admm_result_low = torch.load(admm_file_low)
    solver_time_low = sum(admm_result_low['pre_act_bds_runtime']) + admm_result_low['output_admm_runtime']
    output_bounds_low = admm_result_low['output_admm_objective']

    LP_result = torch.load('LP_result_width_' + str(width) + '_horizon_5_radius_1.0.pt')
    solver_time_gurobi = sum(LP_result['solver_time_seq'])
    output_bounds_LP = LP_result['output_bds']

    print('solver time: {} {} {}'.format(solver_time_gurobi, solver_time_high, solver_time_low))

    # compare with fast LP bounds
    nn_system.to(device)
    nn_model = SequentialModel(nn_system, horizon)
    method = 'CROWN-optimized'
    output_lb, output_ub = output_Lp_bounds_LiRPA(nn_model, x0_lb, x0_ub, method=method)

