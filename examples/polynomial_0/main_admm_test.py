
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("always")
import numpy as np
import nn_reachability.utilities as ut

from pympc.geometry.polyhedron import Polyhedron
import matplotlib.pyplot as plt
from nn_reachability.nn_models import preactivation_bounds_of_sequential_nn_model_LiRPA
from nn_reachability.ADMM import init_sequential_admm_session, run_ADMM, intermediate_bounds_from_ADMM
from nn_reachability.ADMM import InitModule, run_ADMM, ADMM_Session
import matplotlib.pyplot as plt

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device= torch.device('cpu')

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    nn_system = torch.load('trained_nn_model_4_layer.pt')
    nn_system.to(device)
    option = 1

    if option == 1:
        # plug in LiRPA pre-activation bounds
        nx = 2
        c_output = ut.unif_normal_vecs(nx, n = 4)

        x0 = torch.tensor([[-1.75, -1.75], [2.0, -1.2]]).to(device)
        # x0 = torch.tensor([[-1.75, -1.75]])

        epsilon = 0.1
        x0_lb = x0 - epsilon
        x0_ub = x0 + epsilon

        c = torch.tensor([[1.0, 0], [1.0, 0.0]])
        # c = torch.tensor([[0.1, 0.1]])

        rho = 0.5

        base_nn_model_list = list(nn_system)
        horizon = 2
        admm_sess = init_sequential_admm_session(base_nn_model_list, horizon, x0, x0_lb, x0_ub, c, rho)
        pre_act_bds, num_act_layers, _ = preactivation_bounds_of_sequential_nn_model_LiRPA(nn_system, horizon, x0_lb, x0_ub, method = 'backward')

        admm_sess.assign_pre_activation_bounds(pre_act_bds)

        alg_options = {'rho': 0.1, 'eps_abs': 1e-5, 'eps_rel': 1e-4, 'residual_balancing': True, 'max_iter': 20000,
                       'record': True, 'verbose': True, 'alpha': 1.6, 'adaptive_rho': False, 'rho_update_freq': 2, 'alpha':1.6,  'view_id': 1}

        objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)

        print('running time {}'.format(running_time))
        view_id = 1
        plt.figure()
        plt.plot([entry[view_id].item() for entry in result['obj_list']])
        plt.title('ADMM objective')

        plt.figure()
        plt.semilogy([entry[view_id].item() for entry in result['rp_list']])
        plt.semilogy([entry[view_id].item() for entry in result['p_tol_list']])
        plt.title('ADMM primal residual')

        plt.figure()
        plt.semilogy([entry[view_id].item() for entry in result['rd_list']])
        plt.semilogy([entry[view_id].item() for entry in result['d_tol_list']])
        plt.title('ADMM dual residual')

        plt.figure()
        plt.plot([entry[view_id].item() for entry in result['rho_list']])
        plt.title('ADMM rho')
        plt.show()

    elif option == 2:
        # compute intermediate bounds from ADMM
        nx = 2

        x0 = torch.tensor([[-1.75, -1.75]]).to(device)

        epsilon = 0.1
        x0_lb = x0 - epsilon
        x0_ub = x0 + epsilon

        base_nn_model_list = list(nn_system)
        horizon = 2
        nn_layers_list = base_nn_model_list*horizon

        alg_options = {'rho': 1.0}

        file_name = 'temp_ADMM_intermediate_bds.pt'
        load_file = 0
        if load_file == 1:
            data = torch.load(file_name)
            pre_act_bds_admm = data
        else:
            pre_act_bds_admm, runtime = intermediate_bounds_from_ADMM(nn_layers_list, x0_lb, x0_ub, alg_options, file_name)

        # find the output overapproximation
        c_output = ut.unif_normal_vecs(nx, n=4)
        c_output = torch.from_numpy(c_output).type(torch.float32)
        c_output = c_output.to(device)
        rho = 1.0

        num_batches = c_output.size(0)
        x_input  = x0.repeat(num_batches, 1)
        lb_input = x0_lb.repeat(num_batches, 1)
        ub_input = x0_ub.repeat(num_batches, 1)

        init_module = InitModule(nn_layers_list, x_input, lb_input, ub_input, pre_act_bds_list=None)
        admm_module = init_module.init_ADMM_module()
        admm_sess = ADMM_Session([admm_module], lb_input, ub_input, c_output, rho)

        pre_act_bds = [{'lb': item['lb'].repeat(num_batches, 1), 'ub': item['ub'].repeat(num_batches, 1)} for item in pre_act_bds_admm]
        admm_sess.assign_pre_activation_bounds(pre_act_bds)

        alg_options = {'rho': 0.1, 'eps_abs': 1e-5, 'eps_rel': 1e-4, 'residual_balancing': True, 'max_iter': 20000,
                       'record': True, 'verbose': True, 'alpha': 1.6, 'adaptive_rho': True, 'rho_update_freq': 2, 'view_id': 1}

        objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)

        # final_result = {'ADMM_result':result, 'obj': objective, 'running_time': running_time, 'runtime': runtime, 'intermediate_bds': pre_act_bds_admm}
        # torch.save(final_result, 'layer_4_admm_layerwise_results.pt')

        view_id = 1
        plt.figure()
        plt.plot([entry[view_id].item() for entry in result['obj_list']])
        plt.title('ADMM objective id {}'.format(view_id))


        print('')