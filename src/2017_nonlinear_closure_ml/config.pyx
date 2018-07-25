# import python modules
import os
import numpy as np

def compute_uniform_time_and_dt(config_dict):
    """
    compute the dt and equally spaced time array

    :param config_dict:
    :return:
    """

    config_dict['aposteriori_config']['t'] = np.linspace(0, config_dict['aposteriori_config']['tot'],
                                                         config_dict['aposteriori_config']['tstep'],
                                                         endpoint=False)
    config_dict['aposteriori_config']['dt'] = config_dict['aposteriori_config']['t'][1] - \
                                              config_dict['aposteriori_config']['t'][0]
    return config_dict

def set_cfg(problem_name):
    """
    set up simulation configuration: polynomial

    :return: a dictionary contains all configuration
    """

    # init config
    config_dict = {}

    # config for different cases
    if problem_name == '3d_linear':
        """
        3D linear case
        """

        # resolved part
        config_dict['A11'] = np.array([[0]])

        # data information
        config_dict['aposteriori_config'] = {'tot': 40, 'tstep': 4000}

        # compute time step and dt
        config_dict = compute_uniform_time_and_dt(config_dict)

        ## parameters for modelling closure
        config_dict['reduced_modes'] = 1
        config_dict['num_full_modes'] = 3
        config_dict['train_total_ratio'] = 0.1

        ## closure file path
        config_dict['file_path'] = "../../data/3d_linear_system/data/"
        config_dict['file_name'] = "closure_3d_linear_ntsnap_4000_tot_40.npz"

        # feature scaling
        config_dict['scaling'] = False

        config_dict['verbose'] = True

        ## particularly for sindy selected l1
        # using hyperparameter search
        config_dict['l1'] = [1e-12] # p=1
        # config_dict['l1'] = [10**(-1.75)] # p=0

        # image path
        config_dict['image_path'] = "../../data/3d_linear_system/image/"

    elif problem_name == '2d_vdp':
        """
        2D vdp system
        """
        config_dict['A11'] = np.array([[0]])

        config_dict['aposteriori_config'] = {
            'tot': 60,  # match
            'tstep': 6000,  # match
        }

        #  compute the time
        config_dict = compute_uniform_time_and_dt(config_dict)

        ## parameters for modelling closure
        config_dict['reduced_modes'] = 1
        config_dict['num_full_modes'] = 2
        config_dict['train_total_ratio'] = 0.3

        ## closure file path
        config_dict['file_path'] = "../../data/2d_vdp/data/"
        config_dict['file_name'] = "closure_2d_vdp_ntsnap_6000_tot_60.npz"

        # feature scaling
        # for 2d vdp, sure, we use sindy... so scaling is turned to be false, so that feature is sparse and everything is nice
        config_dict['scaling'] = False

        config_dict['verbose'] = True

        ## particularly for sindy selected l1
        config_dict['l1'] = [1e-7]*1

        # image path
        config_dict['image_path'] = "../../data/2d_vdp/image/"

    elif 'lorenz' in problem_name:
        """
        Lorenz system with first as the only observable
        """

        if 'chaos' in problem_name:
            lorenz_case = 'chaos'
        # if resolved part is instable an chaotic,
        # it is impossible to close... instablility
        else:
            lorenz_case = 'equ'


        if lorenz_case == 'chaos':
            config_dict['aposteriori_config'] = {
                'tot': 400,  # match
                'tstep': 40000 # match
            }
            rho = 35

        elif lorenz_case == 'equ':
            config_dict['aposteriori_config'] = {
                'tot': 20,  # match
                'tstep': 8000# match
            }
            rho = 15

        sigma = 10
        config_dict['A11'] = np.array([[-sigma]])

        # compute time step and dt
        config_dict = compute_uniform_time_and_dt(config_dict)

        ## parameters for modelling closure
        config_dict['reduced_modes'] = 1
        config_dict['num_full_modes'] = 3
        config_dict['train_total_ratio'] = 0.5

        ## closure file path
        config_dict['file_path'] = "../../data/lorenz/data/"

        if lorenz_case == 'chaos':
            config_dict['file_name'] = "closure_3d_lorenz_ntsnap_" + 'case_' + lorenz_case +  \
            "_" + str(config_dict['aposteriori_config']['tstep']) + "_tot_" + \
            str(config_dict['aposteriori_config']['tot']) + '_reduced_modes_' + \
                                       str(config_dict['reduced_modes']) + ".npz"
        elif lorenz_case == 'equ':
            config_dict['file_name'] = "closure_3d_lorenz_ntsnap_" + 'case_' + lorenz_case +  \
                                       "_" + str(config_dict['aposteriori_config']['tstep']) + "_tot_" + \
                                        str(config_dict['aposteriori_config']['tot']) + '_reduced_modes_' + \
                                       str(config_dict['reduced_modes']) + ".npz"

        # feature scaling
        # for lorenz, sure, we use sindy... so scaling is turned to be false,
        # so that feature is sparse and everything is nice
        config_dict['scaling'] = True

        config_dict['verbose'] = True

        ## particularly for sindy selected l1
        config_dict['l1'] = [1e-14]*1

        # image path
        config_dict['image_path'] = "../../data/lorenz/image/"

    return config_dict


def set_cfg_poly(problem_name, num_pre_state, fix_order, order):
    """
    set configuration for polynomial models

    :param problem_name:
    :param num_pre_state:
    :param fix_order:
    :param order:
    :return:
    """

    # configuration from base dictionary
    config_dict = set_cfg(problem_name)
    config_dict['num_pre_state'] = num_pre_state  # tune

    # ---> sindy
    config_dict['ML_type'] = 'sindy'

    # we automatically forcing time_parallel_delta_x_pair in ALL sindy simulation

    config_dict['ML_method'] = {'name': config_dict['ML_type'] + '_time_parallel_delta_x_pair',
                                # 'sindy_cross_time', # sindy_time_parallel_delta_x_pair
                                'fix_order': fix_order,
                                'order': order,
                                'l1': config_dict['l1'], # we assume closure is sparse in polynomial
                                'poly_type': 'monomial',  # monomial, legendre
                                'turn_on_mylibrary_poly_feature': True,
                                'solver_type': 'lasso'  # cvx_gradient (support sparsity), direct (does not support sparsity) # lasso
                                }
    # plot
    config_dict['markersize'] = 2

    ## generate configuration file folder:
    folder_case = 'nps_' + str(config_dict['num_pre_state']) + '_ML_' + config_dict['ML_method']['name'] + \
                  '_' + config_dict['ML_method']['fix_order'] + '_order_' + str(config_dict['ML_method']['order']) + '/'
    config_dict['image_path'] = config_dict['image_path'] + folder_case
    ensure_dir(config_dict['image_path'])

    return config_dict


def set_cfg_ann(problem_name, num_pre_state, multi_time_type, number_hidden_units):
    """
    set configuration for ANN model

    :param problem_name:
    :param num_pre_state:
    :param multi_time_type:
    :param number_hidden_units:
    :return:
    """

    # configuration from base dictionary
    config_dict = set_cfg(problem_name)
    config_dict['num_pre_state'] = num_pre_state  # tune

    # ---> ann
    # --> set multi-time style
    config_dict['ML_type'] = 'ann'

    if 'time_parallel' in multi_time_type:
        type_multi_time = '_time_parallel_delta_x_pair'

    elif 'cross_time_eco' in multi_time_type:
        type_multi_time = '_cross_time_eco'

    elif 'cross_time' in multi_time_type:
        type_multi_time = '_cross_time'

    # set configuration for ann
    config_dict['ML_method'] = {'name': config_dict['ML_type'] + type_multi_time,
                                # '_cross_time', # ann_time_parallel_delta_x_pair
                                'number_hidden_units': number_hidden_units,
                                'activationFun': 'tanh',
                                'mini_batch': 64,
                                'npoch': 20000, # for vbe 2000
                                'weight_regularizer':'l2',
                                    'l1_regu_coef':1e-4,
                                    'l2_regu_coef':1e-6,
                                'lr': 1e-4,
                                'decay': 1e-5,
                                'training_validation_split_ratio': 0.95
                                }

    # plot
    config_dict['markersize'] = 2

    ## generate configuration file folder:
    folder_case = 'nps_' + str(config_dict['num_pre_state']) + '_ML_' + \
                  config_dict['ML_method']['name'] + '_nh_' + str(number_hidden_units) + '/'
    config_dict['image_path'] = config_dict['image_path'] + folder_case
    ensure_dir(config_dict['image_path'])

    return config_dict


def ensure_dir(file_path):
    """
    make dir if there is not one

    :param file_path:
    :return:
    """

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
