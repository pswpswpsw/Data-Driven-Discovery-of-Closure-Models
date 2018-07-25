"""
Operator inference framework for closure modelling:

- model: sindy with lasso
- model: ann model with two hidden layers

Note:
    the time difference scheme used is simply first order forward scheme.

author: Shaowu Pan
date: 12/11/2017

## command line arugment examples
#  different testing cases
   - 2D VDP:

    - python cython_main.py 2d_vdp sindy 0 tdf 3

   - 3d linear system:

    - python cython_main.py 3d_linear sindy 1 tdf 1

   - lorenz:
       - python cython_main.py lorenz-equ ann_cross_time 1 18 ## equ
       - python cython_main.py lorenz-chaos ann_cross_time 1 18 ## chaotic


"""

## import python ordinary modules
from pprint import pprint

## import utilization of closure
from config import set_cfg_poly
from config import set_cfg_ann
from plot_result import plot_apriori_standard
from plot_result import plot_aposteriori_standard_3d_linear
from plot_result import plot_aposteriori_standard_2d_vdp
from plot_result import plot_aposteriori_standard_3d_lorenz
from plot_result import plot_time_vs_standard_data
from plot_result import viz_2d_feature
from prepare_data import prepare_data_standard
from training import machine_learning
from validating_closure_data_aposteriori import validate_closure



def main(argv):
    """
    main function to run operator inference framework

    :param argv: argument variable for selecting which model and what model parameters to run
    :return:
    """
    # set case
    problem_name = argv[1]
    # set ML type
    ml_type = argv[2]
    ##############################
    ##### 1. set configuration ###
    ##############################
    if ml_type == 'sindy':
        # ML configuration: polynomial
        # 1. define how many previous states
        num_pre_state =int(argv[3])
        # 2. define what is the polynomial style
        fix_order = argv[4] # 'tdf' or 'edf'
        # define the order of polynomial
        order = int(argv[5])
        # set for Polynomial
        config_dict = set_cfg_poly(problem_name    = problem_name,
                                   num_pre_state   = num_pre_state,
                                   fix_order       = fix_order,
                                   order           = order)
    elif 'ann' in ml_type :
        # 1. determine number of previous steps
        num_pre_state = int(argv[3])
        # 2. determine number of hidden units. default we use two layers
        number_hidden_units = int(argv[4])
        # 3. set for ANN configuration
        config_dict = set_cfg_ann(problem_name       = problem_name,
                                  num_pre_state      = num_pre_state,
                                  multi_time_type    = ml_type,
                                  number_hidden_units= number_hidden_units)

    ##############################
    ######## summary #############
    ##############################
    print '============================'
    print 'Operator inference of closure'
    print ''
    print 'Summary'
    print 'Name of the problem: ', problem_name
    print 'ML type = ',ml_type
    print 'number of previous steps (time delay steps) = ', num_pre_state

    print ''
    print 'Config of current OI framework'
    pprint(config_dict)

    #############################################
    ##### 2. data prepration on raw sequences ###
    #############################################
    if problem_name == '3d_linear':
        ##################
        # linear case
        train_data_dict, test_data_dict, targeted_states, targeted_closure, true_closure_to_add = prepare_data_standard(config_dict)

        # plot training and testing data vs time
        plot_time_vs_standard_data(train_data_dict, config_dict, 'train_data')
        plot_time_vs_standard_data(train_data_dict, config_dict, 'test_data')

        # plot train-test feature comparison
        # viz_2d_feature(train_data_dict, test_data_dict, config_dict)

    elif problem_name == '2d_vdp':
        ##################
        #  2d_vdp case
        train_data_dict, test_data_dict, targeted_states, targeted_closure, true_closure_to_add = prepare_data_standard(config_dict)

        # plot training and testing data vs time
        plot_time_vs_standard_data(train_data_dict, config_dict, 'train_data')
        plot_time_vs_standard_data(train_data_dict, config_dict, 'test_data')

        # plot train-test feature comparison
        # viz_2d_feature(train_data_dict, test_data_dict, config_dict)

    elif 'lorenz' in problem_name:
        ##################
        # lorenz case
        train_data_dict, test_data_dict, targeted_states, targeted_closure, true_closure_to_add = prepare_data_standard(config_dict)

        # plot training and testing data vs time
        plot_time_vs_standard_data(train_data_dict, config_dict, 'train_data')
        plot_time_vs_standard_data(train_data_dict, config_dict, 'test_data')

        # plot train-test feature comparison
        # viz_2d_feature(train_data_dict, test_data_dict, config_dict)

    # ML training to obtain model  
    model = machine_learning(config_dict, train_data_dict)

    if problem_name == '3d_linear':
        ##################
        # 3d linear case
        # postprocessing: apriori testing
        plot_apriori_standard(model, train_data_dict, config_dict, 'train_apriori')
        plot_apriori_standard(model, test_data_dict, config_dict, 'test_apriori')

        ##################
        # postprocessing: a posteriori validation on training and testing data
        plot_aposteriori_standard_3d_linear(model, train_data_dict, config_dict, 'train_aposteriori')
        plot_aposteriori_standard_3d_linear(model, test_data_dict, config_dict, 'test_aposteriori')

    elif problem_name == '2d_vdp':
        ##################
        #  vdp case
        # postprocessing: apriori testing
        plot_apriori_standard(model, train_data_dict, config_dict, 'train_apriori')
        plot_apriori_standard(model, test_data_dict, config_dict, 'test_apriori')

        ##################
        # postprocessing: a posteriori validation on training and testing data
        plot_aposteriori_standard_2d_vdp(model, train_data_dict, config_dict, 'train_aposteriori')
        plot_aposteriori_standard_2d_vdp(model, test_data_dict, config_dict, 'test_aposteriori')

    elif 'lorenz' in problem_name:
        ##################
        # lorenz case
                ##################
        #  vdp case
        # postprocessing: apriori testing
        plot_apriori_standard(model, train_data_dict, config_dict, 'train_apriori')
        plot_apriori_standard(model, test_data_dict, config_dict, 'test_apriori')

        ##################
        # postprocessing: a posteriori validation on training and testing data
        plot_aposteriori_standard_3d_lorenz(model, train_data_dict, config_dict, 'train_aposteriori')
        plot_aposteriori_standard_3d_lorenz(model, test_data_dict, config_dict, 'test_aposteriori')


if __name__=='__main__':
    """
    testing code with 2d vdp
    """

    argv = ['2d_vdp', 'sindy','0', 'tdf', '3']
    main(argv)
