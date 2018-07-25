import numpy as np
from matplotlib import pyplot as plt
from plot_result import plot_u_dist
from plot_result import animate_udist_comparison
from run_aposteriori import compute_aposteriori_3d_linear
from run_aposteriori import compute_aposteriori_2d_vdp
from run_aposteriori import compute_aposteriori_3d_lorenz

# plt.style.use('siads')

def validate_closure(config_dict, data_dict, targeted_states, true_closure_to_add, tsnap, case_name, problem_name):
    """
    Validating the closure data in posteriori sense

    - we will set initial condition in a exact sense, then evolve the flow, adding EXACT closure, to make sure things are ok
    - for 1d VBE cases
        - finally we will animate the results in a spatial
    - for other cases:
        - simply draw the component vs time

    :param config_dict:
    :param data_dict:
    :param targeted_states: x_{p+1}, ...., x_{N-1}
    :param true_closure_to_add: delta_{p},.....,delta_{N-2}
    :param tsnap:
    :param case_name:
    :param problem_name:
    :return:
    """

    # set time array: as takens array, which is from t = p to t = N-2.
    # -- note that N -1 is the last time snap
    tsnap_takens = tsnap[config_dict['num_pre_state']:-1]
    tstep = tsnap_takens.size

    # set initial condition since we have the exact states initially
    feature = data_dict['feature']
    # since initial feature, no matter in which form, time parallel or cross time eco or cross time, the last Q is always the x_{p}
    # while the initial condition we need, is x_{p}
    ic_unscaled_unshifted = feature[0,:][-config_dict['reduced_modes']:]
    ic_unscaled_unshifted = ic_unscaled_unshifted.reshape(1,-1) ## reduce the IC from 2D array into 1D array

    print 'debug: sine wave initial condition unscaled unshifted reduced IC= ',ic_unscaled_unshifted

    # discuss for each case
    if problem_name == '3d_linear':
        """
        3D linear case
        """

        ## no closure
        case_style = 'no_closure_' + case_name

        ## given ic as x_{p}, xsnap = x_{p},...,x_{N-2} [assuming last one is x_{N-1}]
        xsnap = compute_aposteriori_3d_linear(config_dict, ic_unscaled_unshifted, true_closure_to_add, tstep, case_style)

        # component vs time
        plt.figure()
        plt.plot(tsnap_takens, targeted_states,'-')
        plt.plot(tsnap_takens, xsnap,'--')
        plt.xlabel('time')
        plt.ylabel('component value of $x$') # note scaled
        plt.savefig(config_dict['image_path'] + 'xsnap_' + case_style + '.png', bbox_inches='tight')
        plt.close()

        ## true target closure
        case_style = 'true_closure_' + case_name
        xsnap = compute_aposteriori_3d_linear(config_dict, ic_unscaled_unshifted, true_closure_to_add, tstep, case_style)

        # component vs time
        plt.figure()
        plt.plot(tsnap_takens, targeted_states,'-')
        plt.plot(tsnap_takens, xsnap,'--')
        plt.xlabel('time')
        plt.ylabel('component value of $x$') # note scaled
        plt.savefig(config_dict['image_path'] + 'xsnap_' + case_style + '.png', bbox_inches='tight')
        plt.close()

    elif problem_name == '2d_vdp':

        ## no closure
        case_style = 'no_closure_' + case_name
        xsnap = compute_aposteriori_2d_vdp(config_dict, ic_unscaled_unshifted, true_closure_to_add, tstep, case_style)

        # component vs time
        plt.figure()
        plt.plot(tsnap_takens, targeted_states,'-')
        plt.plot(tsnap_takens, xsnap,'--')
        plt.xlabel('time')
        plt.ylabel('component value of $x$') # note scaled
        plt.savefig(config_dict['image_path'] + 'xsnap_' + case_style + '.png', bbox_inches='tight')
        plt.close()

        ## true target closure
        case_style = 'true_closure_' + case_name
        xsnap = compute_aposteriori_2d_vdp(config_dict, ic_unscaled_unshifted, true_closure_to_add, tstep, case_style)

        # component vs time
        plt.figure()
        plt.plot(tsnap_takens, targeted_states,'-')
        plt.plot(tsnap_takens, xsnap,'--')
        plt.xlabel('time')
        plt.ylabel('component value of $x$') # note scaled
        plt.savefig(config_dict['image_path'] + 'xsnap_' + case_style + '.png', bbox_inches='tight')
        plt.close()

    elif problem_name == 'lorenz':

        ## no closure
        case_style = 'no_closure_' + case_name
        xsnap = compute_aposteriori_3d_lorenz(config_dict, ic_unscaled_unshifted, true_closure_to_add, tstep, case_style)

        # component vs time
        plt.figure()
        plt.plot(tsnap_takens, targeted_states,'-')
        plt.plot(tsnap_takens, xsnap,'--')
        plt.xlabel('time')
        plt.ylabel('component value of $x$') # note scaled
        plt.savefig(config_dict['image_path'] + 'xsnap_' + case_style + '.png', bbox_inches='tight')
        plt.close()

        ## true target closure
        case_style = 'true_closure_' + case_name
        xsnap = compute_aposteriori_3d_lorenz(config_dict, ic_unscaled_unshifted, true_closure_to_add, tstep, case_style)

        # component vs time
        plt.figure()
        plt.plot(tsnap_takens, targeted_states,'-')
        plt.plot(tsnap_takens, xsnap,'--')
        plt.xlabel('time')
        plt.ylabel('component value of $x$') # note scaled
        plt.savefig(config_dict['image_path'] + 'xsnap_' + case_style + '.png', bbox_inches='tight')
        plt.close()



