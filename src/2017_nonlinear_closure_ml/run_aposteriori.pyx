import numpy as np
import scipy.sparse.linalg

def compute_aposteriori_3d_lorenz(config_dict, ic, targeted_closure, tstep, type_closure):
    """
    a posteriori run on 3d lorenz

    :param config_dict:
    :param ic:
    :param targeted_closure:
    :param tstep:
    :param type_closure:
    :return:
    """
    # simulation parameters
    A11 = config_dict['A11']
    dt = config_dict['aposteriori_config']['dt']

    # initial condition
    x = ic
    shape_data = (tstep, config_dict['reduced_modes'])
    xsnap = np.zeros(shape_data, dtype=float)

    if 'ml_closure' in type_closure:
        ## get \delta^{0} in a full complex sense
        closure_pred = targeted_closure['takens_feature_generator'].get_current_delta()
        # print 'ml first closure_pred = ', closure_pred

    # iteratively evolving model and closure
    for index in xrange(tstep):
        # record x
        xsnap[index, :] = x

        # t = (index+1)*dt

        # ensure x a row vector
        # x = x.reshape(1,-1)

        # compute no closure update
        x = x + dt*np.dot(x, A11.transpose())

        # compute closure updating
        if 'true_closure' in type_closure:
        # simply validating true closure
            x = x + dt*targeted_closure[index, :]
            # debug
            # if index == 0:
            #    print 'true first closure pred = ', closure[index, :]

        elif 'ml_closure' in type_closure:
        # a posteriori validation with ML closure

            # updating state with closure
            x = x + dt*closure_pred

            # obtain d\delta^{n}/dt in reduced imaginary sense
            current_takens_feature = targeted_closure['takens_feature_generator'].get_takens_feature()
            closure_increment = targeted_closure['model'].predict(current_takens_feature)
            closure_increment = closure_increment.flatten()

            # evolving delta: get \delta^{n+1} in full
            closure_pred = closure_pred + dt*closure_increment

            # update takens_feature with x^{n+1} and \delta^{n+1}
            targeted_closure['takens_feature_generator'].update_state_closure(closure_pred, x)

    return xsnap


def compute_aposteriori_2d_vdp(config_dict, ic, targeted_closure, tstep, type_closure):
    """
    a posteriori run on 2d vdp

    :param config_dict:
    :param ic:
    :param targeted_closure:
    :param tstep:
    :param type_closure:
    :return:
    """
    # simulation parameters
    A11 = config_dict['A11']
    dt = config_dict['aposteriori_config']['dt']

    # initial condition
    x = ic
    shape_data = (tstep, config_dict['reduced_modes'])
    xsnap = np.zeros(shape_data, dtype=float)

    if 'ml_closure' in type_closure:
        ## get \delta^{0} in a full complex sense
        closure_pred = targeted_closure['takens_feature_generator'].get_current_delta()
        # print 'ml first closure_pred = ', closure_pred

    # iteratively evolving model and closure
    for index in xrange(tstep):
        # record x
        xsnap[index, :] = x

        # t = (index+1)*dt

        # compute no closure update
        x = np.dot(scipy.sparse.linalg.expm(dt*A11),x)

        # compute closure updating
        if 'true_closure' in type_closure:
        # simply validating true closure
            x = x + dt*targeted_closure[index, :]
            # debug
            # if index == 0:
            #    print 'true first closure pred = ', closure[index, :]

        elif 'ml_closure' in type_closure:
        # a posteriori validation with ML closure

            # updating state with closure
            x = x + dt*closure_pred

            # obtain d\delta^{n}/dt in reduced imaginary sense
            current_takens_feature = targeted_closure['takens_feature_generator'].get_takens_feature()
            closure_increment = targeted_closure['model'].predict(current_takens_feature)
            closure_increment = closure_increment.flatten()

            # evolving delta: get \delta^{n+1} in full
            closure_pred = closure_pred + dt*closure_increment

            # update takens_feature with x^{n+1} and \delta^{n+1}
            targeted_closure['takens_feature_generator'].update_state_closure(closure_pred, x)

    return xsnap

def compute_aposteriori_3d_linear(config_dict, ic, targeted_closure, tstep, type_closure):
    """
    a posteriori run the 3d linear case

        A = | 0     -1  -1 |
            | 0.5 -1.1 1.5 |
            | 1     -3 0.5 |

    x0 = [3 0 0]

    :param config_dict:
    :param ic:
    :param targeted_closure:
    :param tstep:
    :param type_closure:
    :return: xsnap
    """

    # simulation parameters
    A11 = config_dict['A11']
    dt = config_dict['aposteriori_config']['dt']

    # initial condition
    x = ic
    shape_data = (tstep, config_dict['reduced_modes'])
    xsnap = np.zeros(shape_data, dtype=float)

    if 'ml_closure' in type_closure:
        ## get \delta^{0} in a full complex sense
        closure_pred = targeted_closure['takens_feature_generator'].get_current_delta()
        # print 'ml first closure_pred = ', closure_pred

    # iteratively evolving model and closure
    for index in xrange(tstep):
        # record x
        xsnap[index, :] = x

        # t = (index+1)*dt

        # compute no closure update
        x = np.dot(scipy.sparse.linalg.expm(dt*A11),x)

        # compute closure updating
        if 'true_closure' in type_closure:
        # simply validating true closure
            x = x + dt*targeted_closure[index, :]
            # debug
            # if index == 0:
            #    print 'true first closure pred = ', closure[index, :]

        elif 'ml_closure' in type_closure:
        # a posteriori validation with ML closure

            # updating state with closure
            x = x + dt*closure_pred

            # obtain d\delta^{n}/dt in reduced imaginary sense
            current_takens_feature = targeted_closure['takens_feature_generator'].get_takens_feature()
            closure_increment = targeted_closure['model'].predict(current_takens_feature)
            closure_increment = closure_increment.flatten()

            # evolving delta: get \delta^{n+1} in full
            closure_pred = closure_pred + dt*closure_increment

            # update takens_feature with x^{n+1} and \delta^{n+1}
            targeted_closure['takens_feature_generator'].update_state_closure(closure_pred, x)

    return xsnap


