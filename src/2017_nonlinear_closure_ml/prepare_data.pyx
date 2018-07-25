# import python modules
import numpy as np

def read_data_standard(config_dict):
    """
    given data path, return states and closure

    :param config_dict:
    :return:
    """

    # data path
    folder_path = config_dict['file_path']
    file_name = config_dict['file_name']
    data_path = folder_path + file_name

    print '======================================================='
    print ''
    print 'READING DATA...'
    print ''
    print 'path of data: ', data_path
    print ''
    print '======================================================='

    # load data
    sequence_data = np.load(data_path)

    # states and closure are all scaled
    ## note we use usnap_les as an unified name for box filter and fourier filter result
    states = sequence_data['usnap_les'].transpose()
    closure = sequence_data['ec_snap'].transpose()

    print '======================================================='
    print ''
    print 'SUMMARY OF DATA READ'
    print ''
    print 'shape of sequential states data: ', states.shape
    print 'shape of sequential closure data: ', closure.shape
    print ''
    print '======================================================='

    return states, closure


def compute_takens_embedding(original_data, num_previous_state):
    """ given original data as a data matrix of shape = (N, Q),
    and number of previous state: P
    output a takens embedded matrix with size (N-P, (P+1)*Q)

    with order as | x_1, x_2,....., x_P+1 |
                  | x_2, x_3,....., x_P+2 |
                  | ..................... |
                  | x_N-P, ......., x_N   |

    Note that each x_1 is (1,Q), so its shape = (N-P, (P+1)*Q)

    :param original_data:
    :param num_previous_state:
    :return:
    """
    takens_embedded_matrix = np.zeros((original_data.shape[0] - num_previous_state,
                                       original_data.shape[1] * (num_previous_state + 1)))
    for i in xrange(takens_embedded_matrix.shape[0]):
        takens_embedding_list = []
        # alternative solution
        # takens_embedded_matrix[i, :] = original_data[i:i + num_previous_state + 1]
        for j in xrange(num_previous_state + 1):
            takens_embedding_list.append(original_data[i + j, :])
        takens_embedded_matrix[i, :] = np.hstack(tuple(takens_embedding_list))

    return takens_embedded_matrix


def takens_feature2standard_state(takens_feature, dim_reduced, num_pre_state):
    """
    extract state from takens feature

    :param takens_feature:
    :param dim_reduced:
    :param num_pre_state:
    :return:
    """
    if num_pre_state != 0:
        current_state = takens_feature[-(num_pre_state + 1) * dim_reduced:-(num_pre_state) * dim_reduced]
    else:
        current_state = takens_feature[-dim_reduced:]

    return current_state


class takens_feature_generator:
    """
    Class to build Takens feature, just a single time:
        - from delta, and u
        - the order is \chi_n = | delta_{n-P},...,delta_{n},x_{n-P},...,x_{n} |
        - each time updating is to remove delta_{n-P}, and update delta_{n+1} and x_{n+1}
        - to get \chi_{n+1} = | delta_{n-P+1},...,delta_{n+1},x_{n-P+1},...,x_{n+1} |

    """
    def __init__(self, takens_feature, num_pre_state, type_takens):
        self.takens_feature = takens_feature
        self.num_pre_state = num_pre_state
        self.takens_matrix_shape = takens_feature.shape
        if type_takens == 'eco':
            self.reduced_dimension = takens_feature.size / (num_pre_state + 2)
            self.index_target_end = self.reduced_dimension
        else:
            self.reduced_dimension = takens_feature.size / 2 / (num_pre_state + 1)
            self.index_target_end = takens_feature.size / 2

    def update_state_closure(self, reduced_imag_closure, uk_reduced_imag):
        self.takens_feature = np.hstack((self.takens_feature[self.reduced_dimension:self.index_target_end],
                                         reduced_imag_closure,
                                         self.takens_feature[self.index_target_end + self.reduced_dimension:],
                                         uk_reduced_imag))
        assert self.takens_feature.shape == self.takens_matrix_shape

    def get_takens_feature(self):
        return self.takens_feature

    def get_current_delta(self):
        current_delta = self.takens_feature[self.index_target_end - self.reduced_dimension:self.index_target_end]
        return current_delta


def prepare_data_standard(config_dict):
    """
    prepare data for standard problem, where no transformation is needed
    and everything is in Real number

    :param config_dict:
    :return:
    """

    # unshifted, unscaled states and closure
    states, closure = read_data_standard(config_dict)

    num_state = states.shape[0]
    Q_state = states.shape[1]
    num_pre_state = config_dict['num_pre_state']

    # standard data preparation
    feature = states[:-1, :]  # x_{0},...,x_{N-2}
    target = closure[:-1, :]  # delta_{0},...,x_{N-2}

    assert feature.shape == (num_state-1, Q_state)
    assert target.shape ==  (num_state-1, Q_state)

    # prepare features:
    ## embedding time delayed effects for features
    feature_mem = compute_takens_embedding(original_data=feature,
                                           num_previous_state=config_dict['num_pre_state'])
    # feature mem = | x_{0},...,x_{p}  |  t = p
    #               | x_{N-2-p},..x_{N-2}| t = N-2

    # debug
    # print feature_mem.shape
    # print (num_state-1-num_pre_state, num_pre_state + 1)
    # print num_state, num_pre_state

    assert feature_mem.shape == (num_state-1-num_pre_state, Q_state*(num_pre_state + 1))

    ## embedding time delayed effects for target
    target_mem = compute_takens_embedding(original_data=target,
                                          num_previous_state=config_dict['num_pre_state'])
    assert target_mem.shape == (num_state-1-num_pre_state, Q_state*(num_pre_state + 1))

    ## target without memory
    target_without_previous_mem = compute_takens_embedding(original_data=target,
                                                           num_previous_state=0)


    ## treatment for multi time effects
    if 'cross_time_eco' in config_dict['ML_method']['name']:
        ## assume cross time effects but with a economic form
        feature_final = np.hstack((target_without_previous_mem, feature_mem))
        assert feature_final.shape == (num_state-1-num_pre_state, Q_state*(num_pre_state + 2))

    elif 'cross_time' in config_dict['ML_method']['name']:
        ## assume cross time effects of memory term
        feature_final = np.hstack((target_mem, feature_mem))
        assert feature_final.shape == (num_state-1-num_pre_state, 2*Q_state*(num_pre_state + 1))
        # Now here I simply use more features in the input layer.. feature sets are overdetermined, not independent.
#         feature_final = np.hstack((target[config_dict['num_pre_state']:-1, :], feature_mem))

    elif 'time_parallel_delta_x_pair' in config_dict['ML_method']['name']:
        ## assume linear treatment of memory effects
        feature_final = np.hstack((target_mem, feature_mem))
        assert feature_final.shape == (num_state-1-num_pre_state, 2*Q_state*(num_pre_state + 1))

    # prepare target: d\delta/dt
    target_ddt = (closure[1:, :] - closure[:-1, :])/config_dict['aposteriori_config']['dt']

    ## treatment for multi time target
    target_ddt = target_ddt[config_dict['num_pre_state']:]

    ## target_ddt = |y_p+1-y_p/dt ,...., y_N-1-y_N-2/dt |
    assert target_ddt.shape == (num_state-1-num_pre_state, Q_state)

    # train-test-split: features and target

    ## determine number of modes and total step
    config_dict['tot_step'] = feature_final.shape[0]
    train_step = int(config_dict['tot_step'] * config_dict['train_total_ratio'])

    ## update time snap
    tsnap_takens = config_dict['aposteriori_config']['t'][config_dict['num_pre_state']:-1]

    ## train-test-split
    train_time = tsnap_takens[:train_step]
    test_time = tsnap_takens[train_step:]

    train_feature = feature_final[:train_step, :]
    train_target = target_ddt[:train_step, :]

    test_feature = feature_final[train_step:, :]
    test_target = target_ddt[train_step:, :]

    # test to shrink state to exactly match
    # to match next state
    targeted_states = states[config_dict['num_pre_state']+1:]

    train_next_state = targeted_states[:train_step, :]
    test_next_state = targeted_states[train_step:, :]

    # get full complex closure: ground truth
    # target_closure = \delta_{p+1},....,\delta_{N-1}
    targeted_closure = closure[config_dict['num_pre_state']+1:]
    train_closure = targeted_closure[:train_step, :]
    test_closure =  targeted_closure[train_step:, :]

    # true closure to add
    true_closure_to_add = closure[config_dict['num_pre_state']:-1]

    # debug
    try:
        assert train_time.shape[0] == train_feature.shape[0]
        assert train_time.shape[0] == train_target.shape[0]
        assert test_time.shape[0] == test_feature.shape[0]
        assert test_time.shape[0] == test_target.shape[0]
    except AssertionError:
        print 'AssertionError Found!'
        print '================'
        print train_time.shape[0], train_feature.shape[0]
        print train_time.shape[0], train_target.shape[0]
        print test_time.shape[0], test_feature.shape[0]
        print test_time.shape[0], test_target.shape[0]
        print '================'

    # print shape of data
    print '======================================================='
    print ''
    print 'SUMMARY OF TRAIN-TEST DATA'
    print ''
    print 'shape of training feature data: ', train_feature.shape
    print 'shape of training target data: ', train_target.shape
    print 'shape of testing feature data: ', test_feature.shape
    print 'shape of testing target data: ', test_target.shape
    print 'shape of training state: ', train_next_state.shape
    print 'shape of testing state: ', test_next_state.shape
    print 'shape of training closure: ', train_closure.shape
    print 'shape of testing closure: ', test_closure.shape
    print ''
    print '======================================================='

    # pass into dict
    train_data_dict = {'time': train_time,
                       'feature': train_feature,
                       'target': train_target,
                       'states': train_next_state,
                       'closure': train_closure}

    test_data_dict = {'time': test_time,
                      'feature': test_feature,
                      'target': test_target,
                      'states': test_next_state,
                      'closure': test_closure}

    return train_data_dict, test_data_dict, targeted_states, targeted_closure, true_closure_to_add

