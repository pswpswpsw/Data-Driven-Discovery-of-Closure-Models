import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from run_aposteriori import compute_aposteriori_3d_linear
from run_aposteriori import compute_aposteriori_2d_vdp
from run_aposteriori import compute_aposteriori_3d_lorenz
from prepare_data import takens_feature2standard_state
from prepare_data import takens_feature_generator
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from nolds import lyap_e, corr_dim

# plt.style.use('siads')

color_list = ['r', 'g', 'b', 'k', 'c', 'y']


def compute_pair_index(dim):
    pairIndexList = []
    for i in xrange(dim):
        for j in xrange(dim):
            if i > j:
                pairIndexList.append([i, j])
    return pairIndexList


def viz_2d_feature(TrainDict, TestDict, ConfigDict):
    """
    Genearte 2d plot for train-test comparison
    :param TrainDict:
    :param TestDict:
    :param ConfigDict:
    :return:
    """
    featureTrain = TrainDict['feature']
    featureTest = TestDict['feature']

    pairIndexList = compute_pair_index(featureTrain.shape[1])

    for pair in pairIndexList:
        index1, index2 = pair
        plt.figure()
        plt.scatter(featureTrain[:, index1], featureTrain[:, index2], s=16)
        plt.scatter(featureTest[:, index1], featureTest[:, index2], s=2)
        plt.xlabel('component ' + str(index1 + 1))
        plt.ylabel('component ' + str(index2 + 1))
        lgd = plt.legend(['train', 'test'], bbox_to_anchor=(1, 0.5))
        plt.savefig(ConfigDict['image_path'] + str(index1) + '_vs_' + str(index2) +'_feature_compare.png',
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

    return

def animate_udist_comparison(x, u1, u2, config_dict, legend, case_name):
    """
    create animation of u distributed in spatial with comparision between u1 and u2

    :param x:
    :param u1:
    :param u2:
    :param config_dict:
    :param legend:
    :param case_name:
    :return:
    """

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln1, = plt.plot([], [], 'r-', animated=True)
    ln2, = plt.plot([], [], 'b--', animated=True)
    # plt.legend(legend, bbox_to_anchor=(1, 0.5))
    plt.legend(legend,loc="upper right")
    plt.xlabel(r'x')
    plt.ylabel(r'u(x,t)')

    # set figure limits

    def init():
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1, 1)
        return ln1,ln2

    # set x axis
    xdata = [x]

    # stacking x and y together
    frames = np.hstack((u1, u2))

    index_half = frames.shape[1]/2

    def update(frame):
        ydata = frame[:index_half]
        zdata = frame[index_half:]
        ln1.set_data(xdata, ydata)
        ln2.set_data(xdata, zdata)
        return ln1, ln2

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=10, blit=True)

    ani.save(config_dict['image_path'] + case_name + '_u_distribution.mp4')


def plot_u_dist(x, u, config_dict, case_name):
    """
    plot u distribution  and save it

    :param x:
    :param u:
    :param config_dict:
    :param case_name:
    :return:
    """
    plt.figure()
    plt.plot(x, u.transpose())
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.savefig(config_dict['image_path'] + 'udistribution_' + case_name + '.png')
    plt.close()


def plot_learning_curve(train_loss_list, vali_loss_list, train_r2_list, vali_r2_list, config_dict):
    """
    plot learning curve from ANN models

    :param train_loss_list:
    :param vali_loss_list:
    :param train_r2_list:
    :param vali_r2_list:
    :param config_dict:
    :return:
    """
    # learning curve
    plt.figure()
    plt.semilogy(train_loss_list,'-r')
    plt.semilogy(vali_loss_list,'--b')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    lgd = plt.legend(['train loss','validation loss'],bbox_to_anchor=(1, 0.5))
    plt.savefig(config_dict['image_path'] + 'learning_curve.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # r2
    plt.figure()
    plt.plot(train_r2_list, '-r')
    plt.plot(vali_r2_list, '-b')
    plt.xlabel('iteration')
    plt.ylabel('$R^2$')
    plt.ylim([0, 1])
    lgd = plt.legend(['train $R^2$', 'validation $R^2$'],bbox_to_anchor=(1, 0.5))
    plt.savefig(config_dict['image_path'] + 'r2.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plot_one_component_phase_plot(predict, target, config_dict, case_name):
    """
    plot phase plot of first three components
    :param predict:
    :param target:
    :param case_name:
    :return:
    """

    # plot in phase space: predict
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 270)
    ax.scatter(predict[0:-3], predict[1:-2], predict[2:-1], c='k', marker='o')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlabel(r'$x_3$')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([-25, 25])
    plt.savefig(config_dict['image_path'] + case_name + '_phase_plot_predict.png', bbox_inches='tight')
    plt.close()

    # plot in phase space: true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 0)
    ax.scatter(target[0:-3], target[1:-2], target[2:-1], c='k', marker='o')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlabel(r'$x_3$')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([-25, 25])
    plt.savefig(config_dict['image_path'] + case_name + '_phase_plot_target.png', bbox_inches='tight')
    plt.close()

    # plot in phase space: true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(80, 0)
    ax.scatter(predict[0:-3], predict[1:-2], predict[2:-1], c='r', marker='o')
    ax.scatter(target[0:-3], target[1:-2], target[2:-1], c='b', marker='o')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlabel(r'$x_3$')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([-25, 25])
    plt.savefig(config_dict['image_path'] + case_name + '_phase_plot_compare.png', bbox_inches='tight')
    # np.savez(config_dict['image_path'] + case_name +'data.npz',predict=predict, target=target)
    plt.close()


def plot_total_component_vs_time_standard(tsnap, target, predict, config_dict, case_name):
    """
    plot each component, for standard data

    :param tsnap:
    :param target:
    :param predict:
    :param config_dict:
    :param case_name:
    :return:
    """

    num_target = target.shape[1]
    # total components line plot
    plt.figure()
    # plot target vs time
    for i in xrange(num_target):
        plt.plot(tsnap, target[:, i], color_list[i] + '-',
                 markersize=config_dict['markersize'],
                 label='target: mode: ' + str(-num_target + i))

    # plot prediction vs time
    for i in xrange(num_target):
        plt.plot(tsnap, predict[:, i], color_list[i-1] + '--',
                 label='prediction: mode: ' + str(-num_target + i),
                 markersize=config_dict['markersize'])

    if 'aposter' in case_name:
        plt.ylabel(r'component value of $x$') # unscaled
    else:
        plt.ylabel(r'$\frac{d\delta}{dt}$')

    plt.xlabel(r'time')
    lgd = plt.legend([r'target: mode: ' + str(1 + x) for x in
                xrange(num_target)] + [r'prediction: mode: ' + str(1 + x)
                for x in xrange(num_target)], bbox_to_anchor=(1, 0.5)
                     )
    plt.savefig(config_dict['image_path'] + case_name + '_time_variation_comparison_component.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # save data
    np.savez(config_dict['image_path'] + case_name + '_posteriori_data.npz',target=target, predict=predict)






def plot_time_vs_standard_data(data_dict, config_dict, case_name):
    """
    plot for feature and target versus time

    :param data_dict:
    :param config_dict:
    :param case_name:
    :return:
    """

    tsnap = data_dict['time']
    feature = data_dict['feature']
    n_dim_state = config_dict['reduced_modes']
    target = data_dict['target']

    # state vs time, current state only
    plt.figure()
    plt.plot(tsnap, feature[:,-n_dim_state:], '-', markersize=config_dict['markersize'])
    plt.xlabel('time')
    plt.ylabel('$x$')
    lgd = plt.legend(['$x_'+str(index+1)+'$' for index in xrange(n_dim_state)],bbox_to_anchor=(1, 0.5))
    plt.savefig(config_dict['image_path'] + case_name + '_feature_state_only' + '.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # state and \delta, vs time, it contains multi state and multi delta
    plt.figure()
    plt.plot(tsnap, feature, '-', markersize=config_dict['markersize'])
    plt.xlabel('time')
    plt.ylabel(r'$\delta$ \& $x$')
    plt.savefig(config_dict['image_path'] +case_name + '_feature' + '.png')
    plt.close()

    # target: \d \delta/\dt vs time
    plt.figure()
    plt.plot(tsnap, target, '-', markersize=config_dict['markersize'])
    plt.xlabel('time')
    plt.ylabel(r'$\frac{d\delta}{dt}$')
    lgd = plt.legend(['mode: '+str(index+1) for index in xrange(n_dim_state)],bbox_to_anchor=(1, 0.5))
    plt.savefig(config_dict['image_path'] + case_name + '_target' + '.png',
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()



def plot_apriori_standard(model, data_dict, config_dict, case_name):
    """
    Plot apriori prediction vs target in X-Y scatter plot

    :param model:
    :param data_dict:
    :param config_dict:
    :param case_name:
    :return:
    """
    tsnap = data_dict['time']
    feature = data_dict['feature']
    target = data_dict['target']

    # predict a huge group of features
    predict = model.predict(feature)

    # mse error
    print '======================================================='
    print 'SUMMARY FOR A PRIORI: ' + case_name
    print case_name, ' MSE = ', mean_squared_error(y_true=target,
                                                   y_pred=predict)

    # np.savez('debug_apriori.npz',target=target,predict=predict)

    print '======================================================='
    # xy plot
    num_target = target.shape[1]
    for i in xrange(num_target):
        plt.figure()
        plt.scatter(target[:, i], predict[:, i], s=config_dict['markersize'])
        minval = min(min(target[:, i]), min(predict[:, i]))
        maxval = max(max(target[:, i]), max(predict[:, i]))
        plt.xlabel('target')
        plt.ylabel('prediction')
        plt.xlim(minval, maxval)
        plt.ylim(minval, maxval)
        plt.savefig(config_dict['image_path'] + case_name + '_xy_apriori_comparison_' + str(i) + '_component.png')
        plt.close()

    # total components line plot: apriori : target vs prediction = d \delta / dt vs predicted
    plot_total_component_vs_time_standard(tsnap, target, predict, config_dict, case_name)




def plot_aposteriori_standard_2d_vdp(model, data_dict, config_dict, case_name):
    """
    Plot a posteriori prediction vs target

    :param model:
    :param data_dict:
    :param config_dict:
    :param case_name:
    :return:
    """
    # time array, feature, and total number of interaction
    tsnap = data_dict['time']
    feature = data_dict['feature']
    target = data_dict['target']
    true_next_state = data_dict['states']
    # if case_name == 'test_aposteriori':
    #     true_state = data_dict['states'][:-1]
    # else:
    #     true_state = data_dict['states']
    tstep = tsnap.size

    # init
    num_pre_state = config_dict['num_pre_state']
    dim_reduced = config_dict['reduced_modes']
    initial_feature = feature[0, :]
    # initial_state = takens_feature2standard_state(initial_feature, dim_reduced, num_pre_state)
    initial_state = initial_feature[-1:]

    # print initial_state
    # print initial_feature

    # build closure dictionary: closure as dictionary
    closure = {}
    closure['model'] = model
    closure['takens_feature_generator'] = takens_feature_generator(initial_feature, num_pre_state, 'full')

    # compute aposteriori
    xsnap = compute_aposteriori_2d_vdp(config_dict,
                                          initial_state,
                                          closure,
                                          tstep,
                                          'ml_closure')


    # total components line plot vs time
    plot_total_component_vs_time_standard(tsnap, true_next_state, xsnap, config_dict, case_name)



def plot_aposteriori_standard_3d_lorenz(model, data_dict, config_dict, case_name):
    """
    Plot a posteriori prediction vs target

    :param model:
    :param data_dict:
    :param config_dict:
    :param case_name:
    :return:
    """
    # time array, feature, and total number of interaction
    tsnap = data_dict['time']
    feature = data_dict['feature']
    target = data_dict['target']
    true_next_state = data_dict['states']
    # if case_name == 'test_aposteriori':
    #     true_state = data_dict['states'][:-1]
    # else:
    #     true_state = data_dict['states']
    tstep = tsnap.size

    # init
    num_pre_state = config_dict['num_pre_state']
    dim_reduced = config_dict['reduced_modes']
    initial_feature = feature[0, :]
    # initial_state = takens_feature2standard_state(initial_feature, dim_reduced, num_pre_state)
    initial_state = initial_feature[-config_dict['reduced_modes']:]

    # print 'initial state =', initial_state
    # print 'initialfeature =', initial_feature

    # print initial_state
    # print initial_feature

    # define type of feature
    if 'eco' in config_dict['ML_method']['name']:
        type_takens = 'eco'
    else:
        type_takens = 'full'

    # build closure dictionary: closure as dictionary
    closure = {}
    closure['model'] = model
    closure['takens_feature_generator'] = takens_feature_generator(initial_feature, num_pre_state, type_takens)

    # compute aposteriori
    xsnap = compute_aposteriori_3d_lorenz(config_dict,
                                          initial_state,
                                          closure,
                                          tstep,
                                          'ml_closure')


    # total components line plot vs time
    plot_total_component_vs_time_standard(tsnap, true_next_state, xsnap, config_dict, case_name)

    # draw phase plot
    ## found it was not useful at all..
    # plot_one_component_phase_plot(xsnap, true_next_state, config_dict, case_name)

    ## chaotic system statistics
    embed_dim = 3
    # compute Lyapounov exponent
    lya_pred = max(lyap_e(xsnap.flatten(),embed_dim,matrix_dim=embed_dim))
    lya_true = max(lyap_e(true_next_state.flatten(),embed_dim,matrix_dim=embed_dim))

    # compute correlation dimension
    cd_pred = corr_dim(xsnap.flatten(), embed_dim)
    cd_true = corr_dim(true_next_state.flatten(), embed_dim)


    print '========================'
    print 'chaotic system statistics'
    print 'case: ', case_name
    print 'Lyapunov exponent computed by method of Eckmann et al'
    print 'prediction: ', lya_pred, 'true: ', lya_true
    print 'correlation dimension'
    print 'prediction: ', cd_pred, 'true: ', cd_true
    print ''





def plot_aposteriori_standard_3d_linear(model, data_dict, config_dict, case_name):
    """
    Plot a posteriori prediction vs target

    :param model:
    :param data_dict:
    :param config_dict:
    :param case_name:
    :return:
    """
    # time array, feature, and total number of interaction
    tsnap = data_dict['time']
    feature = data_dict['feature']
    target = data_dict['target']
    true_next_state = data_dict['states']
    # if case_name == 'test_aposteriori':
    #     true_state = data_dict['states'][:-1]
    # else:
    #     true_state = data_dict['states']
    tstep = tsnap.size

    # init
    num_pre_state = config_dict['num_pre_state']
    dim_reduced = config_dict['reduced_modes']
    initial_feature = feature[0, :]
    # initial_state = takens_feature2standard_state(initial_feature, dim_reduced, num_pre_state)
    initial_state = initial_feature[-config_dict['reduced_modes']:]

    # print initial_state
    # print initial_feature

    # build closure dictionary: closure as dictionary
    closure = {}
    closure['model'] = model
    closure['takens_feature_generator'] = takens_feature_generator(initial_feature, num_pre_state,'full')

    # compute aposteriori
    xsnap = compute_aposteriori_3d_linear(config_dict,
                                          initial_state,
                                          closure,
                                          tstep,
                                          'ml_closure')


    # total components line plot vs time
    plot_total_component_vs_time_standard(tsnap, true_next_state, xsnap, config_dict, case_name)



