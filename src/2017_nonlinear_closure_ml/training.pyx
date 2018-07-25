# SINDy packages
# import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import lsqr
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sparse_identification import sindy
from gen_poly_feature import ClassGeneratePolynomialFeature
from sklearn.metrics import mean_squared_error

# ANN packages
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras.losses import *
from keras.initializers import *
from keras import backend as K

from plot_result import plot_learning_curve
from sklearn.linear_model import Lasso

class ClassSINDy:
    """
    Class for SINDy model
    """
    def __init__(self, feature, target, order, fix_style,
                 sysDim, l1, featureStyle, poly_type,
                 turn_on_mylibrary_poly_feature,
                 solver_type, log_save_path, scaling):
        self.feature = feature
        self.target = target
        self.featureStyle = featureStyle
        self.scaling = scaling

        # scaling feature or NOT
        if self.scaling:
            self.scaler = StandardScaler()
            self.feature = self.scaler.fit_transform(feature)
        else:
            self.feature = feature

        # compute total number of delta x pair = num_pre_step + 1
        self.num_total_delta_x_pair = feature.shape[1] / (2*sysDim)

        ## choose the generator for polynomial feature
        print 'turning on my implementation of polynomial features? ',turn_on_mylibrary_poly_feature
        if turn_on_mylibrary_poly_feature:
            self.library = ClassGeneratePolynomialFeature(poly_type=poly_type,
                                                          degree=order,
                                                          fix_style=fix_style)
            # here we assume no cross time, simply parallel
            # so 2*sysDim is our input dimension for polynomial feature geneartor!
            self.library.compute_tdf_possible_features(sysDim=2*sysDim)
            # factor of two due to delta, x

        else:
            self.library = PolynomialFeatures(degree=order, include_bias=True)

        # true system dimension, not feature, so same dimension as d delta/ dt
        self.dim = sysDim

        ## generate multi-time features - cross time or time parallel
        self.Theta = self._generate_takens_embedding_feature(self.feature)

        # number of features for a single snap
        self.n_lib = self.library.n_output_features_

        # summarizing the multi task problem into one
        # which is not good at all..
        ThetaList = [self.Theta] * self.dim
        self.A = block_diag(*ThetaList)
        self.b = self.target.flatten(order='F')
        # this computation does not count at all.
        self.shols = sindy(l1=l1, l2=1e-4, solver='lstsq')  # lstsq, lasso

        # save data

        # since vbe sine is dense, using SINDy does not make sense
        if solver_type == 'cvx_gradient':
            print 'using cvx gradient'
            # solver this linear system using SINDy packages
            self.shols.fit(self.A, self.b)

            # debug
            # self.shols.coef_ = np.array([0,-1,0,0,2,0,-2,0,0,0]);
        elif solver_type == 'direct':
            print 'using direct lsqr'
            # solver this linear system using LSQR method, in a direct sense
            self.shols.coef_ = lsqr(self.A, self.b)[0]

        elif solver_type == 'lasso':
            print 'using lasso'
            # for each column, using lasso
            coef_vector = []
            for icol in xrange(self.dim):

                # for different target, use different l1
                l1_for_single_target = l1[icol]

                feature_matrix = self.Theta
                target_vector = self.target[:,icol]

                reg = Lasso(alpha=l1_for_single_target, normalize=True, tol=1e-10, warm_start=False, max_iter=1e6)

                reg.fit(X=feature_matrix, y=target_vector)
                coef_vector.append(reg.coef_)

            # convert list of coefficient into vector
            coef_vector = np.array(coef_vector)
            coef_vector = coef_vector.reshape(-1,1)

            # combine all coefficient for all target together
            self.shols.coef_ = coef_vector

            # debug
            # self.shols.coef_ = np.array([0,-0.2,0,0,1.75,0,0,0,0,-1]);

        print ''
        print 'SINDy begin...'
        print ''
        print 'l1 = ', l1
        print 'Total number of combination of polynomial features considered: ', self.Theta.shape[1]
        print 'Total number of coefficients: ', self.shols.coef_.size
        print 'Computed number of features for a single snaptime: ',self.n_lib
        print 'Total number of equations: ', self.Theta.shape[0] * sysDim
        print 'Number of non-zero coefficients: ', np.count_nonzero(self.shols.coef_)
        print 'Note: this number of coefficients is total coefficents for this vector polynomial function'

        # save meta data to file
        with open(log_save_path + 'poly_meta_data_coefficient.txt', 'w') as the_file:
            the_file.write('l1 = ' + str(l1) + '\n')
            the_file.write('Total number of combination of polynomial features considered: \n' + str(self.Theta.shape[1]) + '\n')
            the_file.write('Total number of coefficients: \n' + str(self.shols.coef_.size) + '\n')
            the_file.write('Total number of equations: \n' + str(self.Theta.shape[0] * sysDim) + '\n')
            the_file.write('Number of non-zero coefficients: \n' + str(np.count_nonzero(self.shols.coef_)) + '\n')
            the_file.write('Coefficients collected as 1D array: \n' + str(self.shols.coef_) + '\n')

        # save coeffs to file
        np.savez('coef.npz',coef=self.shols.coef_)

    def save_Ab(self, filename):
        b_2d_save = np.reshape(self.b, (-1, self.dim))
        np.savez('train_data_poly_Ab_' + filename + '.npz', A=self.A, b=b_2d_save)

    def _generate_takens_embedding_feature(self, feature):
        """
        Generating multi time feature

        :param
            feature: , self.featureStyle
        :return:
            feature matrix
        """
        if self.featureStyle == 'sindy_cross_time':
            Theta = self.library.fit_transform(feature)

        elif self.featureStyle == 'sindy_time_parallel_delta_x_pair':
            # compute target and feature separately
            tmp_num = self.feature.shape[1] / 2
            tmp_target = feature[:, :tmp_num]
            tmp_feature = feature[:, tmp_num:]
            num_delta_hat_pair = tmp_num / self.dim
            tmp_feature_time_linear = []
            for i in xrange(num_delta_hat_pair):
                tmp_feature_transform = np.hstack((tmp_target[:, (i * self.dim):((i + 1) * self.dim)],
                                                   tmp_feature[:, (i * self.dim):((i + 1) * self.dim)]
                                                   ))
                gen_feature_single_snap_time_parallel = self.library.fit_transform(tmp_feature_transform)
                # print 'tmp_feature_transform is shape = ',tmp_feature_transform.shape
                # print 'generated feature single snap is shape = ', gen_feature_single_snap_time_parallel.shape
                tmp_feature_time_linear.append(gen_feature_single_snap_time_parallel)

                # note that first one in tmp_feature_time_linear is Feature(n-p), last one is Feature(n)
            # ensemble (\delta, \hat{x}) generated feature pair
            num_feature_each_time = gen_feature_single_snap_time_parallel.shape[1]
            Theta = np.hstack(tmp_feature_time_linear)
            delete_col_list = [(i+1)*num_feature_each_time for i in xrange(num_delta_hat_pair-1)]
            Theta = np.delete(Theta, delete_col_list, 1)

            ## the following is debug process if things went wrong...
            # print 'debug'
            # print 'num_feature_each_time ',num_feature_each_time
            # print 'gen_feature_single_snap_time_parallel.shape', gen_feature_single_snap_time_parallel.shape
            # print 'num_delta_hat_pair', num_delta_hat_pair
            # print Theta.shape
            # print Theta[0,:]
            # print ''
        return Theta


    def showFitResult(self):
        print 'dimension of system = ', self.dim
        for imode in range(self.dim):
            print ' -- coefficients of No. ', imode, ' -- '
            num_total_features = self.n_lib*self.num_total_delta_x_pair
            print self.shols.coef_[imode * num_total_features:(imode + 1) * num_total_features], "\n"
        prediction = self.predict(self.feature)
        # print 'feature shape', self.feature.shape, type(self.feature)
        # print 'target shape', self.target.shape, type(self.target)
        # print 'prediction shape', prediction.shape, type(prediction)
        mse_error = np.square(self.target - prediction).mean()
        print 'training phase: mse = ', mse_error


    def predict(self, feature):
        """
        predict feature using self.shols.coefs_

        :param feature:

        :return: predicted value, note that the reshape order is 'F'

        """

        # need to transform input feature to scaled feature or NOT!
        if self.scaling:
            if feature.ndim == 2:
                feature = self.scaler.transform(feature)
            elif feature.ndim == 1:
                feature = self.scaler.transform(feature.reshape(1,-1))
            else:
                print 'error! Feature ndim is wrong...', feature.ndim
        else:
            if feature.ndim == 2:
                feature = feature
            elif feature.ndim == 1:
                feature = feature.reshape(1,-1)
            else:
                print 'error! Feature ndim is wrong...', feature.ndim

        Theta = self._generate_takens_embedding_feature(feature)
        ThetaList = [Theta] * self.dim
        self.A = block_diag(*ThetaList)
        pred = self.shols.predict(self.A)

        # reshape prediction into feature size
        pred = pred.reshape(feature.shape[0], -1, order='F')

        return pred


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# def ChooseLoss(loss_name):
#     if loss_name == 'mse':
#         loss = mean_squared_error
#     elif loss_name == 'mae':
#         loss = mean_absolute_error
#     elif loss_name == 'mape':
#         loss = mean_absolute_percentage_error
#     elif loss_name == 'msle':
#         loss = mean_squared_logarithmic_error
#     elif loss_name == 'sh':
#         loss = squared_hinge
#     elif loss_name == 'h':
#         loss = hinge
#     elif loss_name == 'logcosh':
#         loss = logcosh
#     elif loss_name == 'kld':
#         loss = kullback_leibler_divergence
#     elif loss_name == 'cos':
#         loss = cosine_proximity
#     else:
#         print 'no loss function selected!!'
#         sys.exit()
#     return loss


class ClassANN:
    """
    Class for ANN model with two hidden layers
    """
    def __init__(self, config_dict):

        self.config_dict = config_dict

        loss = mean_squared_error # ChooseLoss(loss_name='mse')
        opti = Adam(lr=config_dict['ML_method']['lr'],
                    decay=config_dict['ML_method']['decay'])
        self.bs = config_dict['ML_method']['mini_batch']
        self.nb = config_dict['ML_method']['npoch']
        self.dim = config_dict['reduced_modes']
        self.ML_method_name =  config_dict['ML_method']['name']
        nh = config_dict['ML_method']['number_hidden_units']
        actFunStr = config_dict['ML_method']['activationFun']
        if config_dict['ML_method']['weight_regularizer'] == 'l1':
            weightReg = regularizers.l1(config_dict['ML_method']['l1_regu_coef'])
        elif config_dict['ML_method']['weight_regularizer'] == 'l2':
            weightReg = regularizers.l2(config_dict['ML_method']['l2_regu_coef'])
        else:
            weightReg = None

        ## build model: two hidden layers

        if self.ML_method_name == 'ann_time_parallel_delta_x_pair':

            input_list = []
            output_list = []

            for i_time in xrange(config_dict['num_pre_state']+1):
                # note that sine wave: imag reduced: \delta, state need factor of two
                input = Input(shape=(2 * config_dict['reduced_modes'],))
                input_list.append(input)

                # first hidden layer
                pre_hidden_1 = Dense(units=nh,
                                 kernel_initializer=he_uniform(),
                                 bias_initializer=he_uniform(),
                                 input_dim=2*config_dict['reduced_modes'],
                                 kernel_regularizer=weightReg)(input)
                hidden_1 = Activation(actFunStr)(pre_hidden_1)

                # input_dim=2*config_dict['reduced_modes']*(config_dict['num_pre_state'] + 1),

                # second hidden layer
                pre_hidden_2 = Dense(units=nh,
                                 kernel_initializer=he_uniform(),
                                 bias_initializer=he_uniform(),
                                 kernel_regularizer=weightReg)(hidden_1)
                hidden_2 = Activation(actFunStr)(pre_hidden_2)

                # final output layer # contribution of this step FOR delta at next step
                output = Dense(units=config_dict['reduced_modes'],
                                kernel_initializer=he_uniform(),
                                bias_initializer=he_uniform(),
                                kernel_regularizer=weightReg)(hidden_2)
                output_list.append(output)

            if config_dict['num_pre_state'] > 0:
                final_output = Add()(output_list)
            else:
                final_output = output_list[0]

            self.model = Model(inputs=input_list, outputs=final_output)

        elif self.ML_method_name == 'ann_cross_time':

            # note that sine wave: imag reduced: \delta, state need factor of two
            input = Input(shape=(2*(config_dict['num_pre_state']+1)*config_dict['reduced_modes'],))

            # first hidden layer
            pre_hidden_1 = Dense(units=nh,
                             kernel_initializer=he_uniform(),
                             bias_initializer=he_uniform(),
                             input_dim=2*config_dict['reduced_modes']*(config_dict['num_pre_state'] + 1),
                             kernel_regularizer=weightReg)(input)
            hidden_1 = Activation(actFunStr)(pre_hidden_1)

            # second hidden layer
            pre_hidden_2 = Dense(units=nh,
                             kernel_initializer=he_uniform(),
                             bias_initializer=he_uniform(),
                             kernel_regularizer=weightReg)(hidden_1)
            hidden_2 = Activation(actFunStr)(pre_hidden_2)

            # final output layer
            output = Dense(units=config_dict['reduced_modes'],
                            kernel_initializer=he_uniform(),
                            bias_initializer=he_uniform(),
                            kernel_regularizer=weightReg)(hidden_2)

            self.model = Model(inputs=input, outputs=output)

        elif self.ML_method_name == 'ann_cross_time_eco':

            # note that sine wave: imag reduced: \delta, state need factor of two
            # economic version
            input = Input(shape=((config_dict['num_pre_state']+2)*config_dict['reduced_modes'],))

            # first hidden layer
            pre_hidden_1 = Dense(units=nh,
                             kernel_initializer=he_uniform(),
                             bias_initializer=he_uniform(),
                             input_dim=config_dict['reduced_modes']*(config_dict['num_pre_state'] + 2),
                             kernel_regularizer=weightReg)(input)
            hidden_1 = Activation(actFunStr)(pre_hidden_1)

            # second hidden layer
            pre_hidden_2 = Dense(units=nh,
                             kernel_initializer=he_uniform(),
                             bias_initializer=he_uniform(),
                             kernel_regularizer=weightReg)(hidden_1)
            hidden_2 = Activation(actFunStr)(pre_hidden_2)

            # final output layer
            output = Dense(units=config_dict['reduced_modes'],
                            kernel_initializer=he_uniform(),
                            bias_initializer=he_uniform(),
                            kernel_regularizer=weightReg)(hidden_2)

            self.model = Model(inputs=input, outputs=output)

        ## compile model
        self.model.compile(opti, loss=loss, metrics=[r2_keras])
        with open(config_dict['image_path'] + 'ann_summary_report.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def fit(self, feature, target):
        ## scaling training data
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        featureScaled = self.scaler_x.fit_transform(feature)
        targetScaled = self.scaler_y.fit_transform(target)

        # randomize data before training in ANN
        indexRandom = np.random.permutation(featureScaled.shape[0])
        np.take(featureScaled, indexRandom, axis=0, out=featureScaled)
        np.take(targetScaled, indexRandom, axis=0, out=targetScaled)

        # compute time parallel features
        if self.ML_method_name == 'ann_cross_time' or self.ML_method_name == 'ann_cross_time_eco':
            feature_multi_time = self.compute_cross_time_feature(featureScaled)
        elif self.ML_method_name == 'ann_time_parallel_delta_x_pair':
            feature_multi_time = self.compute_time_parallel_feature_list(featureScaled)

        # save feature and target
        self.train_scaled_feature = feature_multi_time
        self.train_scaled_target = targetScaled

        # print 'debug: shape of time parallel feature list', len(feature_multi_time), feature_multi_time[0].shape,feature_multi_time[1].shape

        # featureScaled
        # fit the model
        self.history = self.model.fit(feature_multi_time, targetScaled,
                            validation_split=1 - self.config_dict['ML_method']['training_validation_split_ratio'],
                            shuffle=False,
                            epochs=self.nb,
                            verbose=0,
                            batch_size=self.bs)

        # plot learning curve
        train_loss = self.history.history['loss']
        val_loss = self.history.history['loss']
        train_r2 = self.history.history['r2_keras']
        val_r2 = self.history.history['val_r2_keras']

        plot_learning_curve(train_loss, val_loss, train_r2, val_r2, self.config_dict)

    def save_feature_target(self, filename):
        np.savez('train_data_ann_x_y_' + filename + '.npz',
                 x=self.train_scaled_feature,
                 y=self.train_scaled_target)

    def compute_cross_time_feature(self, featureScaled):
        return np.vstack(featureScaled)

    def compute_time_parallel_feature_list(self, featureScaled):
        time_parallel_feature_list = []

        tmp_num = featureScaled.shape[1] / 2
        tmp_target = featureScaled[:, :tmp_num]
        tmp_feature = featureScaled[:, tmp_num:]
        num_delta_hat_pair = tmp_num / self.dim
        for i in xrange(num_delta_hat_pair):
            tmp_feature_at_i_time = np.hstack((tmp_target[:, (i * self.dim):((i + 1) * self.dim)],
                                               tmp_feature[:, (i * self.dim):((i + 1) * self.dim)]))
            time_parallel_feature_list.append(tmp_feature_at_i_time)

        return time_parallel_feature_list

    def predict(self, feature):
        # need to transform input feature to scaled feature
        if feature.ndim == 2:
            featureScaled = self.scaler_x.transform(feature)
        elif feature.ndim == 1:
            featureScaled = self.scaler_x.transform(feature.reshape(1,-1))

        # compute time parallel features
        if self.ML_method_name == 'ann_cross_time' or self.ML_method_name == 'ann_cross_time_eco':
            feature_multi_time = self.compute_cross_time_feature(featureScaled)
        elif self.ML_method_name == 'ann_time_parallel_delta_x_pair':
            feature_multi_time = self.compute_time_parallel_feature_list(featureScaled)

        # prediction of ANN
        # predScaled = self.model.predict(featureScaled)
        predScaled = self.model.predict(feature_multi_time)

        # inverse to unscaled target value
        pred = self.scaler_y.inverse_transform(predScaled)

        return pred


# machine learning function
# -- call sindy
# -- call ANN
def machine_learning(config_dict, train_data_dict):
    # prepare feature and target, and feature generation style
    featureStyle = config_dict['ML_method']['name']
    feature = train_data_dict['feature']
    target = train_data_dict['target']

    # implementation ML model
    if config_dict['ML_type'] == 'sindy':
    # --> sindy_cross_time & sindy parallel time + delta_x_pair
        print 'ML model trained using SINDy'
        # if config_dict['ML_method']['fix_order'] == 'tdf':
        #     system_dimension_factor = 2
        # elif config_dict['ML_method']['fix_order'] == 'edf':
        #     system_dimension_factor = 1
        model = ClassSINDy(feature=feature,
                           target=target,
                           order=config_dict['ML_method']['order'],
                           fix_style=config_dict['ML_method']['fix_order'],
                           sysDim=config_dict['reduced_modes'],
                           l1=config_dict['ML_method']['l1'],
                           featureStyle=featureStyle,
                           poly_type=config_dict['ML_method']['poly_type'],
                           turn_on_mylibrary_poly_feature=config_dict['ML_method']['turn_on_mylibrary_poly_feature'],
                           solver_type=config_dict['ML_method']['solver_type'],
                           log_save_path=config_dict['image_path'],
                           scaling=config_dict['scaling'])
        if config_dict['verbose']:
            model.showFitResult()
            filename = 'order_'+str(config_dict['ML_method']['order']) + '_' + \
                       'fix_style_' + str(config_dict['ML_method']['fix_order']) + \
                       'sysDim_' + str(config_dict['reduced_modes']) + \
                       'num_pre_states_' + str(config_dict['num_pre_state'])
            model.save_Ab(filename)

    elif 'ann' in config_dict['ML_type']:
    # --> ann
        print 'ML model trained using ANN'

        model = ClassANN(config_dict=config_dict)
        model.fit(feature=feature, target=target)

        if config_dict['verbose']:
            filename = 'scaled_num_pre_states_' + str(config_dict['num_pre_state'])
            model.save_feature_target(filename)

    return model
