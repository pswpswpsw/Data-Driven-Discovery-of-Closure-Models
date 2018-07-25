# immport python modules from numpy and scipy
import numpy as np
from itertools import product
from numpy import polyval
from scipy.special import hermitenorm, legendre


def number_of_partitions(max_range, max_sum):
    """
    Returns an array arr of the same shape as max_range, where
    arr[j] = number of admissible partitions for
             j summands bounded by max_range[j:] and with sum <= max_sum

    :param max_range:
    :param max_sum:
    :return:
    """

    M = max_sum + 1
    N = len(max_range)
    arr = np.zeros(shape=(M, N), dtype=int)
    arr[:, -1] = np.where(np.arange(M) <= min(max_range[-1], max_sum), 1, 0)
    for i in range(N - 2, -1, -1):
        for j in range(max_range[i] + 1):
            arr[j:, i] += arr[:M - j, i + 1]
    return arr.sum(axis=0)


def partition3(max_range, max_sum, out=None, n_part=None):

    if out is None:
        max_range = np.asarray(max_range, dtype=int).ravel()
        n_part = number_of_partitions(max_range, max_sum)
        out = np.zeros(shape=(n_part[0], max_range.size), dtype=int)

    if (max_range.size == 1):
        out[:] = np.arange(min(max_range[0], max_sum) + 1, dtype=int).reshape(-1, 1)
        return out

    P = partition3(max_range[1:], max_sum, out=out[:n_part[1], 1:], n_part=n_part[1:])
    # P is now a useful reference

    S = np.minimum(max_sum - P.sum(axis=1), max_range[0])
    offset, sz = 0, S.size
    out[:sz, 0] = 0
    for i in range(1, max_range[0] + 1):
        ind, = np.nonzero(S)
        offset, sz = offset + sz, ind.size
        out[offset:offset + sz, 0] = i
        out[offset:offset + sz, 1:] = P[ind]
        S[ind] -= 1
    return out


def monomial(degree):
    """
    return monomial
    :param degree:
    :return:
    """
    return np.poly1d([1] + degree * [0])

class ClassGeneratePolynomialFeature:
    """
    Class of generate polynomial features

    Parameters
    ----------

    poly_type:
        'hermitenorm', 'monomial', 'legendre'

    degree:
        int

    fix_style:
        each component fixed order: 'edf'
        total component fixed order: 'tdf'

    """

    def __init__(self, poly_type, degree, fix_style):

        self.poly_type = poly_type
        if poly_type == 'hermitenorm':
            self.polyfun = hermitenorm
        elif poly_type == 'monomial':
            self.polyfun = monomial
        elif poly_type == 'legendre':
            self.polyfun = legendre

        self.order = degree
        self.fix_style = fix_style

    def compute_tdf_possible_features(self, sysDim):
        self.all_possible_features = partition3([self.order] * sysDim, self.order)

    def fit_transform(self, x):
        data_row = x.shape[0]
        q = x.shape[1]

        if self.fix_style == 'edf':
            feature_array = np.zeros((data_row, (self.order + 1) ** q))
            num = 0
            for feature_combination in product(xrange(self.order + 1),
                                               repeat=q):
                # iterate over all feature combinations
                single_combination_feature = 1
                for i_component, current_hermite_degree in enumerate(
                        feature_combination):
                    single_combination_feature *= polyval(self.polyfun(current_hermite_degree), x[:, i_component])

                feature_array[:, num] = single_combination_feature
                num += 1

        elif self.fix_style == 'tdf':

            feature_array = np.zeros((data_row, self.all_possible_features.shape[0]))

            num = 0
            for i_comb in xrange(self.all_possible_features.shape[0]):
                # iterate over all feature combinations
                feature_combination = list(self.all_possible_features[i_comb, :])
                single_combination_feature = 1
                for i_component, current_hermite_degree in enumerate(feature_combination):
                    single_combination_feature *= polyval(self.polyfun(current_hermite_degree), x[:, i_component])

                feature_array[:, num] = single_combination_feature
                num += 1

        self.n_output_features_ = num

        return feature_array

        # def print_feature_info(self):
        #     print 'number of features: ', self.n_output_features_, ' with k = ', self.order, ' and q = ', q
        #     print 'condition number of feature matrix =', np.linalg.cond(
        #     feature_array)
