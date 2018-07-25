"""
code for generating data and closure data
     for 2D nonlinear VDP system

     with x_1 as resolved and x_2 as unresolved.

 | x_1^{n+1} | = | x_1^{n} | + dt*|  x_2^{n}                           |
 | x_2^{n+1} |   | x_2^{n} |      |  \nu(1-x_1^{n}^2)x_2^{n} - x_1^{n} |

with 6000 snapshots and tot = 60, dt = 2
"""
import os

import numpy as np
import scipy.sparse.linalg
from matplotlib import pyplot as plt

plt.style.use('siads')





def main():
    """
    simulation of 2D vdp system
    and providing closure for x_1

    :return:
    """
    # total snapshots collected
    ntsnap = 6000

    # total time
    tot = 60.0

    tsnap = np.linspace(0, tot, ntsnap,endpoint=False)
    dt = tsnap[2] - tsnap[1]

    print 'dt = ', dt


    x0 = np.array([1,0])

    def F_vdp(x):
        """
        Function for vdp

        :param x:
        :return:
        """
        nu = 2
        F = np.zeros(x.shape)
        F[0] = x[1]
        F[1] = nu*(1-x[0]*x[0])*x[1] - x[0]
        return F

    # A12
    A12 = np.array([[-1, -1]])

    # matrix exponential
    xsnap = np.zeros((ntsnap, 2))
    closure = np.zeros((ntsnap, 1))
    xsnap[0,:] = x0
    for i_time in range(1, ntsnap):

        # first order euler
        xsnap[i_time, :] = xsnap[i_time - 1, :] + dt*F_vdp(xsnap[i_time - 1, :])
        closure[i_time, :] = xsnap[i_time, 1] # F_vdp(xsnap[i_time - 1])[0]

    # mkdir
    mkdir('data')
    mkdir('image')

    # plot full
    plt.figure()
    plt.plot(tsnap, xsnap[:,0], 'k-', label='$x_1$')
    plt.plot(tsnap, xsnap[:,1], 'r-', label='$x_2$')
    lgd = plt.legend(bbox_to_anchor=(1, 0.5))
    plt.xlabel('time')
    plt.ylabel('component value of $x$')
    plt.savefig('./image/2d_vdp_full.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # save state data
    # first need to cut to a LES
    xsnap_les = xsnap[:,0:1]
    # need to transpose to (1,4000) for consistence
    xsnap_save = xsnap_les.transpose()
    np.save('./data/physical_snapshots_resolved_2d_vdp_ntsnap_' + \
            str(ntsnap) + '_tot_' + str(int(tot)) + '.npy', xsnap_save)

    # save closure+state data
    closure_save = closure.transpose()
    np.savez('./data/closure_2d_vdp_ntsnap_' + \
             str(ntsnap) + '_tot_' + str(int(tot)) + '.npz', usnap_les=xsnap_save, ec_snap=closure_save)

    # print
    print xsnap_save.shape
    print closure_save.shape

    
    ##############################################################
    # debug phase: check MSE of dy/dt vs analytical expression
    ## target without last term
    target = (closure[1:,:] - closure[:-1,:])/dt
    eff_states = xsnap_les[:-1,:]
    eff_closure = closure[:-1,:]

    ## analytically:
    nu = 2.0
    
    analytical_prediction = nu*eff_closure - nu*eff_states*eff_states*eff_closure - eff_states
    

    print 'mean squared error on whole data =', np.square(analytical_prediction-target).mean()

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    main()
