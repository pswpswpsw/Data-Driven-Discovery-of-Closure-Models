"""
code for generating data and closure data for 3D linear system with

x_1 as resolved
x_2 as unresolved

A = | 0     -1  -1 |
    | 0.5 -1.1 1.5 |
    | 1     -3 0.5 |

x0 = [3 0 0]

with 4000 snapshots and tot = 40

"""
import os

import numpy as np
import scipy.sparse.linalg
from matplotlib import pyplot as plt

plt.style.use('siads')





def main():
    """
    simulation of linear system evolution
    :return:
    """
    # total snapshots collected
    ntsnap = 4000

    # total time
    tot = 40.0

    tsnap = np.linspace(0,tot,ntsnap,endpoint=False)
    dt = tsnap[2] - tsnap[1]

    x0 = np.array([3,0,0])

    # full matrix A
    A = np.array([[0, -1, -1],[0.5, -1.1, 1.5],[1, -3, 0.5]])

    # A12
    A12 = np.array([[-1, -1]])

    # matrix exponential
    xsnap = np.zeros((ntsnap, 3))
    closure = np.zeros((ntsnap, 1))
    xsnap[0,:] = x0
    for i_time in range(1, ntsnap):
        t = tsnap[i_time]
        # 1. exponential intergrator
        # xsnap[i_time, :] = np.dot(scipy.sparse.linalg.expm(t*A),x0)
        # 2. first order euler
        xsnap[i_time, :] = xsnap[i_time - 1] + np.dot(A, dt*xsnap[i_time - 1])

        x_unresolved = xsnap[i_time, 1:]
        closure[i_time, :] = np.dot(A12, x_unresolved)

    # mkdir
    mkdir('data')
    mkdir('image')

    # plot full
    plt.figure()
    plt.plot(tsnap, xsnap[:,0], 'k-', label='$x_1$')
    plt.plot(tsnap, xsnap[:,1], 'r-', label='$x_2$')
    plt.plot(tsnap, xsnap[:,2], 'b-', label='$x_3$')
    lgd = plt.legend(bbox_to_anchor=(1, 0.5))
    plt.xlabel('time')
    plt.ylabel('component value of $x$')
    plt.savefig('./image/linear_system_full.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # save state data
    # first need to cut to a LES
    xsnap_les = xsnap[:,0:1]
    # need to transpose to (1,4000) for consistence
    xsnap_save = xsnap_les.transpose()
    np.save('./data/physical_snapshots_resolved_3d_linear_ntsnap_' + \
            str(ntsnap) + '_tot_' + str(int(tot)) + '.npy', xsnap_save)

    # save closure+state data
    closure_save = closure.transpose()
    np.savez('./data/closure_3d_linear_ntsnap_' + \
             str(ntsnap) + '_tot_' + str(int(tot)) + '.npz', usnap_les=xsnap_save, ec_snap=closure_save)

    # print
    print xsnap_save.shape
    print closure_save.shape

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    main()
