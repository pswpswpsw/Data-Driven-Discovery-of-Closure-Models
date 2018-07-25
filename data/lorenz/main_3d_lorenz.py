"""
code for generating data and closure data for lorenz system

x_1, x_2, as resolved
x_3 as unresolved

x_1^{n+1} = x_1^{n} + dt*\sigma(x_2 - x_1)
x_2^{n+1} = x_2^{n} + dt*((x_1*(\rho - x_3) - x_2)
x_3^{n+1} = x_3^{n} + dt*(x_1*x_2 - \beta x_3)

with closure defined as -x_1*x_3

"""
import os

import numpy as np
import scipy.sparse.linalg
from matplotlib import pyplot as plt
from nolds import *

plt.style.use('siads')

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# dimension of reduced system
r_partition = 1


case = 'chaos' #
# case = 'equ'

if case == 'chaos':
    # total snapshots collected
    ntsnap = 40000
    # total time
    tot = 400.0
    # parameter for lorenz system
    sigma = 10
    beta = 8.0 / 3.0
    rho = 35  # chaotic: 28, equilibrum: 0.5
    tsnap = np.linspace(0,tot,ntsnap,endpoint=False)
    dt = tsnap[2] - tsnap[1]
    ## off attractor IC
    x0 = np.array([0.5, 0, 0])
    ## on attractor IC
    # x0 = np.array( [-2.10565328e-02, -8.51142795e+00,  3.21024403e+01]  )
else:
    # total snapshots collected
    ntsnap = 8000
    # total time
    tot = 20.0
    # parameter for lorenz system
    sigma = 10
    beta = 8.0 / 3.0
    rho = 15  # chaotic: 28, equilibrum: 0.5
    tsnap = np.linspace(0,tot,ntsnap,endpoint=False)
    dt = tsnap[2] - tsnap[1]
    x0 = np.array([0.5,0,0])

def F(x):
    FX = [sigma * (x[1] - x[0]),
          x[0] * (rho - x[2]) - x[1],
          x[0] * x[1] - beta *x[2]]

    return np.array(FX)

# matrix exponential
xsnap = np.zeros((ntsnap, 3))
closure = np.zeros((ntsnap, r_partition))

if r_partition == 2:
    closure[0, :] = np.array([0, -(xsnap[0, 2] * xsnap[0, 0])])
elif r_partition == 1:
    closure[0, :] = np.array([sigma * xsnap[0, 1]])


xsnap[0,:] = x0

for i_time in range(1, ntsnap):
    # 1. first order euler
    xsnap[i_time, :] = xsnap[i_time-1, :] + dt*F(xsnap[i_time-1, :])

    # record closure as well
    if r_partition == 2:
        closure[i_time, :] = np.array([0, -(xsnap[i_time, 2]*xsnap[i_time, 0])])
    elif r_partition == 1:
        closure[i_time, :] = np.array([sigma * xsnap[i_time, 1]])

# validation on reduced system with closure
if r_partition == 2:
    A11 = np.array([[-sigma, sigma], [rho, -1]])
elif r_partition == 1:
    A11 = np.array([[-sigma]])

xsnap_validation = np.zeros((ntsnap, r_partition))
xsnap_validation[0, :] = x0[0:r_partition]
for i_time in range(1, ntsnap):
    # 1. first order euler on resolved flux
    xsnap_validation[i_time, :] = xsnap_validation[i_time-1, :] + \
                                  dt*np.dot(xsnap_validation[i_time-1, :], A11.transpose())

    # 2. add closure
    xsnap_validation[i_time, :] = xsnap_validation[i_time, :] + \
                                  dt*closure[i_time-1, :]



plt.plot(xsnap[:,0],'b-')
plt.plot(xsnap_validation[:,0],'r--')
plt.ylim([-16,32])
plt.savefig('test.png')
plt.close()


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
plt.savefig('./image/rho_' + str((rho)) + '_lorenz_system_full.png',
            bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

# save state data
# first need to cut to a LES
xsnap_les = xsnap[: ,0:r_partition]
# need to transpose to (1,4000) for consistence
xsnap_save = xsnap_les.transpose()
np.save('./data/physical_snapshots_resolved_3d_lorenz_ntsnap_'  + 'case_' + case + '_' + \
        str(ntsnap) + '_tot_' + str(int(tot)) + '_reduced_modes_' + str(r_partition) + '.npy', xsnap_save)

# save closure+state data
closure_save = closure.transpose()
np.savez('./data/closure_3d_lorenz_ntsnap_' + 'case_' + case + '_' + \
         str(ntsnap) + '_tot_' + str(int(tot)) +  '_reduced_modes_' + str(r_partition) +  '.npz', usnap_les=xsnap_save, ec_snap=closure_save)

# print
print xsnap_save.shape
print closure_save.shape



xsnap = xsnap[:8000,0]

## compute statiists
embed_dim = 3
# compute Lyapounov exponent
lya = max(lyap_e(xsnap.flatten(),embed_dim, matrix_dim=embed_dim))

# compute correlation dimension
cd = corr_dim(xsnap.flatten(), embed_dim)

print 'max Lyapunov: ', lya
print 'correlation dimension: ', cd

