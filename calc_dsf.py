import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import ncon as nc
import functools
import os
import sys
import matplotlib.colors as colors
from mps_tools import checks
import hamiltonians

def lorentzian(x, x0, gamma):
    return (1 / np.pi) * ((0.5 * gamma)/((x - x0)**2 + (0.5 * gamma)**2))

def calc_dsf(AL, AR, AC, 
             excit_energy, excit_states, 
             mom_vec, freq_vec, gamma, O):

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AL.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)

        if p == 0:
            XL = np.trace(X) * (C @ C.T.conj())
            return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()
        else:
            return (X - np.exp(+1.0j * p) * XT).ravel()
        
    print('<o>', nc.ncon([AC, O, AC.conj()], [[1, 3, 4], [2, 3], [1, 2, 4]]))

    O = (O 
         - nc.ncon([AC, O, AC.conj()], [[1, 3, 4], [2, 3], [1, 2, 4]]) 
           * np.eye(d)
           )

    dsf = []
    for i in range(mom_vec.size):
        p = mom_vec[i]
        print('p', p)

        dsf_p = np.zeros(freq_vec.size)
        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

            # print('left gauge check', spla.norm(nc.ncon([B, AL.conj()], [(1, 2, -2), (1, 2, -1)])))
            # print('left gauge check', spla.norm(nc.ncon([B, AC.conj()], [(1, 2, -2), (1, 2, -1)])))

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            right_env_op = spspla.LinearOperator((D * D, D * D), 
                                                 matvec=right_env
                                                 )

            RB = spspla.gmres(right_env_op, right_vec.ravel(), 
                                            x0=rand_init.ravel(), 
                                            tol=1e-14, 
                                            atol=1e-14
                                            )[0].reshape(D, D)

            tensors = [B, O, AC.conj()]
            indices = [(3, 2, 4), (1, 2), (3, 1, 4)]
            contord = [3, 4, 1, 2]
            t1 = nc.ncon(tensors, indices, contord)

            tensors = [AL, O, AL.conj(), RB]
            indices = [(3, 2, 4), (1, 2), (3, 1, 5), (4, 5)]
            contord = [4, 5, 3, 1, 2]
            t2 = nc.ncon(tensors, indices, contord)

            spec_weight = np.abs(t1 + np.exp(+1j * p) * t2)

            print('omega, s', excit_energy[i,j], spec_weight)

            lorentz_j = (2 * np.pi 
                         * lorentzian(freq_vec, excit_energy[i,j], gamma)
                         * spec_weight**2
                         )

            dsf_p += lorentz_j
        dsf.append(dsf_p)
    return np.array(dsf).reshape(mom_vec.size, freq_vec.size).T

############ Load data #############

model = str(sys.argv[1])

d = int(sys.argv[2])
D = int(sys.argv[3])

x = float(sys.argv[4])
y = float(sys.argv[5])
z = float(sys.argv[6])

N = int(sys.argv[7])
gamma = float(sys.argv[8])

params = (model, z, D)

path = '/Users/joshuabaktay/Desktop/local data/states'

filename = '%s_AL_%.2f_%03i_.txt' % params
AL = np.loadtxt(os.path.join(path, filename), dtype=complex)
AL = AL.reshape(d, D, D).transpose(1, 0, 2)

filename = '%s_AR_%.2f_%03i_.txt' % params
AR = np.loadtxt(os.path.join(path, filename), dtype=complex)
AR = AR.reshape(d, D, D).transpose(1, 0, 2)

filename = '%s_C_%.2f_%03i_.txt' % params
C = np.loadtxt(os.path.join(path, filename), dtype=complex)
C = C.reshape(D, D)

filename = '%s_disp_%.2f_%03i_%03i_.dat' % (*params, N)
disp = np.loadtxt(os.path.join(path, filename))
print(filename)

mom_vec = disp[:, 0]
excit_energy  = disp[:, 1:]

print('excit_energy', excit_energy.shape)
print('excit_energy min', excit_energy.min())
print('excit_energy max', excit_energy.max())

filename = '%s_estate_%.2f_%03i_%03i_.dat' % (*params, N)
excit_states = np.loadtxt(os.path.join(path, filename), dtype=complex)
excit_states = excit_states.reshape(excit_energy.shape[0], 
                                    (d - 1) * D**2, 
                                    excit_energy.shape[1]
                                    )

print('excit_states', excit_states.shape)
print(filename)

checks(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)

if d == 3:
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # gets 1/sqrt2
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) # gets 1/sqrt2
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

############ Precompute steps ############

VL = spla.null_space(AL.conj().reshape(D * d, D).T)
VL = VL.reshape(D, d, (d - 1) * D)

freq_min = excit_energy.min() - (3 * gamma)
freq_max = excit_energy.max() + (3 * gamma)

num = int(np.ceil(5 * ((freq_max - freq_min) / gamma)))

freq_vec = np.linspace(freq_min, freq_max, num)

print('gamma', gamma)
print('freq vec size', freq_vec.size)
print('delta omega', freq_vec[1] - freq_vec[0])

dsf = calc_dsf(AL, AR, np.tensordot(AL, C, axes=(2, 0)), 
               excit_energy, excit_states, 
               mom_vec, freq_vec, gamma, sx
               )

print(dsf.shape)

fig, ax = plt.subplots()
A, B = np.meshgrid(mom_vec / np.pi, freq_vec)

Z = dsf

print(Z.min(), Z.max())

Z_min, Z_max = Z.min(), Z.max()
Z = (Z - Z_min) / (Z_max - Z_min)

print(Z.min(), Z.max())

plot = ax.contourf(A, B, Z, levels=100)

cbar = fig.colorbar(plot, format='%.3f')
cbar.ax.set_ylabel('S(q, \u03C9)')
ax.set_xlabel('q')
ax.set_ylabel('\u03C9')
# ax.set_ylim((0.3, 3))
plt.show()

exit()

filename = '%s_t_%.2f_%03i_%03i_%.2f_.txt' % (*params, N, gamma)
np.savetxt(filename,
           np.column_stack((freq_vec, dsf)))



















