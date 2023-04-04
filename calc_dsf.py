import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import ncon as nc
import functools
import os
import sys

from mps_tools import checks

def lorentzian(x, x0, gamma):
    return (1 / np.pi) * ((0.5 * gamma)/((x - x0)**2 + (0.5 * gamma)**2))

def calc_dsf(AL, AR, AC, 
             excit_energy, excit_states, 
             mom_vec, freq_vec, gamma, O):

    def left_env(X):
        X = X.reshape(D, D)

        tensors = [AR, X, AR.conj()]
        indices = [(3, 1, -2), (2, 3), (2, 1, -1)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)
        XR = np.trace(X @ C @ C.T.conj()) * np.eye(D)
        return (X - np.exp(-1.0j * p) * (XT - XR)).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AL.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)
        XL = np.trace(C.T.conj() @ C @ X) * np.eye(D)
        return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()

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

            print('left gauge check', spla.norm(nc.ncon([B, AL.conj()], [(1, 2, -2), (1, 2, -1)])))

            print('left gauge check', spla.norm(nc.ncon([B, AC.conj()], [(1, 2, -2), (1, 2, -1)])))


            tensors = [B, AC.conj()]
            indices = [(1, 2, -2), (1, 2, -1)]
            contord = [1, 2]
            left_vec = nc.ncon(tensors, indices, contord)

            print('left vec', spla.norm(left_vec))

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            left_env_op = spspla.LinearOperator((D * D, D * D), 
                                                matvec=left_env
                                                )

            LB = spspla.gmres(left_env_op, left_vec.ravel(), 
                                           x0=rand_init.ravel(), 
                                           tol=tol, 
                                           atol=tol
                                           )[0].reshape(D, D)

            right_env_op = spspla.LinearOperator((D * D, D * D), 
                                                 matvec=right_env
                                                 )

            RB = spspla.gmres(right_env_op, right_vec.ravel(), 
                                            x0=rand_init.ravel(), 
                                            tol=tol, 
                                            atol=tol
                                            )[0].reshape(D, D)

            tensors = [B, O, AC.conj()]
            indices = [(3, 2, 4), (1, 2), (3, 1, 4)]
            contord = [3, 4, 1, 2]
            t1 = nc.ncon(tensors, indices, contord)

            tensors = [AL, O, AL.conj(), RB]
            indices = [(3, 2, 4), (1, 2), (3, 1, 5), (4, 5)]
            contord = [4, 5, 3, 1, 2]
            t2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, O, AR.conj()]
            indices = [(4, 5), (5, 2, 3), (1, 2), (4, 1, 3)]
            contord = [3, 4, 5, 1, 2]
            t3 = nc.ncon(tensors, indices, contord)

            print('t3', spla.norm(t3))

            spec_weight = np.abs(t1 
                               + np.exp(+1j * p) * t2 
                               + np.exp(-1j * p) * t3
                               )

            lorentz_j = (2 * np.pi 
                         * lorentzian(freq_vec, excit_energy[i,j], gamma)
                         * spec_weight**2
                         )

            dsf_p += lorentz_j

        dsf.append(dsf_p)
    return np.array(dsf).reshape(mom_vec.size, freq_vec.size).T

tol = 1e-12

model, d, D = str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

x, y, z = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

N = int(sys.argv[7])

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

filename = '%s_disp_%.2f_%03i_%03i_.dat' % (*params, N); print(filename)
disp = np.loadtxt(os.path.join(path, filename), dtype=complex)

mom_vec = np.real((disp[:, 0]))
excit_energy  = np.real(disp[:, 1:])

print('excit_energy', excit_energy.shape)
print('excit_energy min', excit_energy.min())
print('excit_energy max', excit_energy.max())

filename = '%s_estate_%.2f_%03i_%03i_.dat' % (*params, N); print(filename)
excit_states = np.loadtxt(os.path.join(path, filename), dtype=complex)
excit_states = excit_states.reshape(excit_energy.shape[0], 
                                    (d - 1) * D**2, 
                                    excit_energy.shape[1]
                                    )

print('excit_states', excit_states.shape)

if d == 2:
    si = np.array([[1, 0],[0, 1]])
    sx = np.array([[0, 1],[1, 0]])
    sy = np.array([[0, -1j],[1j, 0]])
    sz = np.array([[1, 0],[0, -1]])

if d == 3:
    si = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sx = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]) 
    sy = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]]) 
    sz = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]) 

sp = 0.5 * (sx + 1.0j * sy)
sm = 0.5 * (sx - 1.0j * sy)
n = 0.5 * (sz + np.eye(d))

checks(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)

VL = spla.null_space(AL.conj().reshape(D * d, D).T)
VL = VL.reshape(D, d, (d - 1) * D)

print('null check 1', 
      spla.norm(nc.ncon([VL, AL.conj()], [(1, 2, -2), (1, 2, -1)]))
      )

print('null check 2',
    spla.norm(nc.ncon([VL, VL.conj()], [(1, 2, -2), (1, 2, -1)])
               - np.eye(D * (d - 1)))
    )

gamma = 0.05

q = 4 * gamma

num = ((excit_energy.max() + q) - (excit_energy.min() - q)) / gamma
num = np.ceil(3 * num)

freq_vec = np.linspace(excit_energy.min() - q, 
                       excit_energy.max() + q, 
                       int(num)
                       )

print('gamma', gamma)
print('freq vec size', freq_vec.size)
print('delta omega', freq_vec[1] - freq_vec[0])

AC = np.tensordot(AL, C, axes=(2, 0))
print('AC', AC.shape)

dsf = calc_dsf(AL, AR, AC, 
               excit_energy, excit_states, 
               mom_vec, freq_vec, gamma, sx
               )

exit()

print('dsf', dsf.shape)

filename = '%s_dsf_%.2f_%03i_%03i_%.2f_.txt' % (*params, N, gamma)
np.savetxt(filename,
           np.column_stack((freq_vec, dsf)))



















