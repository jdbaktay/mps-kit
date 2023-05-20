# Single-particle excitations

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import ncon as nc
import functools
import sys
import os

# My scripts
import hamiltonians
from mps_tools import checks, HeffTerms_two

def fixed_points(A, B):
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), B.conj()]
        indices = [(2, 1, -2), (3, 2), (3, 1, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    def right_transfer_op(X):
        tensors = [A, X.reshape(D, D), B.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    lfp_AB = spspla.eigs(E, k=1, which='LR', tol=1e-14)[1].reshape(D, D)

    E = spspla.LinearOperator((D * D, D * D), matvec=right_transfer_op)
    rfp_AB = spspla.eigs(E, k=1, which='LR', tol=1e-14)[1].reshape(D, D)

    norm = np.trace(lfp_AB @ rfp_AB)

    lfp_AB /= np.sqrt(norm)
    rfp_AB /= np.sqrt(norm)
    return lfp_AB, rfp_AB

def left_vector_solver(O, p):
    def left_env(X):
        X = X.reshape(D, D)

        tensors = [AR, X, AL.conj()]
        indices = [(3, 1, -2), (2, 3), (2, 1, -1)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)
        XR = np.trace(X @ rfp_RL) * lfp_RL
        return (X - np.exp(-1.0j * p) * (XT - XR)).ravel()

    left_env_op = spspla.LinearOperator((D * D, D * D), matvec=left_env)

    v, _ = spspla.gmres(left_env_op, 
                        O.ravel(), 
                        x0=(np.random.rand(D, D) - 0.5).ravel(), 
                        tol=tol, 
                        atol=tol
                        )
    return v.reshape(D, D)

def right_vector_solver(O, p):
    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AR.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)
        XL = np.trace(lfp_LR @ X) * rfp_LR
        return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()

    right_env_op = spspla.LinearOperator((D * D, D * D), matvec=right_env)

    v, _ = spspla.gmres(right_env_op, 
                        O.ravel(), 
                        x0=(np.random.rand(D, D) - 0.5).ravel(), 
                        tol=tol, 
                        atol=tol
                        )
    return v.reshape(D, D)

def Heff(AL, AR, C, Lh, Rh, h, p, Y):
    Y = Y.reshape((d - 1) * D, D)

    tensors = [VL, Y]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    B = nc.ncon(tensors, indices, contord)

    tensors = [B, AR.conj()]
    indices = [(-1, 1, 2), (-2, 1, 2)]
    contord = [2, 1]
    RB = nc.ncon(tensors, indices, contord)
    RB = right_vector_solver(RB, p)

    tensors = [Lh, B, AL.conj()]
    indices = [(2, 3), (3, 1, -2), (2, 1, -1)]
    contord = [2, 3, 1]
    L1_1 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AL.conj(), AL.conj()]
    indices = [(6, 3, 7), (7, 4, -2), (1, 2, 3, 4), (6, 1, 5), (5, 2, -1)]
    contord = [5, 6, 7, 1, 2, 3, 4]
    L1_2 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, h, AL.conj(), AL.conj()]
    indices = [(6, 3, 7), (7, 4, -2), (1, 2, 3, 4), (6, 1, 5), (5, 2, -1)]
    contord = [5, 6, 7, 1, 2, 3, 4]
    L1_3 = nc.ncon(tensors, indices, contord)

    L1 = L1_1 + L1_2 + (L1_3 * np.exp(-1.0j * p))
    L1 = left_vector_solver(L1, p)

    tensors = [B, AR, h, AR.conj()]
    indices = [(-1, 3, 5), (5, 4, 6), (-2, 2, 3, 4), (-3, 2, 6)]
    contord = [5, 6, 2, 3, 4]
    H_1 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, h, AL.conj()]
    indices = [(5, 3, 6), (6, 4, -3), (1, -2, 3, 4), (5, 1, -1)]
    contord = [5, 6, 1, 3, 4]
    H_2 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AR.conj()]
    indices = [(-1, 3, 5), (5, 4, 6), (-2, 2, 3, 4), (-3, 2, 6)]
    contord = [5, 6, 2, 3, 4]
    H_3 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AL.conj()]
    indices = [(5, 3, 6), (6, 4, -3), (1, -2, 3, 4), (5, 1, -1)]
    contord = [5, 6, 1, 3, 4]
    H_4 = nc.ncon(tensors, indices, contord)

    tensors = [B, Rh]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    H_5 = nc.ncon(tensors, indices, contord)

    tensors = [Lh, B]
    indices = [(-1, 1), (1, -2, -3)]
    contord = [1]
    H_6 = nc.ncon(tensors, indices, contord)

    tensors = [L1, AR]
    indices = [(-1, 1), (1, -2, -3)]
    contord = [1]
    H_789 = nc.ncon(tensors, indices, contord)

    tensors = [Lh, AL, RB]
    indices = [(-1, 1), (1, -2, 2), (2, -3)]
    contord = [1, 2]
    H_10 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, RB, h, AL.conj()]
    indices = [(5, 3, 6), (6, 4, 7), (7, -3), (1, -2, 3, 4), (5, 1, -1)]
    contord = [5, 6, 7, 1, 3, 4]
    H_11 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, RB, h, AR.conj()]
    indices = [(-1, 3, 5), (5, 4, 6), (6, 7), (-2, 2, 3, 4), (-3, 2, 7)]
    contord = [5, 6, 7, 2, 3, 4]
    H_12 = nc.ncon(tensors, indices, contord)

    H = (H_1 
       + np.exp(-1.0j * p) * H_2
       + np.exp(+1.0j * p) * H_3
       + H_4
       + H_5
       + H_6
       + np.exp(-1.0j * p) * H_789
       + np.exp(+1.0j * p) * H_10
       + np.exp(+1.0j * p) * H_11
       + np.exp(+2.0j * p) * H_12
       )

    tensors = [H, VL.conj()]
    indices = [(2, 1, -2), (2, 1, -1)]
    contord = [2, 1]
    Y = nc.ncon(tensors, indices, contord)
    return Y.ravel()

def quasi_particle(AL, AR, C, Lh, Rh, h, p, N, guess):
    f = functools.partial(Heff, AL, AR, C, Lh, Rh, h, p)
    H = spspla.LinearOperator(((d - 1) * D**2, (d - 1) * D**2), matvec=f)

    w, v = spspla.eigsh(H, k=N, v0=guess, which='SR', tol=tol)
    return w, v

def gs_energy(AL, AR, C, h):
    tensors = [AL, C, AR, h.reshape(d, d, d, d), 
               AL.conj(), C.conj(), AR.conj()]
    indices = [(5, 3, 6),  (6, 7),  (7, 4, 8), (1, 2, 3, 4), 
               (5, 1, 10), (10, 9), (9, 2, 8)]
    contord = [5, 6, 7, 8, 9, 10, 1, 2, 3, 4] 
    return nc.ncon(tensors, indices, contord)

########################### Initialization #############################
excit_energy, excit_states = [], []

tol = 1e-12

model, d, D = str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
x, y, z = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
g = float(sys.argv[7])
N = int(sys.argv[8])

params = (model, x, y, z, g, D)

path = '' #'/Users/joshuabaktay/Desktop/local data/states'

filename = '%s_AL_%.2f_%.2f_%.2f_%.2f_%03i_.txt' % params
AL = np.loadtxt(os.path.join(path, filename), dtype=complex)
AL = AL.reshape(d, D, D).transpose(1, 0, 2)

filename = '%s_AR_%.2f_%.2f_%.2f_%.2f_%03i_.txt' % params
AR = np.loadtxt(os.path.join(path, filename), dtype=complex)
AR = AR.reshape(d, D, D).transpose(1, 0, 2)

filename = '%s_C_%.2f_%.2f_%.2f_%.2f_%03i_.txt' % params
C = np.loadtxt(os.path.join(path, filename), dtype=complex)
C = C.reshape(D, D)

Lh, Rh = np.eye(D, dtype=AL.dtype), np.eye(D, dtype=AR.dtype)

if model == 'halfXXZ':
    h = hamiltonians.XYZ_half(x, y, z, g, size='two')

if model == 'TFI':
    h = hamiltonians.TFI(x, g, size='two')

if model == 'oneXXZ':
    h = hamiltonians.XYZ_one(x, y, z, size='two')

if model == 'tV':
    h = hamiltonians.tV(1, z, g)

checks(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)
print('gse', gs_energy(AL, AR, C, h))

######################### Pre-compute steps ############################
Lh, Rh, _ = HeffTerms_two(AL, AR, C, Lh, Rh, h, tol)

h -= np.real(gs_energy(AL, AR, C, h)) * np.eye(d**2)  # Regularize hamiltonian
h = h.reshape(d, d, d, d) 
print('reg. gse', gs_energy(AL, AR, C, h))

VL = spla.null_space(AL.conj().reshape(D * d, D).T)

print('VL', VL.shape)

VL = VL.reshape(D, d, (d - 1) * D)

print('null check 1', 
      spla.norm(nc.ncon([VL, AL.conj()], [(1, 2, -2), (1, 2, -1)]))
      )

print('null check 2',
    spla.norm(nc.ncon([VL, VL.conj()], [(1, 2, -2), (1, 2, -1)])
               - np.eye(D * (d - 1)))
    )

lfp_LR, rfp_LR = fixed_points(AL, AR)
lfp_RL, rfp_RL = fixed_points(AR, AL)

######################### Compute excitations ##########################
mom_vec = np.linspace(0, np.pi, 5)

for p in mom_vec:
    print('p', p)
    guess = v[:, 0] if p > 0 else None
    w, v = quasi_particle(AL, AR, C, Lh, Rh, h, p, N=N, guess=guess)

    excit_energy.append(w)
    excit_states.append(v)
    print('excit. energy', min(w))

excit_energy = np.array(excit_energy)
print('all excit. energy', excit_energy.shape)
print('energy min', excit_energy.min())
print('energy max', excit_energy.max())

excit_states = np.array(excit_states)
print('all excit. states', excit_states.shape)

filename = '%s_disp_%.2f_%.2f_%.2f_%.2f_%03i_%05i_.dat' % (*params, N)
with open(os.path.join(path, filename), 'w') as outfile:
    np.savetxt(os.path.join(path, filename),
               np.column_stack((mom_vec, excit_energy))
               )

filename = '%s_estate_%.2f_%.2f_%.2f_%.2f_%03i_%05i_.dat' % (*params, N)
with open(os.path.join(path, filename), 'w') as outfile:
    for data_slice in excit_states:
        np.savetxt(outfile, data_slice)
