# Single-particle excitations

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import ncon as nc
import functools
import sys

# My scripts
import hamiltonians
from mps_tools import checks, HeffTerms

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
    lfp_AB = spspla.eigs(E, k=1, which="LR", tol=1e-14)[1].reshape(D, D)

    E = spspla.LinearOperator((D * D, D * D), matvec=right_transfer_op)
    rfp_AB = spspla.eigs(E, k=1, which="LR", tol=1e-14)[1].reshape(D, D)

    lfp_AB /= np.trace(lfp_AB @ rfp_AB)
    rfp_AB /= np.trace(lfp_AB @ rfp_AB)
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
    Y = Y.reshape(D * (d - 1), D)

    tensors = [VL, Y]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    B = nc.ncon(tensors, indices, contord)

    # print('left gauge check 1', 
    #       spla.norm(nc.ncon([B, AL.conj()], [(1, 2, -2), (1, 2, -1)]))
    #       )

    # print('left gauge check 2', 
    #       spla.norm(nc.ncon([AL, B.conj()], [(1, 2, -2), (1, 2, -1)]))
    #       )

    # print('perp to gs.', 
    #       nc.ncon([B, C.conj(), AR.conj()], [(2, 1, 4), (2, 3), (3, 1, 4)])
    #       )

    tensors = [B, AR.conj()]
    indices = [(-1, 1, 2), (-2, 1, 2)]
    contord = [2, 1]
    RB = nc.ncon(tensors, indices, contord)
    RB = right_vector_solver(RB, p)

    tensors = [Lh, B, AL.conj()]
    indices = [(2, 3), (3, 1, -2), (2, 1, -1)]
    contord = [2, 3, 1]
    L1_0 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, B, h, 
               AL.conj(), AL.conj(), AL.conj()]
    indices = [(9, 4, 10), (10, 5, 11), (11, 6, -2), (1, 2, 3, 4, 5, 6),
               (9, 1, 8), (8, 2, 7), (7, 3, -1)]
    contord = [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6]
    L1_1 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, AR, h, 
               AL.conj(), AL.conj(), AL.conj()]
    indices = [(9, 4, 10), (10, 5, 11), (11, 6, -2), (1, 2, 3, 4, 5, 6),
               (9, 1, 8), (8, 2, 7), (7, 3, -1)]
    contord = [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6]
    L1_2 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, AR, h, 
               AL.conj(), AL.conj(), AL.conj()]
    indices = [(9, 4, 10), (10, 5, 11), (11, 6, -2), (1, 2, 3, 4, 5, 6),
               (9, 1, 8), (8, 2, 7), (7, 3, -1)]
    contord = [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6]
    L1_3 = nc.ncon(tensors, indices, contord)

    L1 = (L1_0 
        + L1_1 
        + np.exp(-1j * p) * L1_2
        + np.exp(-2j * p) * L1_3
        )

    L1 = left_vector_solver(L1, p)

    tensors = [B, AR, AR, h, AR.conj(), AR.conj()]
    indices = [(-1, 4, 7), (7, 5, 8), (8, 6, 9), (-2, 2, 3, 4, 5, 6), 
               (-3, 2, 10), (10, 3, 9)]
    contord = [7, 8, 9, 10, 2, 3, 4, 5, 6]
    H_1 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, AR, h, AR.conj(), AR.conj()]
    indices = [(-1, 4, 7), (7, 5, 8), (8, 6, 9), (-2, 2, 3, 4, 5, 6), 
               (-3, 2, 10), (10, 3, 9)]
    contord = [7, 8, 9, 10, 2, 3, 4, 5, 6]
    H_2 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, B, h, AR.conj(), AR.conj()]
    indices = [(-1, 4, 7), (7, 5, 8), (8, 6, 9), (-2, 2, 3, 4, 5, 6), 
               (-3, 2, 10), (10, 3, 9)]
    contord = [7, 8, 9, 10, 2, 3, 4, 5, 6]
    H_3 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, AR, h, AL.conj(), AR.conj()]
    indices = [(7, 4, 8), (8, 5, 9), (9, 6, 10), (1, -2, 3, 4, 5, 6), 
               (7, 1, -1), (-3, 3, 10)]
    contord = [7, 8, 9, 10, 1, 3, 4, 5, 6]
    H_4 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, B, h, AL.conj(), AR.conj()]
    indices = [(7, 4, 8), (8, 5, 9), (9, 6, 10), (1, -2, 3, 4, 5, 6), 
               (7, 1, -1), (-3, 3, 10)]
    contord = [7, 8, 9, 10, 1, 3, 4, 5, 6]
    H_5 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, AR, h, AL.conj(), AR.conj()]
    indices = [(7, 4, 8), (8, 5, 9), (9, 6, 10), (1, -2, 3, 4, 5, 6), 
               (7, 1, -1), (-3, 3, 10)]
    contord = [7, 8, 9, 10, 1, 3, 4, 5, 6]
    H_6 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, B, h, AL.conj(), AL.conj()]
    indices = [(8, 4, 9), (9, 5, 10), (10, 6, -3), (1, 2, -2, 4, 5, 6), 
               (8, 1, 7), (7, 2, -1)]
    contord = [7, 8, 9, 10, 1, 2, 4, 5, 6]
    H_7 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, AR, h, AL.conj(), AL.conj()]
    indices = [(8, 4, 9), (9, 5, 10), (10, 6, -3), (1, 2, -2, 4, 5, 6), 
               (8, 1, 7), (7, 2, -1)]
    contord = [7, 8, 9, 10, 1, 2, 4, 5, 6]
    H_8 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, AR, h, AL.conj(), AL.conj()]
    indices = [(8, 4, 9), (9, 5, 10), (10, 6, -3), (1, 2, -2, 4, 5, 6), 
               (8, 1, 7), (7, 2, -1)]
    contord = [7, 8, 9, 10, 1, 2, 4, 5, 6]
    H_9 = nc.ncon(tensors, indices, contord)

    tensors = [B, Rh]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    H_10 = nc.ncon(tensors, indices, contord)

    tensors = [Lh, B]
    indices = [(-1, 1), (1, -2, -3)]
    contord = [1]
    H_11 = nc.ncon(tensors, indices, contord)

    tensors = [L1, AR]
    indices = [(-1, 1), (1, -2, -3)]
    contord = [1]
    H_12 = nc.ncon(tensors, indices, contord)

    tensors = [Lh, AL, RB]
    indices = [(-1, 1), (1, -2, 2), (2, -3)]
    contord = [1, 2]
    H_13 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, AL, RB, 
               h, AL.conj(), AL.conj()]
    indices = [(8, 4, 9), (9, 5, 10), (10, 6, 11), (11,-3), 
               (1, 2, -2, 4, 5, 6), (8, 1, 7), (7, 2, -1)]
    contord = [7, 8, 9, 10, 11, 1, 2, 4, 5, 6]
    H_14 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, AL, RB, 
               h, AL.conj(), AR.conj()]
    indices = [(7, 4, 8), (8, 5, 9), (9, 6, 10), (10, 11), 
               (1, -2, 3, 4, 5, 6), (7, 1, -1), (-3, 3, 11)]
    contord = [7, 8, 9, 10, 11, 1, 3, 4, 5, 6]
    H_15 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, AL, RB, h, 
               AR.conj(), AR.conj()]
    indices = [(-1, 4, 7), (7, 5, 8), (8, 6, 9), (9, 10),
               (-2, 2, 3, 4, 5, 6), (-3, 2, 11), (11, 3, 10)]
    contord = [7, 8, 9, 10, 11, 2, 3, 4, 5, 6]
    H_16 = nc.ncon(tensors, indices, contord)

    H = (H_1
       + np.exp(+1j * p) * H_2
       + np.exp(+2j * p) * H_3
       + H_4
       + np.exp(+1j * p) * H_5
       + np.exp(-1j * p) * H_6
       + H_7
       + np.exp(-1j * p) * H_8
       + np.exp(-2j * p) * H_9
       + H_10
       + H_11
       + np.exp(-1j * p) * H_12
       + np.exp(+1j * p) * H_13
       + np.exp(+1j * p) * H_14
       + np.exp(+2j * p) * H_15
       + np.exp(+3j * p) * H_16
       )

    tensors = [H, VL.conj()]
    indices = [(2, 1, -2), (2, 1, -1)]
    contord = [2, 1]
    Y = nc.ncon(tensors, indices, contord)
    return Y.ravel()

def quasi_particle(AL, AR, C, Lh, Rh, h, p, N):
    f = functools.partial(Heff, AL, AR, C, Lh, Rh, h, p)
    H = spspla.LinearOperator((D**2 * (d - 1), D**2 * (d - 1)), matvec=f)

    w, v = spspla.eigsh(H, k=N, which='SR', tol=tol)
    # print('norm check', 
    #       np.trace(v.reshape(D * (d - 1), D).conj().T 
    #              @ v.reshape(D * (d - 1), D)
    #              )
    #       )
    return w, v

def gs_energy(AL, AR, C, h):
    tensors = [AL, AL, C, AR, h.reshape(d, d, d, d, d, d), 
               AL.conj(), AL.conj(), C.conj(), AR.conj()]
    indices = [(7, 4, 8), (8, 5, 9), (9, 10), (10, 6, 11), (1, 2, 3, 4, 5, 6), 
               (7, 1, 14), (14, 2, 13), (13, 12), (12, 3, 11)]
    contord = [7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6] 
    return nc.ncon(tensors, indices, contord)

########################### Initialization #############################
excit_energy, excit_states, spectral_fxn = [], [], []

tol = 1e-12

model, d, D = str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

x, y, z = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

params = (model, z, D)

AL = np.loadtxt('%s_AL_%.2f_%03i_.txt' % params, dtype=complex)
AL = AL.reshape(d, D, D).transpose(1, 0, 2)

AR = np.loadtxt('%s_AR_%.2f_%03i_.txt' % params, dtype=complex)
AR = AR.reshape(d, D, D).transpose(1, 0, 2)

C = np.loadtxt('%s_C_%.2f_%03i_.txt' % params, dtype=complex)
C = C.reshape(D, D)

Lh = np.loadtxt('%s_Lh_%.2f_%03i_.txt' % params, dtype=complex)
Lh = Lh.reshape(D, D)

Rh = np.loadtxt('%s_Rh_%.2f_%03i_.txt' % params, dtype=complex)
Rh = Rh.reshape(D, D)

if model == 'halfXXZ':
    h = hamiltonians.XYZ_half(x, y, z, size='three')

if model == 'TFI':
    h = hamiltonians.TFI(y, z, size='three')

if model == 'oneXXZ':
    h = hamiltonians.XYZ_one(x, y, z, size='three')

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
print('gse', gs_energy(AL, AR, C, h))

######################### Pre-compute steps ############################
h -= np.real(gs_energy(AL, AR, C, h)) * np.eye(d**3)  # Regularize hamiltonian
h = h.reshape(d, d, d, d, d, d) 
print('reg. gse', gs_energy(AL, AR, C, h))

VL = spla.null_space(AL.conj().reshape(D * d, D).T)
VL = VL.reshape(D, d, D * (d - 1))

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
mom_vec = np.linspace(0, np.pi, 21)

for p in mom_vec:
    w, v = quasi_particle(AL, AR, C, Lh, Rh, h, p, N=1)

    excit_energy.append(w)
    excit_states.append(v)
    print('excit. energy', w[0]) # 0.410479248463


excit_energy = np.array(excit_energy)
print('all excit. energy', excit_energy.shape)
print('energy min', excit_energy.min())
print('energy max', excit_energy.max())

excit_states = np.array(excit_states)
print('all excit. states', excit_states.shape)

plt.title('s=1/2, %s, h=%.2f, D=%i ' % params)
# plt.ylabel('\u03C9 / 0.410479248463')

plt.plot(mom_vec, excit_energy, label ='approx')

# plt.plot(mom_vec, np.abs(np.cos(mom_vec - np.pi / 2)), label ='exact')

plt.plot(mom_vec, 
         2 * np.sqrt(z**2 + y**2 - 2 * z * y * np.cos(mom_vec)), 
         label='exact'
         )

# plt.plot(mom_vec, np.pi / 2 * np.abs(np.sin(mom_vec)), label='exact')

# plt.grid()
plt.legend()
plt.show()


















