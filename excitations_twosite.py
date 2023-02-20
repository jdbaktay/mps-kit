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

def Heff(AL, AR, C, Lh, Rh, LR_block_inv, RL_block_inv, h, p, Y):
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

    ################# 12 diagrams ######################################
    tensors = [np.eye(D), AR, h, AR.conj()]
    indices = [(-1, -4), (-6, 4, 6), (-2, 2, -5, 4), (-3, 2, 6)]
    H_1 = nc.ncon(tensors, indices).reshape(D * d * D, D * d * D)

    tensors = [AR, h, AL.conj()]
    indices = [(-6, 4, -3), (1, -2, -5, 4), (-4, 1, -1)]
    H_2 = nc.ncon(tensors, indices).reshape(D * d * D, D * d * D)

    tensors = [AL, h, AR.conj()]
    indices = [(-1, 3, -4), (-2, 2, 3, -5), (-3, 2, -6)]
    H_3 = nc.ncon(tensors, indices).reshape(D * d * D, D * d * D)

    tensors = [AL, np.eye(D), h, AL.conj()]
    indices = [(5, 3, -4), (-6, -3), (1, -2, 3, -5), (5, 1, -1)]
    H_4 = nc.ncon(tensors, indices).reshape(D * d * D, D * d * D)

    tensors = [np.eye(D), np.eye(d), Rh]
    indices = [(-1, -4), (-2, -5), (-6, -3)]
    H_5 = nc.ncon(tensors, indices).reshape(D * d * D, D * d * D)

    tensors = [Lh, np.eye(d), np.eye(D)]
    indices = [(-1, -4), (-2, -5), (-6, -3)]
    H_6 = nc.ncon(tensors, indices).reshape(D * d * D, D * d * D)

    tensors = [Lh, AL.conj(), RL_block_inv, AR]
    indices = [(6, -4), (6, -5, 5), (-6, 5, 7, -1), (7, -2, -3)]
    contord = [6, 5, 7]
    H_7 = nc.ncon(tensors, indices, contord).reshape(D * d * D, D * d * D)

    tensors = [AL, h, AL.conj(), AL.conj(),
               RL_block_inv, AR]
    indices = [(7, 3, -4), (1, 2, 3, -5), (7, 1, 6), (6, 2, 5),
               (-6, 5, 8, -1), (8, -2, -3)]
    contord = [7, 6, 5, 8, 1, 2, 3]
    H_8 = nc.ncon(tensors, indices, contord).reshape(D * d * D, D * d * D)

    tensors = [AR, h, AL.conj(), AL.conj(),
               RL_block_inv, AR]
    indices = [(-6, 4, 7), (1, 2, -5, 4), (-4, 1, 6), (6, 2, 5),
               (7, 5, 8, -1), (8, -2, -3)]
    contord = [6, 5, 7, 8, 1, 2, 4]
    H_9 = nc.ncon(tensors, indices, contord).reshape(D * d * D, D * d * D)

    tensors = [Lh, AL, LR_block_inv, AR.conj()]
    indices = [(-1, 5), (5, -2, 6), (6, -3, -4, 7), (7, -5, -6)]
    contord = [5, 6, 7]
    H_10 = nc.ncon(tensors, indices, contord).reshape(D * d * D, D * d * D)

    tensors = [AL, AL, h, AL.conj(),
               LR_block_inv, AR.conj()]
    indices = [(5, 3, 6), (6, 4, 7), (1, -2, 3, 4), (5, 1, -1),
               (7, -3, -4, 8), (8, -5, -6)]
    contord = [5, 6, 7, 8, 1, 3, 4]
    H_11 = nc.ncon(tensors, indices, contord).reshape(D * d * D, D * d * D)

    tensors = [AL, AL, h, AR.conj(), 
               LR_block_inv, AR.conj()]
    indices = [(-1, 3, 5), (5, 4, 6), (-2, 2, 3, 4), (-3, 2, 7),
               (6, 7, -4, 8), (8, -5, -6)]
    contord = [5, 6, 7, 8, 2, 3 ,4]
    H_12 = nc.ncon(tensors, indices, contord).reshape(D * d * D, D * d * D)
    ####################################################################

    H = (H_1 
       + np.exp(-1.0j * p) * H_2
       + np.exp(+1.0j * p) * H_3
       + H_4
       + H_5
       + H_6
       + np.exp(-1.0j * p) * H_7
       + np.exp(-1.0j * p) * H_8
       + np.exp(-2.0j * p) * H_9
       + np.exp(+1.0j * p) * H_10
       + np.exp(+1.0j * p) * H_11
       + np.exp(+2.0j * p) * H_12
       )

    if p == 0:
        print('H == H+', spla.norm(H - H.conj().T))

    HB = H @ B.ravel()
    HB = HB.reshape(D, d, D)

    tensors = [HB, VL.conj()]
    indices = [(2, 1, -2), (2, 1, -1)]
    contord = [2, 1]
    Y = nc.ncon(tensors, indices, contord)
    return Y.ravel()

def quasi_particle(AL, AR, C, Lh, Rh, h, p, N):
    LR_block = (np.eye(D**2).reshape(D, D, D, D) 
                 - np.exp(+1.0j * p) * E_LR
                 )

    LR_block_inv = spla.inv(LR_block.reshape(D**2, D**2))
    LR_block_inv = LR_block_inv.reshape(D, D, D, D)

    RL_block = (np.eye(D**2).reshape(D, D, D, D) 
                - np.exp(-1.0j * p) * E_RL
                )

    RL_block_inv = spla.inv(RL_block.reshape(D**2, D**2))
    RL_block_inv = RL_block_inv.reshape(D, D, D, D) 

    f = functools.partial(Heff, AL, AR, C, Lh, Rh,
                          LR_block_inv, RL_block_inv, h, p)
    H = spspla.LinearOperator((D**2 * (d - 1), D**2 * (d - 1)), matvec=f)

    w, v = spspla.eigsh(H, k=N, which='SR', tol=tol)
    # print('norm check', 
    #       np.trace(v.reshape(D, D).conj().T @ v.reshape(D, D))
    #       )
    return w, v

def calc_nullspace(n):
    u, s, vh = spla.svd(n, full_matrices=True)
    print('null check', vh.conj().T[:, :D * (d - 1)].shape)
    return vh.conj().T[:, D * (d - 1):]

def gs_energy(AL, AR, C, h):
    tensors = [AL, C, AR, h.reshape(d, d, d, d), 
               AL.conj(), C.conj(), AR.conj()]
    indices = [(5, 3, 6),  (6, 7),  (7, 4, 8), (1, 2, 3, 4), 
               (5, 1, 10), (10, 9), (9, 2, 8)]
    contord = [5, 6, 7, 8, 9, 10, 1, 2, 3, 4] 
    return nc.ncon(tensors, indices, contord)

def dynamical_correlations(AL, AR, C, excited_energy, excited_states, 
                           mom_dist, freq_dist, gamma, O, i):
    corr_fxn = []

    AC = np.tensordot(C, AR, axes=(1, 0))
    print('AC', AC.shape)

    def left_env(X):
        X = X.reshape(D, D)

        t = X @ AR.transpose(1, 0, 2).reshape(D, d * D)
        XT = (AR.conj().transpose(2, 1, 0).reshape(D, D * d) 
               @ t.reshape(D * d, D))

        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        t = AL.reshape(d * D, D) @ X
        t = t.reshape(d, D, D).transpose(1, 0, 2).reshape(D, d * D)
        XT = t @ AL.conj().transpose(0, 2, 1).reshape(d * D, D)
        return (X - np.exp(+1.0j * p) * XT).ravel()

    for omega in freq_dist:

        s = 0
        for j in range(excited_states.shape[2]):

            B = np.tensordot(VL, excited_states[i,j], axes=(2, 0))
            print(B.shape)

            tensors = [B, AC.conj()]
            indices = [(1, 2, -2), (1, 2, -1)]
            contord = [1, 2]
            left_vec = nc.ncon(tensors, indices, contord)

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
            w1 = nc.ncon(tensors, indices, contord)

            tensors = [AL, O, AL.conj(), RB]
            indices = [(3, 2, 4), (1, 2), (3, 1, 5), (4, 5)]
            contord = [4, 5, 3, 1, 2]
            w2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, O, AR.conj()]
            indices = [(4, 5), (5, 2, 3), (1, 2), (4, 1, 3)]
            contord = [3, 4, 5, 1, 2]
            w3 = nc.ncon(tensors, indices, contord)

            spectral_weight = np.abs(w1 
                                   + np.exp(+1j * p) * w2 
                                   + np.exp(-1j * p) * w3
                                   )

            s += (2 * np.pi
                    * lorentzian(omega, excited_energy[i,j], gamma)
                    * spectral_weight**2 # is this right
                    )

        corr_fxn.append(s)
    return np.array(corr_fxn)

def lorentzian(x, x0, gamma):
    return (1 / np.pi) * ((0.5 * gamma)/((x - x0)**2 + (0.5 * gamma)**2))


########################### Initialization #############################
excit_energy, excit_states = [], []

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
    h = hamiltonians.XYZ_half(x, y, z, size='two')

if model == 'TFI':
    h = hamiltonians.TFI(y, z)

if model == 'oneXXZ':
    h = hamiltonians.XYZ_one(x, y, z, size='two')

checks(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)
print('gse', gs_energy(AL, AR, C, h))

######################### Pre-compute steps ############################
h -= np.real(gs_energy(AL, AR, C, h)) * np.eye(d**2)  # Regularize hamiltonian
h = h.reshape(d, d, d, d) 
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

tensors = [AL, AR.conj()]
indices = [(-1, 1, -3), (-2, 1, -4)]
E_LR = nc.ncon(tensors, indices)

tensors = [AR, AL.conj()]
indices = [(-1, 1, -3), (-2, 1, -4)]
E_RL = nc.ncon(tensors, indices)

def left_fixed_point(A, B):
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), B.conj()]
        indices = [(2, 1, -2), (3, 2), (3, 1, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    evals, evecs = spspla.eigs(E, k=1, which="LR", tol=1e-14)
    return evecs[:,0].reshape(D, D)

def right_fixed_point(A, B):
    def right_transfer_op(X):
        tensors = [A, X.reshape(D, D), B.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=right_transfer_op)
    evals, evecs = spspla.eigs(E, k=1, which="LR", tol=1e-14)
    return evecs[:,0].reshape(D, D)

lfp_LR = left_fixed_point(AL, AR)
rfp_LR = right_fixed_point(AL, AR)

lfp_LR /= np.trace(lfp_LR @ rfp_LR)
rfp_LR /= np.trace(lfp_LR @ rfp_LR)

lfp_RL = left_fixed_point(AR, AL)
rfp_RL = right_fixed_point(AR, AL)

lfp_RL /= np.trace(lfp_RL @ rfp_RL)
rfp_RL /= np.trace(lfp_RL @ rfp_RL)

print('ELR: left fix point check',
      spla.norm(lfp_LR - nc.ncon([lfp_LR, E_LR], [(2, 1), (1, 2, -2, -1)]))
      )

print('ELR: right fix point check',
      spla.norm(rfp_LR - nc.ncon([E_LR, rfp_LR], [(-1, -2, 1, 2), (1, 2)]))
      )

print('ELR (l|r)', np.trace(lfp_LR @ rfp_LR))

print('ERL: left fix point check',
      spla.norm(lfp_RL - nc.ncon([lfp_RL, E_RL], [(2, 1), (1, 2, -2, -1)]))
      )

print('ERL: right fix point check',
      spla.norm(rfp_RL - nc.ncon([E_RL, rfp_RL], [(-1, -2, 1, 2), (1, 2)]))
      )

print('ERL (l|r)', np.trace(lfp_RL @ rfp_RL))

tensors = [rfp_LR, lfp_LR]
indices = [(-1, -2), (-4, -3)]
P_LR = nc.ncon(tensors, indices)

E_LR -= P_LR

tensors = [rfp_RL, lfp_RL]
indices = [(-1, -2), (-4, -3)]
P_RL = nc.ncon(tensors, indices)

E_RL -= P_RL
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

plt.plot(mom_vec, excit_energy)
plt.title('s=1/2, %s, z=%.2f, D=%i ' % params)
plt.show()

exit()

gamma = 0.05
freq_dist = np.linspace(excit_energy.min() - 0.25,
                        excit_energy.max() + 0.25, 
                        150)

for i in range(mom_dist.shape[0]):
    print('mom', mom_dist[i])
    spectral_fxn.append(
        dynamical_correlations(AL, AR, C, 
                               excit_energy, excit_states, 
                               mom_dist, freq_dist, 
                               gamma, sz, i
                               )
        )

# plt.plot(mom_vec, np.abs(np.cos(mom_vec - np.pi / 2)), label ='exact')

# plt.ylabel('\u03C9 / 0.410479248463')  

# plt.plot(mom_vec, 
#          2 * np.sqrt(z**2 + y**2 - 2 * z * y * np.cos(mom_vec)), 
#          label='exact'
#          )

# plt.plot(mom_vec, np.pi / 2 * np.abs(np.sin(mom_vec)), label='exact') 
















