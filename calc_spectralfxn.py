import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools
import sys
import os

from mps_tools import checks

def calc_expectation_val(o, AC, lfp):
    tensors = [lfp, AC, o, AC.conj()]
    indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
    return nc.ncon(tensors, indices)

def my_corr_length(A, X0, tol):
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), A.conj()]
        indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)

    # k must be LARGER THAN OR EQUAL TO 2
    evals = spspla.eigs(E, k=4, which="LM", v0=X0, tol=tol, 
                                return_eigenvectors=False
                                )
    return -1.0 / np.log(np.abs(evals[-2])), evals

def op_transfer_matrix(A, B):
    def left_transfer_op(X):
        tensors = [X.reshape(D, D), A, -sz, B.conj()]
        indices = [(4, 5), (5, 2, -2), (1, 2), (4, 1, -1)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    def right_transfer_op(X):
        tensors = [A, -sz, B.conj(), X.reshape(D, D)]
        indices = [(-1, 2, 4), (1, 2), (-2, 1, 5), (4, 5)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    wl, lfp_AB = spspla.eigs(E, k=2, which='LM', tol=1e-14)
    print('wl', wl, np.abs(wl))
    print('phi', np.angle(wl))
   
    if np.angle(wl[0]) > 0:
        lfp_AB = lfp_AB[:,0].reshape(D, D)
        print(np.angle(wl[0]))
        phi = np.angle(wl[0])
    if np.angle(wl[0]) < 0:
        lfp_AB = lfp_AB[:,1].reshape(D, D)
        print(np.angle(wl[1]))
        phi = np.angle(wl[1])         

    E = spspla.LinearOperator((D * D, D * D), matvec=right_transfer_op)
    wr, rfp_AB = spspla.eigs(E, k=2, which='LM', tol=1e-14)
    print('wr', wr, np.abs(wr))
    print('phi', np.angle(wr))
    
    if np.angle(wr[0]) > 0:
        rfp_AB = rfp_AB[:,0].reshape(D, D)
        print(np.angle(wr[0]))
    if np.angle(wr[0]) < 0:
        rfp_AB = rfp_AB[:,1].reshape(D, D)
        print(np.angle(wr[1]))

    norm = np.trace(lfp_AB @ rfp_AB)
    print('(l|r)', norm)

    # Normalize fixed points
    lfp_AB /= np.sqrt(norm)
    rfp_AB /= np.sqrt(norm)
    
    print('(l|r)', np.trace(lfp_AB @ rfp_AB))
    return lfp_AB, np.exp(1j * phi)

def calc_specfxn(AL, AR, AC, 
                 excit_energy, excit_states, mom_vec, 
                 sp, sm):

    lz, phase = op_transfer_matrix(AL, AL)
    
    def left_env(X):
        X = X.reshape(D, D)

        tensors = [X, AR, -sz, (phase * AR).conj()]
        indices = [(3, 4), (4, 2, -2), (1, 2), (3, 1, -1)]
        contord = [3, 4, 1, 2]
        XT = nc.ncon(tensors, indices, contord)
        return (X - np.exp(-1.0j * p) * XT).ravel()

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

    Ap, Ah = [], []
    for i, p in enumerate(mom_vec):
        if p / np.pi == 0:
            sp -= calc_expectation_val(sp, AC, np.eye(D)) * np.eye(d)
            sm -= calc_expectation_val(sm, AC, np.eye(D)) * np.eye(d)
            print('p', p)
   
        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

            tensors = [lz, B, -sz, (phase * AC).conj()]
            indices = [(3, 4), (4, 2, -2), (1, 2), (3, 1, -1)]
            contord = [3, 4, 1, 2]
            left_vec = nc.ncon(tensors, indices, contord)

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            left_env_op = spspla.LinearOperator((D * D, D * D),
                                                matvec=left_env
                                                )

            right_env_op = spspla.LinearOperator((D * D, D * D), 
                                                 matvec=right_env
                                                 )

            LB = spspla.gmres(left_env_op, left_vec.ravel(),
                                           x0=rand_init.ravel(), 
                                           tol=1e-14, 
                                           atol=1e-14
                                           )[0].reshape(D, D)

            RB = spspla.gmres(right_env_op, right_vec.ravel(), 
                                            x0=rand_init.ravel(), 
                                            tol=1e-14, 
                                            atol=1e-14
                                            )[0].reshape(D, D)

            tensors = [lz, B, sp, AC.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sp1 = nc.ncon(tensors, indices, contord)

            tensors = [lz, AL, sp, AL.conj(), RB]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 6), (5, 6)]
            contord = [3, 4, 5, 6, 1, 2]
            sp2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, sp, AR.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sp3 = nc.ncon(tensors, indices, contord)

            tensors = [lz, B, sm, AC.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sm1 = nc.ncon(tensors, indices, contord)

            tensors = [lz, AL, sm, AL.conj(), RB]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 6), (5, 6)]
            contord = [3, 4, 5, 6, 1, 2]
            sm2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, sm, AR.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sm3 = nc.ncon(tensors, indices, contord)

            Ap.append(np.abs(sp1 
                           + np.exp(+1j * p) * sp2 
                           + np.exp(-1j * p) * sp3
                           )
                      )

            Ah.append(np.abs(sm1
                           + np.exp(+1j * p) * sm2
                           + np.exp(-1j * p) * sm3
                            )
                      )

    Ap = np.array(Ap).reshape(mom_vec.size, excit_states.shape[2])
    Ah = np.array(Ah).reshape(mom_vec.size, excit_states.shape[2])
    return Ap, Ah

########################### Initialization #############################

model = str(sys.argv[1])
d = int(sys.argv[2])
D = int(sys.argv[3])
x = float(sys.argv[4])
y = float(sys.argv[5])
z = float(sys.argv[6])
g = float(sys.argv[7])
N = int(sys.argv[8])

params = (model, x, y, z, g, D, N)
print('input params', params)

path = '' 

filename = f'{model}_gs_{x}_{y}_{z}_{g}_{D:03}_.npz'
gs_mps = np.load(os.path.join(path, filename))

# transpose needed for change to easier index conventions
AL = gs_mps['AL'].transpose(1, 0, 2)
AR = gs_mps['AR'].transpose(1, 0, 2)
C = gs_mps['C']

path = ''

filename = f'{model}_excits_{x}_{y}_{z}_{g}_{D:03}_{N:05}_.npz'
excits = np.load(os.path.join(path, filename))

mom_vec = excits['mom']
excit_energy = excits['evals']
excit_states = excits['estates']

if d == 2:
    sx = np.array([[0, 1],[1, 0]]) 
    sy = np.array([[0, -1j],[1j, 0]]) 
    sz = np.array([[1, 0],[0, -1]]) 

if d == 3:
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) 
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) 
    sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

sp = 0.5 * (sx + 1.0j * sy)
sm = 0.5 * (sx - 1.0j * sy)
n = 0.5 * (sz + np.eye(d))

###################### Compute spectral fxn ############################

VL = spla.null_space(AL.conj().reshape(D * d, D).T)
VL = VL.reshape(D, d, (d - 1) * D)

Ap, Ah = calc_specfxn(AL, AR, np.tensordot(AL, C, axes=(2, 0)), 
                      excit_energy, excit_states, mom_vec, 
                      sp, sm
                      )

print('Ap, Ah:', Ap.shape, Ah.shape)

_, corrlens = my_corr_length(AL.transpose(1, 0, 2), C, 1e-14)

path = ''

filename = f'{model}_specweights_{x}_{y}_{z}_{g}_{D:03}_{N:05}_.npz'
np.savez(os.path.join(path, filename), mom=mom_vec, 
                                       evals=excit_energy, 
                                       Ap=Ap, 
                                       Ah=Ah, 
                                       corrlens=corrlens
                                       )

