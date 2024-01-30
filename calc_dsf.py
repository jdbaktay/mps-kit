import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools
import sys
import os

from mps_tools import checks

def calc_dsf(AL, AR, AC, 
             excit_energy, excit_states, 
             mom_vec, O):

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

    O = (O 
         - nc.ncon([AC, O, AC.conj()], [[1, 3, 4], [2, 3], [1, 2, 4]]) 
           * np.eye(d)
           )

    dsf = []
    for i, p in enumerate(mom_vec):
        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

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

            dsf.append(np.abs(t1 + np.exp(+1j * p) * t2))
    return np.array(dsf).reshape(mom_vec.size, excit_states.shape[2])

def my_corr_length(A, X0, tol):
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), A.conj()]
        indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)

    # k must be LARGER THAN OR EQUAL TO 2
    evals = spspla.eigs(E, k=4, which="LM", v0=X0, tol=tol, 
                                return_eigenvectors=False)
    return -1.0 / np.log(np.abs(evals[-2])), evals

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

######################### Compute dsf ##################################

VL = spla.null_space(AL.conj().reshape(D * d, D).T)
VL = VL.reshape(D, d, (d - 1) * D)

dsf = calc_dsf(AL, AR, np.tensordot(AL, C, axes=(2, 0)), 
               excit_energy, excit_states, 
               mom_vec, n
               )

print('dsf:', dsf.shape)

_, corrlens = my_corr_length(AL.transpose(1, 0 ,2), C, 1e-14)

path = ''

filename = f'{model}_specweights_{x}_{y}_{z}_{g}_{D:03}_{N:05}_.npz'
np.savez(os.path.join(path, filename), mom=mom_vec, 
                                       evals=excit_energy, 
                                       dsf=dsf, 
                                       corrlens=corrlens
                                       )



