import ncon as nc
import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import functools
import hamiltonians
import mps_tools
import sys
import os

def calc_lfp(A, B, o3):
    def left_transfer_op(X):
        tensors = [X.reshape(D, D), A, o3, B.conj()]
        indices = [(4, 5), (2, 5, -2), (1, 2), (1, 4, -1)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    wl, lfp_AB = spspla.eigs(E, k=1, which='LM', tol=1e-14)

    lfp_AB = lfp_AB.reshape(D, D)

    print('const. diag', lfp_AB[0,0])
    print('mag. const. diag', np.abs(lfp_AB[0,0]))
    print('1/sqrt(D)', 1 / np.sqrt(D))

    lfp_AB /= lfp_AB[0,0] # yields identity for o3 = identity

    return lfp_AB

def calc_expectation_val(o, AC, lfp):
    tensors = [lfp, AC, o, AC.conj()]
    indices = [(3, 4), (2, 4, 5), (1, 2), (1, 3, 5)]
    return nc.ncon(tensors, indices)

def calc_scf(AL, AR, C, o1, o2, o3, mom_vec):
    scf = []

    AC = np.tensordot(AL, C, axes=(2, 0))

    lfp = calc_lfp(AL, AL, o3)

    print('<o1>', calc_expectation_val(o1, AC, lfp))
    print('<o2>', calc_expectation_val(o2, AC, lfp))

    o1 = o1 - calc_expectation_val(o1, AC, lfp) * np.eye(d)
    o2 = o2 - calc_expectation_val(o2, AC, lfp) * np.eye(d)

    print('<o1>', calc_expectation_val(o1, AC, lfp))
    print('<o2>', calc_expectation_val(o2, AC, lfp))

    def left_env(X):
        X = X.reshape(D, D)

        t = X @ AR.transpose(1, 0, 2).reshape(D, d * D)
        t = o3 @ t.reshape(D, d, D).transpose(1, 0, 2).reshape(d, D * D)
        t = t.reshape(d, D, D).transpose(1, 0, 2).reshape(D * d, D)
        XT = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t
        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D,D)

        t = AL.reshape(d * D, D) @ X
        t = o3 @ t.reshape(d, D * D)
        t = t.reshape(d, D, D).transpose(1, 0, 2).reshape(D, d * D)
        XT = t @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)
        return (X - np.exp(+1.0j * p) * XT).ravel()

    tensors = [AC, o2, o1, AC.conj()]
    indices = [(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc.ncon(tensors, indices, contord)
    print('n --> s1', s1)

    def left(X,o,Y):
        indices = [(2, 1, -2), (3, 2), (3, 1, -1)]
        return nc.ncon([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices = [(2,-1,1), (3,2), (3,-2,1)]
        return nc.ncon([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC, o1, AL), right(AR, o2, AC)
    s3l, s3r = left(AL, o2, AC), right(AC, o1, AR)

    for p in mom_vec:
        print('scf(', p, ')')
        left_env_op = spspla.LinearOperator((D * D, D * D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D * D, D * D), matvec=right_env)

        L1 = spspla.gmres(left_env_op, s2l.ravel(), 
                          x0=(np.random.rand(D, D) - 0.5).ravel(), 
                          tol=10**-12, 
                          atol=10**-12
                          )[0].reshape(D, D)

        R1 = spspla.gmres(right_env_op, s3r.ravel(), 
                          x0=(np.random.rand(D, D) - 0.5).ravel(), 
                          tol=10**-12, 
                          atol=10**-12
                          )[0].reshape(D, D)

        s2 = np.exp(-1.0j * p) * np.tensordot(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j * p) * np.tensordot(s3l, R1, axes=([1,0], [0,1]))

        s = s1 + s2 + s3

        scf.append(s.real)
    return np.array(scf)

def my_corr_length(A, X0, tol):
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), A.conj()]
        indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    evals = spspla.eigs(E, k=2, which="LM", v0=X0, tol=tol, 
                           return_eigenvectors=False
                           )
    return -1.0 / np.log(np.abs(evals[-2]))

def calc_expectations(AL, AR, C, O):
    AC = np.tensordot(AL, C, axes=(2, 0))

    if O.shape[0] == d:
        tensors = [AC, O, AC.conj()]
        indices = [(2, 3, 4), (1, 2), (1, 3, 4)]
        contord = [3, 4, 1, 2]
        expectation_value = nc.ncon(tensors, indices, contord)

    if O.shape[0] == d**2:
        pass
    return expectation_value


tol = 1e-12

model, d, D = str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

x, y, z = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

params = (model, x, y, z, D)

path = '/Users/joshuabaktay/Desktop/local data/states'

filename = '%s_AL_%.2f_%.2f_%.2f_%03i_.txt' % params
AL = np.loadtxt(os.path.join(path, filename), dtype=complex)
AL = AL.reshape(d, D, D)

filename = '%s_AR_%.2f_%.2f_%.2f_%03i_.txt' % params
AR = np.loadtxt(os.path.join(path, filename), dtype=complex)
AR = AR.reshape(d, D, D)

filename = '%s_C_%.2f_%.2f_%.2f_%03i_.txt' % params
C = np.loadtxt(os.path.join(path, filename), dtype=complex)
C = C.reshape(D, D)

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

correlation_length = my_corr_length(AL, C, tol/100)
print('correlation length', correlation_length)

N = int(np.floor(correlation_length))
print('N for scf', N)

qs = np.linspace(0, 1, N) * np.pi
ssf = calc_scf(AL, AR, C, n, n, si, qs)

qs /= np.pi

plt.plot(qs, ssf, 'x')
plt.grid()
plt.show()

density = calc_expectations(AL, AR, C, n)

filling = np.real(density)
qm = np.concatenate(
        (np.linspace(0, filling, 
                     int(np.floor(N * filling)), endpoint=False
                     ),
         np.linspace(filling, 1, 
                     N - int(np.floor(N * filling))
                     )
        )
        ) * np.pi

mom_dist = calc_scf(AL, AR, C, sp, sm, -sz, qm)

qm /= np.pi

plt.plot(qm, mom_dist, 'x')
plt.grid()
plt.show()



