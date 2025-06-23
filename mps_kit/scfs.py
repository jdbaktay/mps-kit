import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools

from .mps_tools import calc_corr_length

def momentum_dist(AL, AR, C, o1, o2, o3):
    d = AL.shape[1]
    D = C.shape[0]

    AL = AL.transpose(1, 0, 2)
    AR = AR.transpose(1, 0, 2)

    momentum = []

    AC = np.tensordot(AL, C, axes=(2, 0))

    tensors = [AC, o2, o1, AC.conj()]
    indices = [(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc.ncon(tensors, indices, contord)
    print('n --> s1', s1)

    N = int(np.floor(calc_corr_length(AL, C, 1e-14)[0]))
    print('N for n(k)', N)

    filling = s1.real
    q = np.concatenate((np.linspace(0, filling, int(np.floor(N * filling)), endpoint=False),
                        np.linspace(filling, 1, N - int(np.floor(N * filling))))) * np.pi

    def left(X,o,Y):
        indices = [(2,1,-2), (3,2), (3,1,-1)]
        return nc.ncon([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices = [(2,-1,1), (3,2), (3,-2,1)]
        return nc.ncon([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC, o1, AL), right(AR, o2, AC)
    s3l, s3r = left(AL, o2, AC), right(AC, o1, AR)

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

    L1, R1 = np.random.rand(D, D) - .5, np.random.rand(D, D) - .5

    for p in q:
        print('n(', p, ')')
        left_env_op = spspla.LinearOperator((D * D, D * D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D * D, D * D), matvec=right_env)

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(),
                             x0=L1.ravel(), rtol=10**-12, atol=10**-12
                             )

        R1, _ = spspla.gmres(right_env_op, s3r.ravel(),
                             x0=R1.ravel(), rtol=10**-12, atol=10**-12
                             )

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * np.tensordot(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * np.tensordot(s3l, R1, axes=([1,0], [0,1]))

        s = s1 + s2 + s3

        momentum.append(s.real)
    return q, np.array(momentum)

def stat_struc_fact(AL, AR, C, o1, o2, o3):
    d = AL.shape[1]
    D = C.shape[0]

    AL = AL.transpose(1, 0, 2)
    AR = AR.transpose(1, 0, 2)

    stat_struc_fact = []

    N = int(np.floor(calc_corr_length(AL, C, 1e-14)[0]))
    print('N for s(k)', N)

    AC = np.tensordot(AL, C, axes=(2,0))

    q = np.linspace(0, 1, N) * np.pi

    o1 = o1 - nc.ncon([AC, o1, AC.conj()], [[3,1,4], [2,3], [2,1,4]]) * np.eye(d)
    o2 = o2 - nc.ncon([AC, o2, AC.conj()], [[3,1,4], [2,3], [2,1,4]]) * np.eye(d)

    tensors = [AC, o1, o2, AC.conj()]
    indices = [(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc.ncon(tensors, indices, contord)
    print('s --> s1', s1)

    def left(X,o,Y):
        indices = [(2,1,-2), (3,2), (3,1,-1)]
        return nc.ncon([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices = [(2,-1,1), (3,2), (3,-2,1)]
        return nc.ncon([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC, o1, AL), right(AR, o2, AC)
    s3l, s3r = left(AL, o2, AC), right(AC, o1, AR)

    def left_env(X):
        X = X.reshape(D, D)

        t = X @ AR.transpose(1, 0, 2).reshape(D, d * D)
        XT = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t.reshape(D * d, D)
        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        t = AL.reshape(d * D, D) @ X
        t = t.reshape(d, D, D).transpose(1, 0, 2).reshape(D, d * D)
        XT = t @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)
        return (X - np.exp(+1.0j * p) * XT).ravel()

    L1, R1 = np.random.rand(D, D) - .5, np.random.rand(D, D) - .5

    for p in q:
        print('s(', p, ')')
        left_env_op = spspla.LinearOperator((D * D, D * D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D * D, D * D), matvec=right_env)

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(),
                             x0=L1.ravel(), rtol=10**-12, atol=10**-12
                             )

        R1, _ = spspla.gmres(right_env_op, s3r.ravel(),
                             x0=R1.ravel(), rtol=10**-12, atol=10**-12
                             )

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * np.tensordot(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * np.tensordot(s3l, R1, axes=([1,0], [0,1]))

        s = s1 + s2 + s3

        stat_struc_fact.append(s.real)
    return q, np.array(stat_struc_fact)
