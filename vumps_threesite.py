# for q in `jot 10 24 199`; do python3 -Walways -Werror ../vumps_threesite.py +2.0 0.0 +1.0 $q; done
# python3 -Walways -Werror ../vumps_threesite.py +0.25 -0.5 +1.0 67
#cd Desktop/code/vumps 

import ncon as nc
import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import functools
import sys
import os

from scipy import stats

def left_ortho(A, X0, tol, stol):
    def left_fixed_point(A, B):
        def left_transfer_op(X):
            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1,2,-2), (3, 2), (1, 3, -1)]
            contord = [2, 3, 1]
            return nc.ncon(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
        evals, evecs = spspla.eigs(E, k=1, which="LR", v0=X0, tol=tol)
        return evals[0], evecs[:,0].reshape(D, D)

    eval_LR, l = left_fixed_point(A, A)

    l = l + l.T.conj()
    l /= np.trace(l)

    A = A/np.sqrt(eval_LR)

    w, v = spla.eigh(l)
    L = np.diag(np.sqrt(np.abs(w))) @ v.T.conj()

    u, s, vh = spla.svd(L)

    si = 1/s
    for i in range(s.size):
        if s[i] < stol:
            si[i] = 0

    Li = vh.conj().T @ np.diag(1/s) @ u.conj().T

    AL = nc.ncon([L, A, Li], [(-2,1), (-1,1,2), (2,-3)])
    return AL, L

def right_ortho(A, X0, tol, stol):
    A, L = left_ortho(np.transpose(A, (0, 2, 1)), X0, tol, stol)
    A, L = np.transpose(A, (0, 2, 1)), np.transpose(L, (1, 0))
    return A, L

def dynamic_expansion(AL, AR, C, Hl, Hr, h, delta_D):
    Al = AL.reshape(d * D, D)
    Ar = AR.transpose(1, 0, 2).reshape(D, d * D)

    def calcnullspace(n):
        u, s, vh = spla.svd(n, full_matrices=True)

        right_null = vh.conj().T[:,D:]
        left_null = u.conj().T[D:,:]

        return left_null, right_null

    _, Nl = calcnullspace(Al.T.conj())
    Nr, _ = calcnullspace(Ar.T.conj())

    def eff_ham(X):
        X = X.reshape(d, D, d, D)

        tensors = [AL, AL, X, h, AL.conj(), AL.conj()]
        indices = [(7, 1, 2), (8, 2, 4), (9, 4, -3, -4), (5, 6, -1, 7, 8, 9), (5, 1, 3), (6, 3, -2)]
        contord = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        H1 = nc.ncon(tensors,indices,contord)

        # t = AL.transpose(1,0,2).reshape(D*d,D)@X.transpose(1,0,2,3).reshape(D,d*d*D)
        # t = AL.transpose(1,0,2).reshape(D*d,D)@t.reshape(D,d*d*d*D)
        # t = h.reshape(d**3,d**3)@t.reshape(D,d**3,d*D).transpose(1,0,2).reshape(d**3,D*d*D)
        # t = AL.conj().transpose(2,0,1).reshape(D,d*D)@t.reshape(d,d**2,D,d*D).transpose(0,2,1,3).reshape(d*D,d*d*d*D)
        # t = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,d*d*D)
        # t = t.reshape(D,d,d,D).transpose(1,0,2,3)
        # print(spla.norm(t - H1))

        tensors = [AL, X, h, AL.conj()]
        indices = [(4, 1, 2), (5, 2, 6, -4), (3, -1, -3, 4, 5, 6), (3, 1, -2)]
        contord = [1, 2, 3, 4, 5, 6]
        H2 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(4, -2, 5, 2), (6, 2, 1), (-1, -3, 3, 4, 5, 6), (3, -4, 1)]
        contord = [1, 2, 3, 4, 5, 6]
        H3 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, AR, h, AR.conj(), AR.conj()]
        indices = [(-1, -2, 7, 4), (8, 4, 2), (9, 2, 1), (-3, 5, 6, 7, 8, 9), (5, -4, 3), (6, 3, 1)]
        contord = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        H4 = nc.ncon(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2, 1), (-1, 1, -3, -4)]
        H5 = nc.ncon(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1, -2, -3, 1), (1, -4)]
        H6 = nc.ncon(tensors,indices)
        return (H1 + H2 + H3 + H4 + H5 + H6).ravel()

    A_two_site = nc.ncon([AL, C, AR], [(-1, -2, 1), (1, 2), (-3, 2, -4)])
    A_two_site = eff_ham(A_two_site).reshape(d * D, d * D)

    t = Nl.conj().T @ A_two_site @ Nr.conj().T
    u, s, vh = spla.svd(t, full_matrices=True)
    print('deltaD svals', s[:delta_D])
    print('>deltaD svals', s[delta_D:])

    u = u[:, :delta_D]
    vh = vh[:delta_D, :]

    if delta_D > D:
        expand_left = (Nl @ u).reshape(d, D, D)
        expand_right = (vh @ Nr).reshape(D, d, D).transpose(1, 0, 2)
        t = delta_D - D
    else:
        expand_left = (Nl @ u).reshape(d, D, delta_D)
        expand_right = (vh @ Nr).reshape(delta_D, d, D).transpose(1, 0, 2)
        t = 0

    AL_new = np.concatenate((AL, expand_left), axis=2)
    AR_new = np.concatenate((AR, expand_right), axis=1)

    AL, AR = [], []

    for i in range(AL_new.shape[0]):
        AL.append(np.pad(AL_new[i,:,:], pad_width=((0, delta_D), (0, t)), mode='constant'))

    for i in range(AR_new.shape[0]):
        AR.append(np.pad(AR_new[i,:,:], pad_width=((0, t), (0, delta_D)), mode='constant'))

    C = np.pad(C, pad_width=((0, delta_D), (0, delta_D)), mode='constant')
    Hl = np.pad(Hl, pad_width=((0, delta_D), (0, delta_D)), mode='minimum')
    Hr = np.pad(Hr, pad_width=((0, delta_D), (0, delta_D)), mode='minimum')
    return np.array(AL), np.array(AR), C, Hl, Hr

def HeffTerms(AL, AR, C, h, Hl, Hr, ep):
    tensors = [AL, AL, AL, h, AL.conj(), AL.conj(), AL.conj()]
    indices = [(4,7,8), (5,8,10), (6,10,-2), (1,2,3,4,5,6), (1,7,9), (2,9,11), (3,11,-1)]
    contord = [7,8,9,10,11,1,2,3,4,5,6]
    hl = nc.ncon(tensors,indices,contord)
    el = np.trace(hl @ C @ C.T.conj())

    tensors = [AR, AR, AR, h, AR.conj(), AR.conj(), AR.conj()]
    indices = [(4,-1,10), (5,10,8), (6,8,7), (1,2,3,4,5,6), (1,-2,11), (2,11,9), (3,9,7)]
    contord = [7,8,9,10,11,1,2,3,4,5,6]
    hr = nc.ncon(tensors,indices,contord)
    er = np.trace(C.T.conj() @ C @ hr)

    e = 0.5 * (el + er)

    hl -= el * np.eye(D)
    hr -= er * np.eye(D)
    print('hl == hl+', spla.norm(hl - hl.T.conj()))
    print('hr == hr+', spla.norm(hr - hr.T.conj()))

    hl = 0.5 * (hl + hl.T.conj())
    hr = 0.5 * (hr + hr.T.conj())

    Hl -= np.trace(Hl @ C @ C.T.conj()) * np.eye(D)
    Hr -= np.trace(C.T.conj() @ C @ Hr) * np.eye(D)

    def left_env(X):
        X = X.reshape(D, D)

        t = X @ AL.transpose(1, 0, 2).reshape(D, d * D)
        XT = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t.reshape(D * d, D)

        XR = np.trace(X @ C @ C.T.conj()) * np.eye(D)
        return (X - XT + XR).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        t = AR.reshape(d * D, D) @ X
        t = t.reshape(d, D, D).transpose(1, 2, 0).reshape(D, D * d)
        XT = t @ AR.conj().transpose(2, 0, 1).reshape(D * d, D)

        XL = np.trace(C.T.conj() @ C @ X) * np.eye(D)
        return (X - XT + XL).ravel()

    Ol = spspla.LinearOperator((D**2,D**2), matvec=left_env)
    Or = spspla.LinearOperator((D**2,D**2), matvec=right_env)

    Hl, _ = spspla.gmres(Ol, hl.ravel(), x0=Hl.ravel(), tol=ep/100, atol=ep/100)
    Hr, _ = spspla.gmres(Or, hr.ravel(), x0=Hr.ravel(), tol=ep/100, atol=ep/100)

    Hl, Hr = Hl.reshape(D,D), Hr.reshape(D,D)
    print('Hl == Hl+', spla.norm(Hl - Hl.T.conj()))
    print('Hr == Hr+', spla.norm(Hr - Hr.T.conj()))

    Hl = 0.5 * (Hl + Hl.T.conj())
    Hr = 0.5 * (Hr + Hr.T.conj())

    print('(L|hr)', np.trace(C.T.conj() @ C @ hr))
    print('(hl|R)', np.trace(hl @ C @ C.T.conj()))

    print('(L|Hr)', np.trace(C.T.conj() @ C @ Hr))
    print('(Hl|R)', np.trace(Hl @ C @ C.T.conj()))
    return Hl, Hr, e

def Apply_HC(hl_mid, hr_mid, AL, AR, h, Hl, Hr, X):
    X = X.reshape(D, D)

    t = hl_mid.transpose(0, 1, 3, 2).reshape(D * d * d, D) @ X
    t = t.reshape(D * d, d * D) @ AR.reshape(d * D, D)
    H1 = t.reshape(D, d * D) @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)

    t = X @ hr_mid.transpose(2, 0, 1, 3).reshape(D, d * D * d)
    t = t.reshape(D, d, D, d).transpose(3, 0, 1, 2).reshape(d * D, d * D)
    t = AL.transpose(1, 0, 2).reshape(D, d * D) @ t
    H2 = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t.reshape(D * d, D)

    H3 = Hl @ X
    H4 = X @ Hr
    return (H1 + H2 + H3 + H4).ravel()

def Apply_HAC(hl_mid, hr_mid, AL, AR, h, Hl, Hr ,X):
    X = X.reshape(D, d, D)

    t = hl_mid.reshape(D * d, D * d) @ X.reshape(D * d, D)
    H1 = t.reshape(D, d, D)

    t = AL.reshape(d * D, D) @ X.reshape(D, d * D)
    t = t.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
    t = t.reshape(d, D, d, d, D).transpose(1, 4, 0, 2, 3).reshape(D * D, d * d * d)
    t = t @ h.reshape(d * d * d, d * d * d).transpose(1, 0)
    t = t.reshape(D, D, d, d, d).transpose(0, 2, 3, 1, 4).reshape(D * d, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t 
    t = t.reshape(D * d, D * d) @ AR.conj().transpose(2, 0, 1).reshape(D * d, D)
    H2 = t.reshape(D, d, D)

    t = X.reshape(D, d * D) @ hr_mid.transpose(3, 2, 0, 1).reshape(d * D,d * D)
    H3 = t.reshape(D, d, D)

    t = Hl @ X.reshape(D, d * D)
    H4 = t.reshape(D, d, D)

    t = X.reshape(D * d, D) @ Hr
    H5 = t.reshape(D, d, D)
    return (H1 + H2 + H3 + H4 + H5).ravel()

def calc_new_A(AL, AR, AC, C):
    Al = AL.reshape(d * D, D)
    Ar = AR.transpose(1,0,2).reshape(D, d * D)

    def calcnullspace(n):
        u,s,vh = spla.svd(n, full_matrices=True)

        right_null = vh.conj().T[:,D:]
        left_null = u.conj().T[D:,:]

        return left_null, right_null

    _, Al_right_null = calcnullspace(Al.T.conj())
    Ar_left_null, _  = calcnullspace(Ar.T.conj())

    Bl = Al_right_null.T.conj() @ AC.transpose(1, 0, 2).reshape(d * D, D)
    Br = AC.reshape(D, d * D) @ Ar_left_null.T.conj()

    epl = spla.norm(Bl)
    epr = spla.norm(Br)

    s = spla.svdvals(C)
    print('first svals', s[:5])
    print('last svals', s[-5:])

    ulAC, plAC = spla.polar(AC.reshape(D * d, D), side='right')
    urAC, prAC = spla.polar(AC.reshape(D, d * D), side='left')

    ulC, plC = spla.polar(C, side='right')
    urC, prC = spla.polar(C, side='left')

    AL = (ulAC @ ulC.T.conj()).reshape(D, d, D).transpose(1, 0, 2)
    AR = (urC.T.conj() @ urAC).reshape(D, d, D).transpose(1, 0, 2)
    return epl, epr, AL, AR

def vumps(AL, AR, C, h, Hl, Hr, ep):
    AC = np.tensordot(C, AR, axes=(1, 1))

    Hl, Hr, e = HeffTerms(AL, AR, C, h, Hl, Hr, ep)

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(4, 7, 8), (5, 8, -3), (1, 2, -2, 4, 5, -4), (1, 7, 9), (2, 9, -1)]
    contord = [7, 8, 9, 1, 2, 4, 5]
    hl_mid = nc.ncon(tensors,indices,contord)

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(5, -3, 8), (6, 8, 7), (-1, 2, 3, -4, 5, 6), (2, -2, 9), (3, 9,7 )]
    contord = [7, 8, 9, 2, 3, 5, 6]
    hr_mid = nc.ncon(tensors,indices,contord)

    f = functools.partial(Apply_HC, hl_mid, hr_mid, AL, AR, h, Hl, Hr)
    g = functools.partial(Apply_HAC, hl_mid, hr_mid, AL, AR, h, Hl, Hr)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)
    w, v = spspla.eigsh(H, k=1, which='SA', v0=C.ravel(), tol=ep/100)
    C = v[:,0].reshape(D, D)
    print('C_eval', w[0], C.shape)

    H = spspla.LinearOperator((D * d * D, D * d * D), matvec=g)
    w, v = spspla.eigsh(H, k=1, which='SA', v0=AC.ravel(), tol=ep/100)
    AC = v[:,0].reshape(D, d, D)
    print('AC_eval', w[0], AC.shape)

    epl, epr, AL, AR = calc_new_A(AL, AR, AC, C)
    return AL, AR, C, Hl, Hr, e, epl, epr

def calc_discard_weight(AL, AR, C, h, Hl, Hr):
    def eff_ham(X):
        X = X.reshape(d, D, d, D)

        t = AL.transpose(1,0,2).reshape(D*d,D)@X.transpose(1,0,2,3).reshape(D,d*d*D)
        t = AL.transpose(1,0,2).reshape(D*d,D)@t.reshape(D,d*d*d*D)
        t = h.reshape(d**3,d**3)@t.reshape(D,d**3,d*D).transpose(1,0,2).reshape(d**3,D*d*D)
        t = AL.conj().transpose(2,0,1).reshape(D,d*D)@t.reshape(d,d**2,D,d*D).transpose(0,2,1,3).reshape(d*D,d*d*d*D)
        t = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,d*d*D)
        H1 = t.reshape(D,d,d,D).transpose(1,0,2,3)

        t = (AL.reshape(d * D, D) 
                              @ X.transpose(1, 0, 2, 3).reshape(D, d * d * D))
        t = t.reshape(d, D, d, d, D).transpose(0, 2, 3, 1, 4)
        t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
        t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 2, 4)
        t = AL.conj().reshape(d * D, D).T @ t.reshape(d * D, d * d * D)
        H2 = t.reshape(D, d, d, D).transpose(1, 0, 2, 3)

        t = X.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
        t = t.reshape(d, D, d, d, D).transpose(0, 2, 3, 1, 4)
        t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
        t = t.reshape(d, d, d, D, D).transpose(0, 1, 3, 2, 4)
        t = (t.reshape(d * d * D, d * D) 
                            @ AR.conj().transpose(0, 2, 1).reshape(d * D, D))
        H3 = t.reshape(d, d, D, D).transpose(0, 2, 1, 3)

        t = X.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
        t = (t.reshape(d * D * d * d, D) 
                                @ AR.transpose(1, 0, 2).reshape(D, d * D))
        t = t.reshape(d * D, d, d, d, D).transpose(0, 4, 1, 2, 3)
        t = t.reshape(d * D * D, d**3) @ h.reshape(d**3, d**3).T
        t = t.reshape(d * D, D, d**2, d).transpose(0, 2, 3, 1)
        t = (t.reshape(d * D * d * d, d * D) 
                            @ AR.conj().transpose(0, 2, 1).reshape(d * D, D))
        t = (t.reshape(d * D * d, d * D) 
                            @ AR.conj().transpose(0, 2, 1).reshape(d * D, D))
        H4 = t.reshape(d, D, d, D)

        t = Hl @ X.transpose(1, 0, 2, 3).reshape(D, d * d * D)
        H5 = t.reshape(D, d, d, D).transpose(1, 0, 2, 3)

        t = X.reshape(d * D * d, D) @ Hr
        H6 = t.reshape(d, D, d, D)
        return (H1 + H2 + H3 + H4 + H5 + H6).ravel()

    two_site = nc.ncon([AL, C, AR], [(-1, -2, 1), (1, 2), (-3, 2, -4)])

    H = spspla.LinearOperator((d * D * d * D, d * D * d * D), matvec=eff_ham)

    w, v = spspla.eigs(H, k=1, which='SR', v0=two_site.ravel(), tol=1e-12, return_eigenvectors=True)

    s = spla.svdvals(v[:, 0].reshape(d * D, d * D))

    return sum(ss**2 for ss in s[D:])

def calc_stat_struc_fact(AL, AR, C, o1, o2, o3, N):
    stat_struc_fact = []

    # q = nonuniform_mesh(npts_left=0, npts_mid=N, npts_right=20, k0=0.05, dk=0.05) * np.pi
    N = int(np.floor(correlation_length))
    print('N for s(k)', N)
    q = np.linspace(0, 1, N) * np.pi

    AC = np.tensordot(AL, C, axes=(2,0))

    o1 = o1 - nc.ncon([AC, o1, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)
    o2 = o2 - nc.ncon([AC, o2, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)

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

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(), x0=L1.ravel(), tol=10**-12, atol=10**-12)
        R1, _ = spspla.gmres(right_env_op, s3r.ravel(), x0=R1.ravel(), tol=10**-12, atol=10**-12)

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * np.tensordot(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * np.tensordot(s3l, R1, axes=([1,0], [0,1]))

        s = s1 + s2 + s3

        stat_struc_fact.append(s.real)
    return q, np.array(stat_struc_fact)

def calc_momentum(AL, AR, C, o1, o2, o3, N):
    momentum = []

    AC = np.tensordot(AL, C, axes=(2,0))

    tensors = [AC, o2, o1, AC.conj()]
    indices = [(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc.ncon(tensors, indices, contord)
    print('n --> s1', s1)

    # q = nonuniform_mesh(npts_left=10, npts_mid=N, npts_right=10, k0=0.5, dk=0.1) * np.pi
    # N = int(np.ceil(correlation_length) / 2) * 2 - 1

    N = int(np.floor(correlation_length))
    print('N for n(k)', N)

    # q = np.linspace(0, 1, N) * np.pi
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

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(), x0=L1.ravel(), tol=10**-12, atol=10**-12)
        R1, _ = spspla.gmres(right_env_op, s3r.ravel(), x0=R1.ravel(), tol=10**-12, atol=10**-12)

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * np.tensordot(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * np.tensordot(s3l, R1, axes=([1,0], [0,1]))

        s = s1 + s2 + s3

        momentum.append(s.real)
    return q, np.array(momentum)

def nonuniform_mesh(npts_left, npts_mid, npts_right, k0, dk):
    k_left, k_right = k0 - dk, k0 + dk
    mesh = np.concatenate((
            np.linspace(0.0,     k_left,  npts_left, endpoint=False),
            np.linspace(k_left,  k_right, npts_mid,  endpoint=False),
            np.linspace(k_right, 1.0,     npts_right)
            ))

    test, step = np.linspace(k_left,  k_right, npts_mid, endpoint=False, retstep=True)
    print('step', step)
    return mesh

def calc_entent(C):
    s = spla.svdvals(C)

    b = -np.log(s[0])
    entent = -sum(ss**2 * np.log(ss**2) for ss in s)
    return entent, b

def calc_fidelity(X, Y):
    '''Presumes that MPS tensors X and Y are both properly normalized'''
    E = np.tensordot(X,Y.conj(),axes=(0, 0)).transpose(0, 2, 1, 3).reshape(D * D, D * D)

    evals = spspla.eigs(E, k=4, which='LM', return_eigenvectors=False)
    return np.max(np.abs(evals))

def calc_correlation_length(A):
    E = nc.ncon([A, A.conj()],[[1, -1, -3], [1, -2, -4]]).reshape(D * D, D * D)
    evals = spspla.eigs(E, k=4, which='LM', return_eigenvectors=False)
    return -1/np.log(np.abs(evals[np.argmax(evals) - 1]))

def my_corr_length(A, X0, tol):
    '''NEED TO SAVE ALL THREE EIGENVALUES TO DO FES.'''
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), A.conj()]
        indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    # k must be LARGER THAN OR EQUAL TO 2 so return statement makes sense
    evals = spspla.eigs(E, k=4, which="LM", v0=X0, tol=tol, return_eigenvectors=False)
    print('argmax', np.argmax(evals), evals)
    return -1.0 / np.log(np.abs(evals[-2]))

def checks(AL, AR, C):
    print('left iso', spla.norm(nc.ncon([AL, AL.conj()], [[3,1,-2], [3,1,-1]]) - np.eye(D)))
    print('right iso', spla.norm(nc.ncon([AR, AR.conj()], [[3,-1,1], [3,-2,1]]) - np.eye(D)))
    print('norm', nc.ncon([AL, AL.conj(), C, C.conj(), AR, AR.conj()], [[7,1,2],[7,1,3],[2,4],[3,5],[8,4,6],[8,5,6]]))
    print('ALC - CAR', spla.norm(nc.ncon([AL,C],[[-1,-2,1],[1,-3]]) - nc.ncon([C,AR],[[-2,1], [-1,1,-3]])))

##############################################################################

energy, error = [], []

count, d = 0, 2

tol, stol, ep = 1e-12, 1e-12, 1e-2

D = int(sys.argv[4])
Dmax = 0
delta_D = 0
N = 1000

si = np.array([[1, 0],[0, 1]])
sx = np.array([[0, 1],[1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

sp = 0.5 * (sx + 1.0j*sy)
sm = 0.5 * (sx - 1.0j*sy)
n = 0.5 * (sz + np.eye(d))

J, g = -1, -0.5
TFI = ((J / 4) * np.kron(si, np.kron(sx, sx)) 
    + (g / 2) * np.kron(si, np.kron(sz, si)))

x, y, z, m = -1, -1, 0, 0
XYZ = ((x / 4) * np.kron(np.kron(sx, sx), si) 
    + (y / 4) * np.kron(np.kron(sy, sy), si) 
    + (z / 4) * np.kron(np.kron(sz, sz), si) 
    + (m/2) * np.kron(np.kron(sz, si), si))

t, V, V2, g = -1, float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tVV2 = ((t / 4) * (np.kron(np.kron(sx, sx), si) + np.kron(np.kron(sy, sy), si))
      + (t / 4) * (np.kron(si, np.kron(sx, sx)) + np.kron(si, np.kron(sy, sy)))
      + (V / 8) * np.kron(np.kron(sz, sz), si)
      + (V / 8) * np.kron(si, np.kron(sz, sz))
      + (V2 / 4) * np.kron(np.kron(sz, si), sz)
      + (g / 6) * (np.kron(sz, np.kron(si, si)) + np.kron(np.kron(si, sz), si)
                   + np.kron(np.kron(si, si), sz)))

t, t2, tc, g = -1, float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tt2tc = ((t / 4) * (np.kron(np.kron(sx, sx), si) + np.kron(np.kron(sy, sy), si))
      + (t / 4) * (np.kron(si, np.kron(sx, sx)) + np.kron(si, np.kron(sy, sy)))
      + (t2 / 16) * (np.kron(sx, np.kron(sz, sx)) + np.kron(sy, np.kron(sz, sy)))
      + (tc / 32) * (np.kron(sz, np.kron(sx, sx)) + np.kron(sz, np.kron(sy, sy)))
      + (g / 6) * (np.kron(sz, np.kron(si, si)) + np.kron(np.kron(si, sz), si)
                   + np.kron(np.kron(si, si), sz)))

h = tt2tc
h = h.reshape(d, d, d, d, d, d)

A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
C = np.random.rand(D, D) - 0.5
Hl, Hr = np.eye(D, dtype=A.dtype), np.eye(D, dtype=A.dtype)

AL, C = left_ortho(A, C, tol/100, stol)
AR, C = right_ortho(AL, C, tol/100, stol)
checks(AL, AR, C)

AL, AR, C, Hl, Hr, *_ = vumps(AL, AR, C, h, Hl, Hr, ep)

AL, C = left_ortho(AR, C, tol/100, stol)
AR, C = right_ortho(AL, C, tol/100, stol)

while (ep > tol or D < Dmax) and count < 5000:
    print(count)
    print('AL', AL.shape)
    print('AR', AR.shape)
    print('C', C.shape)

    if ep < tol and delta_D != 0:
        AL, AR, C, Hl, Hr = dynamic_expansion(AL, AR, C, Hl, Hr, h, delta_D)
        D = D + delta_D

        print('AL new', AL.shape)
        print('AR new', AR.shape)
        print('C new', C.shape)
        print('Hl new', Hl.shape)
        print('Hr new', Hr.shape)

    AL, AR, C, Hl, Hr, e, epl, epr = vumps(AL, AR, C, h, Hl, Hr, ep)

    checks(AL, AR, C)
    print('energy', e)
    print('epl', epl)
    print('epr', epr)

    ep = np.maximum(epl, epr)

    print('ep ', ep)
    print()

    energy.append(e)
    error.append(ep)

    count += 1

print('final AL', AL.shape)
print('final AR', AR.shape)
print('V, V2, g:', V, V2, g)

#AL, C = left_ortho(AR, C, tol/100, stol)
checks(AL, AR, C)

correlation_length = my_corr_length(AL, C, tol/100)
print('correlation length', correlation_length)

vonneumann = calc_entent(C)
print('entanglement entropy', *vonneumann)

disc_weight = calc_discard_weight(AL, AR, C, h, Hl, Hr)
print('discarded weight', disc_weight)

qm, momentum = calc_momentum(AL, AR, C, sp, sm, -sz, N)
qs, stat_struc_fact = calc_stat_struc_fact(AL, AR, C, n, n, None, N)

params = stats.linregress(qs[:8], stat_struc_fact[:8])
print('K = ', (2 * np.pi * params.slope))
print('R = ', params.rvalue)

params = stats.linregress(qs[:16], stat_struc_fact[:16])
print('K = ', (2 * np.pi * params.slope))
print('R = ', params.rvalue)

params = stats.linregress(qs[:32], stat_struc_fact[:32])
print('K = ', (2 * np.pi * params.slope))
print('R = ', params.rvalue)

plt.plot(np.array(energy).real)
plt.grid(); plt.show()

plt.plot(np.array(error))
plt.yscale('log'); plt.grid(); plt.show()

model = 'tVV2'
qm /= np.pi
qs /= np.pi

filename = "%s_momentum_%.2f_%.2f_%03i_.dat" % (model, V, V2, D)
# np.savetxt(filename, np.column_stack((qm, momentum)), fmt='%s %s')
plt.plot(qm, momentum, 'x')
plt.grid(); plt.show()

filename = "%s_statstrucfact_%.2f_%.2f_%03i_.dat" % (model, V, V2, D)
# np.savetxt(filename, np.column_stack((qs, stat_struc_fact)), fmt='%s %s')
plt.plot(qs, stat_struc_fact, 'x')
plt.grid(); plt.show()

path = '/Users/joshuabaktay/Desktop/code/vumps'
# path = '/home/baktay.j/vumps/data'

# filename = "%s_energy_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), energy)

# filename = "%s_error_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), error)

# filename = "%s_discweight_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), disc_weight)

# filename = "%s_statstrucfact_%.2f_%.2f_%i_.dat" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), np.column_stack((qs, stat_struc_fact)))

# filename = "%s_momentum_%.2f_%.2f_%i_.dat" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), np.column_stack((qm, momentum)))

# filename = "%s_AL_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# with open(os.path.join(path, filename), 'a') as outfile:
#     for data_slice in AL:
#         np.savetxt(outfile, data_slice)

# filename = "%s_AR_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# with open(os.path.join(path, filename), 'a') as outfile:
#     for data_slice in AR:
#         np.savetxt(outfile, data_slice)

# filename = "%s_C_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), C)






