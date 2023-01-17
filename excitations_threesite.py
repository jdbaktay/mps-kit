#Excitations

import ncon as nc
import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import functools
import canon_forms
import hamiltonians
import sys

def calc_nullspace(n):
    u, s, vh = spla.svd(n, full_matrices=True)
    nullspace = vh.conj().T[:, D:]
    return nullspace

def HeffTerms(AL, AR, C, h, Hl, Hr, ep):
    tensors = [AL, AL, AL, h, AL.conj(), AL.conj(), AL.conj()]
    indices = [(4, 7, 8), (5, 8, 10), (6, 10, -2), (1, 2, 3, 4, 5, 6), 
               (1, 7, 9), (2, 9, 11), (3, 11, -1)]
    contord = [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6]
    hl = nc.ncon(tensors,indices,contord)
    el = np.trace(hl @ C @ C.T.conj())

    tensors = [AR, AR, AR, h, AR.conj(), AR.conj(), AR.conj()]
    indices = [(4, -1, 10), (5, 10, 8), (6, 8, 7), (1, 2, 3, 4, 5, 6), 
               (1, -2, 11), (2, 11, 9), (3, 9, 7)]
    contord = [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6]
    hr = nc.ncon(tensors,indices,contord)
    er = np.trace(C.T.conj() @ C @ hr)

    e = 0.5 * (el + er)

    hl -= el * np.eye(D)
    hr -= er * np.eye(D)
    # print('hl == hl+', spla.norm(hl - hl.T.conj()))
    # print('hr == hr+', spla.norm(hr - hr.T.conj()))

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
    # print('Hl == Hl+', spla.norm(Hl - Hl.T.conj()))
    # print('Hr == Hr+', spla.norm(Hr - Hr.T.conj()))

    Hl = 0.5 * (Hl + Hl.T.conj())
    Hr = 0.5 * (Hr + Hr.T.conj())

    # print('(L|hr)', np.trace(C.T.conj() @ C @ hr))
    # print('(hl|R)', np.trace(hl @ C @ C.T.conj()))

    # print('(L|Hr)', np.trace(C.T.conj() @ C @ Hr))
    # print('(Hl|R)', np.trace(Hl @ C @ C.T.conj()))
    return Hl, Hr

def left_vector_solver(O, p):
    def left_env(X):
        X = X.reshape(D, D)

        t = X @ AR.transpose(1, 0, 2).reshape(D, d * D)
        XT = (AL.conj().transpose(2, 1, 0).reshape(D, D * d) 
               @ t.reshape(D * d, D))

        return (X - np.exp(-1.0j * p) * XT).ravel()

    left_env_op = spspla.LinearOperator((D * D, D * D), matvec=left_env)

    rand_init = np.random.rand(D, D) - 0.5
    left_vec, _ = spspla.gmres(left_env_op, O.ravel(), 
                               x0=rand_init.ravel(), 
                               tol=10**-12, 
                               atol=10**-12
                               )

    return left_vec.reshape(D, D)

def right_vector_solver(O, p):
    def right_env(X):
        X = X.reshape(D, D)

        t = AL.reshape(d * D, D) @ X
        t = t.reshape(d, D, D).transpose(1, 0, 2).reshape(D, d * D)
        XT = t @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)
        return (X - np.exp(+1.0j * p) * XT).ravel()

    right_env_op = spspla.LinearOperator((D * D, D * D), matvec=right_env)


    rand_init = np.random.rand(D, D) - 0.5
    right_vec, _ = spspla.gmres(right_env_op, O.ravel(), 
                                x0=rand_init.ravel(), 
                                tol=10**-12, 
                                atol=10**-12
                                )

    return right_vec.reshape(D, D)

def EffectiveH(AL, AR, Hl, Hr, L1_tensors, R1_tensors, H_tensors,
               VL, h, p, Y):
    ### Compute B
    B = (VL @ Y.reshape(D, D)).reshape(d, D, D).transpose(1, 0, 2)

    ### Compute RB
    t1 = (B.reshape(D, d * D) 
       @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)
       )

    RB = right_vector_solver(t1, p)

    ### Compute L1
    L1_0 = L1_tensors[0] @ B.reshape(D * d, D)

    L1_1 = L1_tensors[1] @ B.reshape(D * d, D)

    ### Compute L1_2
    AL, AR = AL.transpose(1, 0, 2), AR.transpose(1, 0, 2)

    t = AL.reshape(D * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 2, 4)
    t = t.reshape(d * D, d * d * D)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, d * D)
    t = AL.conj().transpose(2, 0, 1).reshape(D, D * d) @ t
    t = t.reshape(D * d, D)
    t = AL.conj().transpose(2, 0, 1).reshape(D, D * d) @ t
    L1_2 = t

    ### Compute L1_3
    t = B.reshape(D * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 2, 4)
    t = t.reshape(d * D, d * d * D)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, d * D)
    t = AL.conj().transpose(2, 0, 1).reshape(D, D * d) @ t
    t = t.reshape(D * d, D)
    t = AL.conj().transpose(2, 0, 1).reshape(D, D * d) @ t
    L1_3 = t

    t2 = (L1_0
    	+ L1_1
    	+ np.exp(-1j * p) * L1_2
    	+ np.exp(-2j * p) * L1_3
    	)

    L1 = left_vector_solver(t2, p)

    ### Compute R1
    R1_0 = B.reshape(D, d * D) @ R1_tensors[0]

    R1_1 = B.reshape(D, d * D) @ R1_tensors[1]

    ### Compute R1_2
    t = AL.reshape(D * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(3, 0, 1, 4, 2)
    t = t.reshape(D * d * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D * d, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    t = t.reshape(D, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    R1_2 = t

    ### Compute R1_3
    t = AL.reshape(D * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(3, 0, 1, 4, 2)
    t = t.reshape(D * d * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D * d, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    t = t.reshape(D, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    R1_3 = t

    ### Compute R1_4
    t = AL.reshape(D * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d * d, D) @ RB
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(3, 0, 1, 4, 2)
    t = t.reshape(D * d * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D * d, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    t = t.reshape(D, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    R1_4 = t

    t3 = (R1_0
    	+ R1_1
    	+ np.exp(+1j * p) * R1_2
    	+ np.exp(+2j * p) * R1_3
    	+ np.exp(+3j * p) * R1_4
    	)
    
    R1 = right_vector_solver(t3, p)

    ### Compute Heff

    # H_0
    H_0 = (B.reshape(D, d * D) @ H_tensors[0]).reshape(D, d, D)

    # H_1
    t = AL.reshape(D * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(3, 0, 1, 4, 2)
    t = t.reshape(D * d * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D * d, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    t = t.reshape(D, d, D)
    H_1 = t

    # H_2
    t = AL.reshape(D * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(3, 0, 1, 4, 2)
    t = t.reshape(D * d * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D * d, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    t = t.reshape(D, d, D)
    H_2 = t

    # H_3
    t = AL.reshape(D * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 4, 2)
    t = t.reshape(d * D, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D, d, D)
    H_3 = t

    # H_4
    t = AL.reshape(D * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 4, 2)
    t = t.reshape(d * D, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D, d, D)
    H_4 = t

    # H_5
    t = B.reshape(D * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 4, 2)
    t = t.reshape(d * D, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D, d, D)
    H_5 = t

    # H_6
    H_6 = (H_tensors[1] @ B.reshape(D * d, D)).reshape(D, d, D)

    # H_7
    t = AL.reshape(D * d, D) @ B.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 2, 4)
    t = t.reshape(d * D, d * d * D)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, d * D)
    t = AL.conj().transpose(2, 0, 1).reshape(D, D * d) @ t
    t = t.reshape(D, d, D)
    H_7 = t

    # H_8
    t = B.reshape(D * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AR.reshape(D, d * D)
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 2, 4)
    t = t.reshape(d * D, d * d * D)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, d * D)
    t = AL.conj().transpose(2, 0, 1).reshape(D, D * d) @ t
    t = t.reshape(D, d, D)
    H_8 = t 

    AL, AR = AL.transpose(1, 0, 2), AR.transpose(1, 0, 2)

    # H_9
    H_9 = (B.reshape(D * d, D) @ Hr).reshape(D, d, D)

    # H_10
    H_10 = (Hl @ B.reshape(D, d * D)).reshape(D, d, D)

    # H_11
    H_11 = (L1 
         @ AR.transpose(1, 0, 2).reshape(D, d * D)).reshape(D, d, D)

    # H_12
    H_12 = (AL.transpose(1, 0, 2).reshape(D * d, D) 
         @ R1).reshape(D, d, D)

    # H_13
    H_13 = (H_tensors[2] @ RB).reshape(D, d, D)

    # H_14
    H_14  = (H_tensors[3] @ RB).reshape(D, d, D)

    # H_15
    AL, AR = AL.transpose(1, 0, 2), AR.transpose(1, 0, 2)

    t = AL.reshape(D * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d * d, D) @ RB
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 4, 2)
    t = t.reshape(d * D, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, d * D) @ t
    t = t.reshape(D * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D, d, D)
    H_15 = t

    # H_16
    t = AL.reshape(D * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d, D) @ AL.reshape(D, d * D)
    t = t.reshape(D * d * d * d, D) @ RB
    t = t.reshape(D, d, d, d, D).transpose(1, 2, 3, 0, 4)
    t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
    t = t.reshape(d, d, d, D, D).transpose(3, 0, 1, 4, 2)
    t = t.reshape(D * d * d, D * d)
    t = t @ AR.conj().transpose(2, 1, 0).reshape(D * d, D)
    t = t.reshape(D * d, d * D)
    t = t @ AR.conj().transpose(1, 2, 0).reshape(d * D, D)
    t = t.reshape(D, d, D)
    H_16 = t

    H = (H_0
       + np.exp(+1j * p) * H_1
       + np.exp(+2j * p) * H_2
       + H_3
       + np.exp(+1j * p) * H_4
       + np.exp(-1j * p) * H_5
       + H_6
       + np.exp(-1j * p) * H_7
       + np.exp(-2j * p) * H_8
       + H_9
       + H_10
       + np.exp(-1j * p) * H_11
       + np.exp(+1j * p) * H_12
       + np.exp(+1j * p) * H_13
       + np.exp(+1j * p) * H_14
       + np.exp(+2j * p) * H_15
       + np.exp(+3j * p) * H_16
       )

    Y = (VL.conj().transpose(1, 0) 
       @ H.transpose(1, 0, 2).reshape(d * D, D))
    return Y.ravel()

def quasiparticle(AL, AR, C, Hl, Hr, h, p, N, eta):
    Hl, Hr = HeffTerms(AL, AR, C, h, Hl, Hr, eta)

    ### Precompute L1 terms
    tensors = [Hl, AL.conj()]
    indices = [(1, -2), (-3, 1, -1)]
    contord = [1]
    L1_pre_0 = nc.ncon(tensors, indices, contord).reshape(D, D * d)

    tensors = [AL, AL, h, AL.conj(), AL.conj(), AL.conj()]
    indices = [(4, 9, 10), (5, 10, -2), (1, 2, 3, 4, 5, -3), 
               (1, 9, 8), (2, 8, 7), (3, 7, -1)]
    contord = [7, 8, 9, 10, 1, 2, 3, 4, 5]
    L1_pre_1 = nc.ncon(tensors, indices, contord).reshape(D, D * d)

    L1_tensors = [L1_pre_0, L1_pre_1]

    ### Precompute R1 terms
    tensors = [Hr, AR.conj()]
    indices = [(-2, 1), (-1, -3, 1)]
    contord = [1]
    R1_pre_0 = nc.ncon(tensors, indices, contord).reshape(d * D, D)

    tensors = [AR, AR, h, AR.conj(), AR.conj(), AR.conj()]
    indices = [(5, -2, 8), (6, 8, 9), (1, 2, 3, -1, 5, 6), 
               (1, -3, 11), (2, 11, 10), (3, 10, 9)]
    contord = [8, 9, 10, 11, 1, 2, 3, 5, 6]
    R1_pre_1 = nc.ncon(tensors, indices, contord).reshape(d * D, D)

    R1_tensors = [R1_pre_0, R1_pre_1]

    ### Precompute H terms
    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(5, -2, 8), (6, 8, 7), (-3, 2, 3, -1, 5, 6), 
               (2, -4, 9), (3, 9, 7)]
    contord = [7, 8, 9, 2, 3, 5, 6]
    H_pre_0 = nc.ncon(tensors, indices, contord).reshape(d * D, d * D)

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(4, 7, 8), (5, 8, -3), (1, 2, -2, 4, 5, -4), 
               (1, 7, 9), (2, 9, -1)]
    contord = [7, 8, 9, 1, 2, 4, 5]
    H_pre_6 = nc.ncon(tensors, indices, contord).reshape(D * d, D * d)

    tensors = [Hl, AL]
    indices = [(-1, 1), (-2, 1, -3)]
    contord = [1]
    H_pre_8 = nc.ncon(tensors, indices, contord).reshape(D * d, D)

    tensors = [AL, AL, AL, h, AL.conj(), AL.conj()]
    indices = [(4, 8, 9), (5, 9, 10), (6, 10, -3), (1, 2, -2, 4, 5, 6),
               (1, 8, 7), (2, 7, -1)]
    contord = [7, 8, 9, 10, 1, 2, 4, 5, 6]
    H_pre_14 = nc.ncon(tensors, indices, contord).reshape(D * d, D)

    H_tensors = [H_pre_0, H_pre_6, H_pre_8, H_pre_14]

    ### Compute nullspace

    VL = calc_nullspace(AL.conj().transpose(2, 0, 1).reshape(D, d * D))

    ### Solve eigenvalue problem
    f = functools.partial(EffectiveH, AL, AR, Hl, Hr, 
                          L1_tensors, R1_tensors, H_tensors, VL, h, p)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)

    rand_init = np.random.rand(D, D) - 0.5
    w, v = spspla.eigsh(H, k=N, which='SR', 
                           v0=rand_init.ravel(), tol=eta)
    return w, v

########################################################################

energy = []

D, d = int(sys.argv[4]), 2

stol, tol = 1e-12, 1e-12

si = np.array([[1, 0],[0, 1]])
sx = np.array([[0, 1],[1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

sp = 0.5 * (sx + 1.0j * sy)
sm = 0.5 * (sx - 1.0j * sy)
n = 0.5 * (sz + np.eye(d))

AL = np.loadtxt('XXZ_AL_0.00_016_.txt', dtype=complex).reshape(d, D, D)
AR = np.loadtxt('XXZ_AR_0.00_016_.txt', dtype=complex).reshape(d, D, D)
C = np.loadtxt('XXZ_C_0.00_016_.txt', dtype=complex)

x, y, z = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
h = hamiltonians.XYZ_half(x, y, z, size='three')

tensors = [AL, C, AR, AR, h.reshape(d, d, d, d, d, d), 
           AL.conj(), C.conj(), AR.conj(), AR.conj()]
indices = [(4, 11, 12), (12, 13), (5, 13, 14), (6, 14, 7),
           (1, 2, 3, 4, 5, 6), 
           (1, 11, 10), (10, 9), (2, 9, 8), (3, 8, 7)]
contord = [7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6]
gs_energy = nc.ncon(tensors, indices, contord)
print('gs energy', gs_energy)

h = h - gs_energy * np.eye(d**3)
h = h.reshape(d, d, d, d, d, d)

Hl, Hr = np.eye(D, dtype=AL.dtype), np.eye(D, dtype=AR.dtype)

mps = AL, AR, C
canon_forms.checks(*mps)

mom_dist = np.linspace(-np.pi, np.pi, 51)

for p in mom_dist:
    print('mom', p)
    w, v = quasiparticle(AL, AR, C, Hl, Hr, h, p=p, N=1, eta=tol)
    print('excit. energy', w[0])
    print()

    energy.append(w[0])

print('w', w.shape)
print('v', v.shape)

plt.plot(mom_dist, energy)
plt.show()














