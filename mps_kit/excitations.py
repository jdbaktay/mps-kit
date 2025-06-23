import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools

from .gs_tools import HeffTerms_two, HeffTerms_three

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

def left_vector_solver(AL, AR, O, p):
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
                        rtol=1e-12,
                        atol=1e-12
                        )
    return v.reshape(D, D)

def right_vector_solver(AL, AR, O, p):
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
                        rtol=1e-12,
                        atol=1e-12
                        )
    return v.reshape(D, D)

def Heff_two(AL, AR, C, Lh, Rh, h, p, Y):
    Y = Y.reshape((d - 1) * D, D)

    tensors = [VL, Y]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    B = nc.ncon(tensors, indices, contord)

    tensors = [B, AR.conj()]
    indices = [(-1, 1, 2), (-2, 1, 2)]
    contord = [2, 1]
    RB = nc.ncon(tensors, indices, contord)
    RB = right_vector_solver(AL, AR, RB, p)

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
    L1 = left_vector_solver(AL, AR, L1, p)

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

def Heff_three(AL, AR, C, Lh, Rh, h, p, Y):
    Y = Y.reshape((d - 1) * D, D)

    tensors = [VL, Y]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    B = nc.ncon(tensors, indices, contord)

    tensors = [B, AR.conj()]
    indices = [(-1, 1, 2), (-2, 1, 2)]
    contord = [2, 1]
    RB = nc.ncon(tensors, indices, contord)
    RB = right_vector_solver(AL, AR, RB, p)

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

    L1 = left_vector_solver(AL, AR, L1, p)

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

def quasi_particle(AL, AR, C, Lh, Rh, h, p, N, guess):
    f = functools.partial(Heff, AL, AR, C, Lh, Rh, h, p)
    H = spspla.LinearOperator(((d - 1) * D**2, (d - 1) * D**2), matvec=f)

    w, v = spspla.eigsh(H, k=N, v0=guess, which='SR', tol=tol)
    return w, v

def gs_energy(AL, AR, C, h, site):
    if site == 'two':
        tensors = [AL, C, AR, h.reshape(d, d, d, d),
               AL.conj(), C.conj(), AR.conj()]
        indices = [(5, 3, 6),  (6, 7),  (7, 4, 8), (1, 2, 3, 4),
                   (5, 1, 10), (10, 9), (9, 2, 8)]
        contord = [5, 6, 7, 8, 9, 10, 1, 2, 3, 4]

    if site == 'three':
        tensors = [AL, AL, C, AR, h.reshape(d, d, d, d, d, d),
                   AL.conj(), AL.conj(), C.conj(), AR.conj()]
        indices = [(7, 4, 8), (8, 5, 9), (9, 10), (10, 6, 11), (1, 2, 3, 4, 5, 6),
                   (7, 1, 14), (14, 2, 13), (13, 12), (12, 3, 11)]
        contord = [7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6]
    return nc.ncon(tensors, indices, contord)

def excitations(AL, AR, C, h, site, mom_vec, tol, N):
    global D, d, VL, lfp_LR, rfp_LR, lfp_RL, rfp_RL

    D = C.shape[0]
    d = AL.shape[1]

    VL = spla.null_space(AL.conj().reshape(D * d, D).T)
    VL = VL.reshape(D, d, (d - 1) * D)

    lfp_LR, rfp_LR = fixed_points(AL, AR)
    lfp_RL, rfp_RL = fixed_points(AR, AL)

    Lh, Rh = np.eye(D, dtype=AL.dtype), np.eye(D, dtype=AR.dtype)

    if site == 'two':
        Lh, Rh, _ = HeffTerms_two(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2),
                                  C, Lh, Rh, h.reshape(d, d, d, d), tol)

        h -= np.real(gs_energy(AL, AR, C, h, site)) * np.eye(d**2)
        h = h.reshape(d, d, d, d)

    if site =='three':
        Lh, Rh, _ = HeffTerms_three(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2),
                                    C, Lh, Rh, h.reshape(d, d, d, d, d, d), tol)

        h -= np.real(gs_energy(AL, AR, C, h, site)) * np.eye(d**3)
        h = h.reshape(d, d, d, d, d, d)

    excit_energy, excit_states = [], []
    for p in mom_vec:
        print('p', p)
        guess = v[:, 0] if p > 0 else None

        if site == 'two':
            f = functools.partial(Heff_two, AL, AR, C, Lh, Rh, h, p)
        if site == 'three':
            f = functools.partial(Heff_three, AL, AR, C, Lh, Rh, h, p)

        H = spspla.LinearOperator(((d - 1) * D**2, (d - 1) * D**2), matvec=f)
        w, v = spspla.eigsh(H, k=N, v0=guess, which='SR', tol=tol)

        excit_energy.append(w)
        excit_states.append(v)
        print('excit. energy', min(w))
    return np.array(excit_energy), np.array(excit_states)
