import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools

from quspin.tools.lanczos import lanczos_iter, expm_lanczos
from .gs_tools import *
from .canonical_forms import *

def tdvp_two(AL, AR, C, Hl, Hr, h, dt, ep):
    d = AL.shape[0]
    D = C.shape[0]

    h = h.reshape(d, d, d, d)

    AC = np.tensordot(C, AR, axes=(1, 1))

    Hl, Hr, e = HeffTerms_two(AL, AR, C, Hl, Hr, h, ep)

    tensors = [AL, h, AL.conj()]
    indices = [(2, 1, -3), (3, -2, 2, -4), (3, 1, -1)]
    contord = [1, 2, 3]
    hL_mid = nc.ncon(tensors, indices, contord)

    tensors = [AR, h, AR.conj()]
    indices = [(2, -4, 1), (-1, 3, -3, 2), (3, -2, 1)]
    contord = [1, 2, 3]
    hR_mid = nc.ncon(tensors, indices, contord)

    f = functools.partial(Apply_HC_two, AL, AR, Hl, Hr, h, D, d)
    g = functools.partial(Apply_HAC_two, hL_mid, hR_mid, Hl, Hr, D, d)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)
    E, V, Q_T = lanczos_iter(H, C.ravel(), 20)
    C = expm_lanczos(E, V, Q_T, a=-dt)
    C = C.reshape(D, D)
    C /= spla.norm(C)
    print('NORM OF C', spla.norm(C))

    H = spspla.LinearOperator((D * d * D, D * d * D), matvec=g)
    E, V, Q_T = lanczos_iter(H, AC.ravel(), 20)
    AC = expm_lanczos(E, V, Q_T, a=-dt)
    AC = AC.reshape(D, d, D)
    AC /= spla.norm(AC)
    print('NORM OF AC', spla.norm(AC))

    epl, epr, AL, AR = calc_new_A(AL, AR, AC, C)
    return AL, AR, C, Hl, Hr, e, epl, epr

def tdvp_three(AL, AR, C, Hl, Hr, h, dt, ep):
    d = AL.shape[0]
    D = C.shape[0]

    h = h.reshape(d, d, d, d, d, d)

    AC = np.tensordot(C, AR, axes=(1, 1))

    Hl, Hr, e = HeffTerms_three(AL, AR, C, Hl, Hr, h, ep)

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(4, 7, 8), (5, 8, -3), (1, 2, -2, 4, 5, -4), (1, 7, 9), (2, 9, -1)]
    contord = [7, 8, 9, 1, 2, 4, 5]
    hl_mid = nc.ncon(tensors,indices,contord)

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(5, -3, 8), (6, 8, 7), (-1, 2, 3, -4, 5, 6), (2, -2, 9), (3, 9,7 )]
    contord = [7, 8, 9, 2, 3, 5, 6]
    hr_mid = nc.ncon(tensors,indices,contord)

    f = functools.partial(Apply_HC_three, hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d)
    g = functools.partial(Apply_HAC_three, hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)
    E, V, Q_T = lanczos_iter(H, C.ravel(), 20)
    C = expm_lanczos(E, V, Q_T, a=-dt)
    C = C.reshape(D, D)
    C /= spla.norm(C)
    print('NORM OF C', spla.norm(C))

    H = spspla.LinearOperator((D * d * D, D * d * D), matvec=g)
    E, V, Q_T = lanczos_iter(H, AC.ravel(), 20)
    AC = expm_lanczos(E, V, Q_T, a=-dt)
    AC = AC.reshape(D, d, D)
    AC /= spla.norm(AC)
    print('NORM OF AC', spla.norm(AC))

    epl, epr, AL, AR = calc_new_A(AL, AR, AC, C)
    return AL, AR, C, Hl, Hr, e, epl, epr

def tdvp(AL, AR, C, h, dt, tol, stol, size):
    D = C.shape[0]

    AL = AL.transpose(1, 0, 2)
    AR = AR.transpose(1, 0, 2)

    energy, error = [], []

    count, ep = 0, 1e-2

    Hl, Hr = np.eye(D, dtype=AL.dtype), np.eye(D, dtype=AR.dtype)

    if size == 'two':
        AL, AR, C, Hl, Hr, *_ = tdvp_two(AL, AR, C, Hl, Hr, h, dt, ep)
    if size == 'three':
        AL, AR, C, Hl, Hr, *_ = tdvp_three(AL, AR, C, Hl, Hr, h, dt, ep)

    AL, C = left_gauge(AR, C, tol / 100, stol)
    AR, C = right_gauge(AL, C, tol / 100, stol)

    while ep > tol and count < 5000:
        print(count)
        print('AL', AL.shape)
        print('AR', AR.shape)
        print('C', C.shape)

        if size == 'two':
            AL, AR, C, Hl, Hr, e, epl, epr = tdvp_two(AL, AR, C, Hl, Hr, h, dt, ep)
        if size == 'three':
            AL, AR, C, Hl, Hr, e, epl, epr = tdvp_three(AL, AR, C, Hl, Hr, h, dt, ep)

        gauge_checks(AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)
        print('energy', e)
        print('epl', epl)
        print('epr', epr)

        ep = np.maximum(epl, epr)

        print('ep ', ep)
        print()

        energy.append(e)
        error.append(ep)

        count += 1

    gs_mps = (AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C)
    return energy, error, gs_mps
