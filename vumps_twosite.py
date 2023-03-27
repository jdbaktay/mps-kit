#cd Desktop/all/research/code/dMPS-TDVP

import multiprocessing
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

from scipy import stats

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

        tensors = [AL, X, h, AL.conj()]
        indices = [(4,1,2), (5,2,-3,-4), (3,-1,4,5),(3,1,-2)]
        contord = [1,2,3,4,5]
        H1 = nc.ncon(tensors,indices,contord)

        tensors = [X, h]
        indices = [(1,-2,2,-4), (-1,-3,1,2)]
        contord = [1,2]
        H2 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(-1,-2,4,2), (5,2,1), (-3,3,4,5), (3,-4,1)]
        contord = [1,2,3,4,5]
        H3 = nc.ncon(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2,1), (-1,1,-3,-4)]
        H4 = nc.ncon(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1,-2,-3,1), (1,-4)]
        H5 = nc.ncon(tensors,indices)
        return H1 + H2 + H3 + H4 + H5

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
        AL.append(np.pad(AL_new[i,:,:], pad_width=((0, delta_D), (0, t)), 
                                        mode='constant')
        )

    for i in range(AR_new.shape[0]):
        AR.append(np.pad(AR_new[i,:,:], pad_width=((0, t), (0, delta_D)), 
                                        mode='constant')
        )

    C = np.pad(C, pad_width=((0, delta_D), (0, delta_D)), mode='constant')
    Hl = np.pad(Hl, pad_width=((0, delta_D), (0, delta_D)), mode='minimum')
    Hr = np.pad(Hr, pad_width=((0, delta_D), (0, delta_D)), mode='minimum')
    return np.array(AL), np.array(AR), C, Hl, Hr

def calc_discard_weight(AL, AR, C, h, Hl, Hr):
    def eff_ham(X):
        X = X.reshape(d, D, d, D)

        tensors = [AL, X, h, AL.conj()]
        indices = [(4,1,2), (5,2,-3,-4), (3,-1,4,5),(3,1,-2)]
        contord = [1,2,3,4,5]
        H1 = nc.ncon(tensors,indices,contord)

        tensors = [X, h]
        indices = [(1,-2,2,-4), (-1,-3,1,2)]
        contord = [1,2]
        H2 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(-1,-2,4,2), (5,2,1), (-3,3,4,5), (3,-4,1)]
        contord = [1,2,3,4,5]
        H3 = nc.ncon(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2,1), (-1,1,-3,-4)]
        H4 = nc.ncon(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1,-2,-3,1), (1,-4)]
        H5 = nc.ncon(tensors,indices)
        return (H1 + H2 + H3 + H4 + H5).ravel()

    two_site = nc.ncon([AL, C, AR], [(-1,-2,1),(1,2),(-3,2,-4)])

    H = spspla.LinearOperator((d * D * d * D, d * D * d * D), 
                              matvec=eff_ham
                              )

    w, v = spspla.eigsh(H, 
                        k=1,
                        which='SR', 
                        v0=two_site.ravel(), 
                        tol=1e-12, return_eigenvectors=True
                        )

    s = spla.svdvals(v[:,0].reshape(d * D, d * D))

    t = 0
    for i in range(D, d * D):
        t += s[i]**2        
    return t

def HeffTerms(AL, AR, C, h, Hl, Hr, ep):
    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(2, 7, 1), (3, 1, -2), (4, 5, 2, 3), (4, 7, 6), (5, 6, -1)]
    contord = [7, 2, 4, 1, 3, 6, 5]
    hl = nc.ncon(tensors, indices, contord)
    el = np.trace(hl @ C @ C.T.conj())

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(2, -1, 1), (3, 1, 7), (4, 5, 2, 3), (4, -2, 6), (5, 6, 7)]
    contord = [7, 3, 5, 1, 2, 6, 4]
    hr = nc.ncon(tensors, indices, contord)
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
        XT = (AL.conj().transpose(2, 1, 0).reshape(D, D * d) 
            @ t.reshape(D * d, D)
            )

        XR = np.trace(X @ C @ C.T.conj()) * np.eye(D)
        return (X - XT + XR).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        t = AR.reshape(d * D, D) @ X
        t = t.reshape(d, D, D).transpose(1, 2, 0).reshape(D, D * d)
        XT = t @ AR.conj().transpose(2, 0, 1).reshape(D * d, D)

        XL = np.trace(C.T.conj() @ C @ X) * np.eye(D)
        return (X - XT + XL).ravel()

    Ol = spspla.LinearOperator((D**2, D**2), matvec=left_env)
    Or = spspla.LinearOperator((D**2, D**2), matvec=right_env)

    Hl, _ = spspla.gmres(Ol, hl.ravel(), 
                         x0=Hl.ravel(), tol=ep/100, atol=ep/100
                         )

    Hr, _ = spspla.gmres(Or, hr.ravel(), 
                         x0=Hr.ravel(), tol=ep/100, atol=ep/100
                         )

    Hl, Hr = Hl.reshape(D, D), Hr.reshape(D, D)
    print('Hl == Hl+', spla.norm(Hl - Hl.T.conj()))
    print('Hr == Hr+', spla.norm(Hr - Hr.T.conj()))

    Hl = 0.5 * (Hl + Hl.T.conj())
    Hr = 0.5 * (Hr + Hr.T.conj())

    print('(L|hr)', np.trace(C.T.conj() @ C @ hr))
    print('(hl|R)', np.trace(hl @ C @ C.T.conj()))

    print('(L|Hr)', np.trace(C.T.conj() @ C @ Hr))
    print('(Hl|R)', np.trace(Hl @ C @ C.T.conj()))
    return Hl, Hr, e

def Apply_HC(AL, AR, h, Hl, Hr, X):
    X = X.reshape(D, D)

    t = AL.reshape(d * D, D) @ X @ AR.transpose(1, 0, 2).reshape(D, d * D)
    t = (h.reshape(d**2, d**2) 
       @ t.reshape(d, D, d, D).transpose(0, 2, 1, 3).reshape(d**2, D * D)
       )

    t = t.reshape(d, d, D, D).transpose(0, 2, 1, 3).reshape(d * D, d * D)

    H1 = (AL.conj().transpose(2, 0, 1).reshape(D, d * D) 
        @ t @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)
        )

    H2 = Hl @ X
    H3 = X @ Hr
    return (H1 + H2 + H3).ravel()

def Apply_HAC(hL_mid, hR_mid, Hl, Hr, X):
    X = X.reshape(D, d, D)

    t = hL_mid.reshape(D * d, D * d) @ X.reshape(D * d, D)
    H1 = t.reshape(D, d, D)

    t = X.reshape(D, d * D) @ hR_mid.reshape(d * D, d * D).transpose(1, 0)
    H2 = t.reshape(D, d, D)

    t = Hl @ X.reshape(D, d * D)
    H3 = t.reshape(D, d, D)

    t = X.reshape(D * d, D ) @ Hr
    H4 = t.reshape(D, d, D)
    return (H1 + H2 + H3 + H4).ravel()

def calc_new_A(AL, AR, AC, C):
    Al = AL.reshape(d * D, D)
    Ar = AR.transpose(1, 0, 2).reshape(D, d * D)

    def calcnullspace(n):
        u, s, vh = spla.svd(n, full_matrices=True)

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

    tensors = [AL, h, AL.conj()]
    indices = [(2, 1, -3), (3, -2, 2, -4), (3, 1, -1)]
    contord = [1, 2, 3]
    hL_mid = nc.ncon(tensors, indices, contord)

    tensors = [AR, h, AR.conj()]
    indices = [(2, -4, 1), (-1, 3, -3, 2), (3, -2, 1)]
    contord = [1, 2, 3]
    hR_mid = nc.ncon(tensors, indices, contord)

    f = functools.partial(Apply_HC, AL, AR, h, Hl, Hr)
    g = functools.partial(Apply_HAC, hL_mid, hR_mid, Hl, Hr)

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

def calc_lfp(A, B, o3):
    def left_transfer_op(X):
        tensors = [X.reshape(D, D), A, o3, B.conj()]
        indices = [(4, 5), (2, 5, -2), (1, 2), (1, 4, -1)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    wl, lfp_AB = spspla.eigs(E, k=1, which='LM', tol=1e-14)

    lfp_AB = lfp_AB.reshape(D, D)

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

def nonuniform_mesh(npts_left, npts_mid, npts_right, k0, dk):
    k_left, k_right = k0 - dk, k0 + dk
    mesh = np.concatenate((
            np.linspace(0.0,     k_left,  npts_left, endpoint=False),
            np.linspace(k_left,  k_right, npts_mid,  endpoint=False),
            np.linspace(k_right, 1.0,     npts_right)
            ))
    # print(mesh)

    # momenta = mesh * np.pi
    # print(momenta)
    return mesh

def my_corr_length(A, X0, tol):
    '''NEED TO SAVE ALL THREE EIGENVALUES TO DO FES.'''
    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), A.conj()]
        indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    # k must be LARGER THAN OR EQUAL TO 2 so return statement makes sense
    evals = spspla.eigs(E, k=4, which="LM", v0=X0, tol=tol, 
                           return_eigenvectors=False
                           )
    print('argmax', np.argmax(evals), evals)
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

##############################################################################
energy, error, discard_weight = [], [], []

count, tol, stol, ep = 0, 1e-12, 1e-12, 1e-2

model, d, D = str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
x, y, z = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

params = (model, z, D)

Dmax, delta_D = 0, 0

if model == 'halfXXZ':
    h = hamiltonians.XYZ_half(x, y, z, size='two')

if model == 'TFI':
    h = hamiltonians.TFI(y, z, size='two')

if model == 'oneXXZ':
    h = hamiltonians.XYZ_one(x, y, z, size='two')

if model == 'tV':
    h = hamiltonians.tV(x, y, z)

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

h = h.reshape(d, d, d, d)

A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
C = np.random.rand(D, D) - 0.5
Hl, Hr = np.eye(D, dtype=A.dtype), np.eye(D, dtype=A.dtype)

mps = AL, AR, C = mps_tools.mix_gauge(A, C, tol, stol)
mps_tools.checks(*mps)

AL, AR, C, Hl, Hr, *_ = vumps(AL, AR, C, h, Hl, Hr, ep)

AL, C = mps_tools.left_gauge(AR, C, tol / 100, stol)
AR, C = mps_tools.right_gauge(AL, C, tol / 100, stol)

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

    mps_tools.checks(AL, AR, C)
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
mps_tools.checks(AL, AR, C)

correlation_length = my_corr_length(AL, C, tol/100)
print('correlation length', correlation_length)

vonneumann = calc_entent(C)
print('entanglement entropy', *vonneumann)

disc_weight = calc_discard_weight(AL, AR, C, h, Hl, Hr)
print('discarded weight', disc_weight)

density = calc_expectations(AL, AR, C, n)
print('density', density)

# plt.plot(spla.svdvals(C), 'x', label='Schmidt vals')

# plt.title('s=1/2, %s, z=%.2f, D=%i ' % params)
# plt.grid()
# plt.legend()
# plt.yscale('log')
# plt.show()

N = int(np.floor(correlation_length))
print('N for scf', N)

qs = np.linspace(0, 1, N) * np.pi
ssf = calc_scf(AL, AR, C, n, n, si, qs)

qs /= np.pi

plt.plot(qs, ssf, 'x')
plt.grid()
plt.show()

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

# vals = stats.linregress(qs[:8], stat_struc_fact[:8])
# print('K = ', (2 * np.pi * vals.slope))
# print('R = ', vals.rvalue)

# vals = stats.linregress(qs[:16], stat_struc_fact[:16])
# print('K = ', (2 * np.pi * vals.slope))
# print('R = ', vals.rvalue)

# vals = stats.linregress(qs[:32], stat_struc_fact[:32])
# print('K = ', (2 * np.pi * vals.slope))
# print('R = ', vals.rvalue)

# plt.plot(np.array(energy).real)
# plt.grid(); plt.show()

# plt.plot(np.array(error))
# plt.yscale('log'); plt.grid(); plt.show()

exit()

path = '/Users/joshuabaktay/Desktop/code/vumps'

# filename = "%s_energy_%.2f_%.2f_%.2f_%03i_.txt" % params
# np.savetxt(os.path.join(path, filename), energy)

# filename = "%s_error_%.2f_%.2f_%.2f_%03i_.txt" % params
# np.savetxt(os.path.join(path, filename), error)

# filename = "%s_entent_%.2f_%.2f_%.2f_%03i_.txt" % params
# np.savetxt(os.path.join(path, filename), np.array(vonneumann))

# filename = "%s_evals_%.2f_%.2f_%.2f_%03i_.txt" % params
# np.savetxt(os.path.join(path, filename), np.array(evals))

# filename = "%s_discweight_%.2f_%.2f_%i_.txt" % params
# np.savetxt(os.path.join(path, filename), disc_weight)

# filename = "%s_statstrucfact_%.2f_%.2f_%.2f_%03i_.dat" % params
# np.savetxt(os.path.join(path, filename), 
#            np.column_stack((qs, stat_struc_fact)), 
#            fmt='%s %s'
#            )

# filename = "%s_momentum_%.2f_%.2f_%.2f_%03i_.dat" % params
# np.savetxt(os.path.join(path, filename), 
#            np.column_stack((qm, momentum)), 
#            fmt='%s %s'
#            )

filename = "%s_AL_%.2f_%03i_.txt" % params
with open(os.path.join(path, filename), 'a') as outfile:
    for data_slice in AL:
        np.savetxt(outfile, data_slice)

filename = "%s_AR_%.2f_%03i_.txt" % params
with open(os.path.join(path, filename), 'a') as outfile:
    for data_slice in AR:
        np.savetxt(outfile, data_slice)

filename = "%s_C_%.2f_%03i_.txt" % params
np.savetxt(os.path.join(path, filename), C)











