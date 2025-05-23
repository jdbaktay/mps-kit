import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools
import sys
import os
import hamiltonians
import mps_tools
import inspect
import matplotlib.pyplot as plt
from quspin.tools.lanczos import lanczos_iter, expm_lanczos

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
                         x0=Hl.ravel(), rtol=ep/100, atol=ep/100
                         )

    Hr, _ = spspla.gmres(Or, hr.ravel(), 
                         x0=Hr.ravel(), rtol=ep/100, atol=ep/100
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

def tdvp(AL, AR, C, Hl, Hr, h, dt, ep):
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

def calc_entent(C):
    s = spla.svdvals(C)

    b = -np.log(s[0])
    entent = -sum(ss**2 * np.log(ss**2) for ss in s)
    return entent, b

def calc_fidelity(X, Y):
    '''Presumes that MPS tensors X and Y are both properly normalized'''
    E = np.tensordot(X, Y.conj(), axes=(0, 0))
    E = E.transpose(0, 2, 1, 3).reshape(D * D, D * D)

    evals = spspla.eigs(E, k=4, which='LM', return_eigenvectors=False)
    return np.max(np.abs(evals))

def nonuniform_mesh(npts_left, npts_mid, npts_right, k0, dk):
    k_left, k_right = k0 - dk, k0 + dk
    mesh = np.concatenate((
            np.linspace(0.0,     k_left,  npts_left, endpoint=False),
            np.linspace(k_left,  k_right, npts_mid,  endpoint=False),
            np.linspace(k_right, 1.0,     npts_right)
            ))
    return mesh

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

########################################################################
energy, error = [], []

count, tol, stol, ep = 0, 1e-12, 1e-12, 1e-2

Dmax, delta_D = 0, 0

model = str(sys.argv[1])
d = int(sys.argv[2])
D = int(sys.argv[3])
x = float(sys.argv[4])
y = float(sys.argv[5])
z = float(sys.argv[6])
g = float(sys.argv[7])
dt = 0.2

params = (model, x, y, z, g, D)
print('input params', params)

hamiltonian_dict = {name: obj for name, obj 
                    in inspect.getmembers(hamiltonians, inspect.isfunction)}

h = hamiltonian_dict[model](x, y, z, g).reshape(d, d, d, d)


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

A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
C = np.random.rand(D, D) - 0.5
Hl, Hr = np.eye(D, dtype=A.dtype), np.eye(D, dtype=A.dtype)

mps = AL, AR, C = mps_tools.mix_gauge(A, C, tol, stol)
mps_tools.checks(*mps)

AL, AR, C, Hl, Hr, *_ = tdvp(AL, AR, C, Hl, Hr, h, dt, ep)

AL, C = mps_tools.left_gauge(AR, C, tol / 100, stol)
AR, C = mps_tools.right_gauge(AL, C, tol / 100, stol)

while (ep > tol or D < Dmax) and count < 5000:
    print(count)
    print('AL', AL.shape)
    print('AR', AR.shape)
    print('C', C.shape)

    AL, AR, C, Hl, Hr, e, epl, epr = tdvp(AL, AR, C, Hl, Hr, h, dt, ep)

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

path = ''

plt.plot(np.array(error))
plt.yscale('log')
plt.show()

exit()

filename = f'{model}_gs_{x}_{y}_{z}_{g}_{D:03}_'
np.savez(os.path.join(path, filename), AL=AL, AR=AR, C=C)
