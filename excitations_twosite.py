#Excitations

import ncon as nc
import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import functools
import canon_forms
import sys

def calc_nullspace(n):
    u, s, vh = spla.svd(n, full_matrices=True)
    nullspace = vh.conj().T[:, D:]
    return nullspace

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
    # print('hl == hl+', spla.norm(hl - hl.T.conj()))
    # print('hr == hr+', spla.norm(hr - hr.T.conj()))

    hl = 0.5 * (hl + hl.T.conj())
    hr = 0.5 * (hr + hr.T.conj())

    Hl -= np.trace(Hl @ C @ C.T.conj()) * np.eye(D)
    Hr -= np.trace(C.T.conj() @ C @ Hr) * np.eye(D)

    def left_env(X):
        X = X.reshape(D, D)

        t = X @ AL.transpose(1, 0, 2).reshape(D, d * D)
        XT = (AL.conj().transpose(2, 1, 0).reshape(D, D * d) 
              @ t.reshape(D * d, D))

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

def EffectiveH(AL, AR, Hl, Hr, 
               VL, h, p, Y):

    ### Compute B
    tensors = [VL.reshape(d, D, D), Y.reshape(D, D)]
    indices = [(-2, -1, 1), (1, -3)]
    contord = [1]
    B = nc.ncon(tensors, indices, contord)

    ### Compute RB
    tensors = [B, AR.conj()]
    indices = [(-1, 1, 2), (1, -2, 2)]
    contord = [1, 2]
    t1 = nc.ncon(tensors, indices, contord)

    RB = right_vector_solver(t1, p)

    ### Compute L1
    tensors = [Hl, B, AL.conj()]
    indices = [(1, 2), (2, 3, -2), (3, 1, -1)]
    contord = [1, 2]
    L1_0 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AL.conj(), AL.conj()]
    indices = [(3, 6, 7), (7, 4, -2), (1, 2, 3, 4), 
               (1, 6, 5), (2, 5, -1)]
    contord = [5, 6, 7, 1, 2, 3, 4]
    L1_1 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, h, AL.conj(), AL.conj()]
    indices = [(7, 3, 5), (4, 5, -3), 
               (1, 2, 3, 4), (1, 7, 6), (2, 6, -1)]
    contord = [7, 5, 6, 1, 2, 4]
    L1_2 = nc.ncon(tensors, indices, contord)

    t2 = (L1_0
        + L1_1
        + np.exp(-1j * p) * L1_2
        )

    L1 = left_vector_solver(t2, p)

    ### Compute R1
    tensors = [B, Hr, AR.conj()]
    indices = [(-1, 3, 1), (1, 2), (3, -2, 2)]
    contord = [1, 2, 3]
    R1_0 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, h, AR.conj(), AR.conj()]
    indices = [(-1, 3, 5), (4, 5, 6), (1, 2, 3, 4), 
               (1, -2, 7), (2, 7, 6)]
    contord = [5, 6, 7, 1, 2, 3, 4]
    R1_1 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AR.conj(), AR.conj()]
    indices = [(3, -1, 6), (6, 4, 5), 
               (1, 2, 3, 4), (1, -2, 7), (2, 7, 5)]
    contord = [5, 6, 7, 1, 2, 3, 4]
    R1_2 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, h, RB, AR.conj(), AR.conj()]
    indices = [(3, -1, 7), (4, 7, 5), 
               (1, 2, 3, 4), (5, 6), (1, -2, 8), (2, 8, 6)]
    contord = [5, 6, 7, 8, 1, 2, 3, 4]
    R1_3 = nc.ncon(tensors, indices, contord)

    t3 = (R1_0
        + R1_1
        + np.exp(+1j * p) * R1_2
        + np.exp(+2j * p) * R1_3
        )
        
    R1 = right_vector_solver(t3, p)

    ### Compute Heff
    tensors = [B, AR, h, AR.conj()]
    indices = [(-1, 3, 5), (4, 5, 6), (-2, 2, 3, 4), (2, -3, 6)]
    contord = [5, 6, 2, 3, 4]
    H_0 = nc.ncon(tensors, indices, contord)

    tensors = [B, AR, h, AL.conj()]
    indices = [(5, 3, 6), (4, 6, -3), (1, -2, 3, 4), (1, 5, -1)]
    contord = [5, 6, 1, 3, 4]
    H_1 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AR.conj()]
    indices = [(3, -1, 6), (6, 4, 5), (-2, 2, 3, 4), (2, -3, 5)]
    contord = [5, 6, 2, 3, 4]
    H_2 = nc.ncon(tensors, indices, contord)

    tensors = [AL, B, h, AL.conj()]
    indices = [(3, 5, 6), (6, 4, -3), (1, -2, 3, 4), (1, 5, -1)]
    contord = [5, 6, 1, 3, 4]
    H_3 = nc.ncon(tensors, indices, contord)

    tensors = [B, Hr]
    indices = [(-1, -2, 1), (1, -3)]
    contord = [1]
    H_4 = nc.ncon(tensors, indices, contord)

    tensors = [Hl, B]
    indices = [(-1, 1), (1, -2, -3)]
    contord = [1]
    H_5 = nc.ncon(tensors, indices, contord)

    tensors = [L1, AR]
    indices = [(-1, 1), (-2, 1, -3)]
    contord = [1]
    H_6 = nc.ncon(tensors, indices, contord)

    tensors = [AL, R1]
    indices = [(-2, -1, 1), (1, -3)]
    contord = [1]
    H_7 = nc.ncon(tensors, indices, contord)

    tensors = [Hl, AL, RB]
    indices = [(-1, 1), (-2, 1, 2), (2, -3)]
    contord = [1, 2]
    H_8 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, RB, h, AL.conj()]
    indices = [(3, 5, 6), (4, 6, 7), (7, -3), (1, -2, 3, 4), (1, 5, -1,)]
    contord = [5, 6, 7, 1, 3, 4]
    H_9 = nc.ncon(tensors, indices, contord)

    tensors = [AL, AL, h, RB, AR.conj()]
    indices = [(3, -1, 7), (4, 7, 5), (-2, 2, 3, 4), (5, 6), (2, -3, 6)]
    contord = [5, 6, 7, 2, 3, 4]
    H_10 = nc.ncon(tensors, indices, contord)

    H = (H_0
       + np.exp(-1j * p) * H_1
       + np.exp(+1j * p) * H_2
       + H_3
       + H_4
       + H_5
       + np.exp(-1j * p) * H_6
       + np.exp(+1j * p) * H_7
       + np.exp(+1j * p) * H_8
       + np.exp(+1j * p) * H_9
       + np.exp(+2j * p) * H_10
       )

    tensors = [H, VL.reshape(d, D, D).conj()]
    indices = [(2, 1, -2), (1, 2, -1)]
    contord = [2, 1]
    Y = nc.ncon(tensors, indices, contord)
    return Y.ravel()

def quasiparticle(AL, AR, C, Hl, Hr, h, p, N, eta):
    Hl, Hr = HeffTerms(AL, AR, C, h, Hl, Hr, eta)

    ### Compute nullspace

    VL = calc_nullspace(AL.conj().transpose(2, 0, 1).reshape(D, d * D))

    ### Solve eigenvalue problem
    f = functools.partial(EffectiveH, AL, AR, Hl, Hr, VL, h, p)

    H = spspla.LinearOperator((D * D, D * D), matvec=f)

    rand_init = np.random.rand(D, D) - 0.5
    w, v = spspla.eigsh(H, k=N, which='SR', 
                           v0=rand_init.ravel(), tol=eta)
    return w, v

########################################################################
energy = []

D, d = int(sys.argv[2]), 2

stol, tol = 1e-12, 1e-12

si = np.array([[1, 0],[0, 1]])
sx = np.array([[0, 1],[1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

sp = 0.5 * (sx + 1.0j*sy)
sm = 0.5 * (sx - 1.0j*sy)
n = 0.5 * (sz + np.eye(d))

x, y, z = 1, 1, float(sys.argv[1])
XYZ = (- x / 4 * np.kron(sx, sx) 
       - y / 4 * np.kron(sy, sy)
       + z / 4 * np.kron(sz, sz) 
      )

t, V = 1, float(sys.argv[1])
tV = (- t / 2 * (np.kron(sx, sx) + np.kron(sy, sy)) 
      + V / 4 * np.kron(sz, sz)
      )

A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
C = np.random.rand(D, D) - 0.5

# AL = np.loadtxt('XXZ_AL_0.00_016_.txt', dtype=complex).reshape(d, D, D)
# AR = np.loadtxt('XXZ_AR_0.00_016_.txt', dtype=complex).reshape(d, D, D)
# C = np.loadtxt('XXZ_C_0.00_016_.txt', dtype=complex)

h = XYZ

e = nc.ncon([AL, C, AR, h.reshape(d, d, d, d), 
             AL.conj(), C.conj(), AR.conj()],
            [(3, 5, 6), (6, 7), (4, 7, 8), (1, 2, 3, 4), 
             (1, 5, 10), (10, 9), (2, 9, 8)],
            [5, 6, 7, 8, 9, 10, 1, 2, 3, 4]
            )
print('gs energy', e)

h = h - e * np.eye(d**2)
h = h.reshape(d, d, d, d)

Hl, Hr = np.eye(D, dtype=A.dtype), np.eye(D, dtype=A.dtype)

mps = AL, AR, C #= canon_forms.mix_gauge(A, C, tol=tol, stol=stol)
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





