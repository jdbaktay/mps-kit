#cd Desktop/all/research/code/dMPS-TDVP

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import functools
import sys
import os

from numpy import tensordot as td
from ncon import ncon as nc 

def check(X,Y):
    return spla.norm(np.subtract(X,Y))

def calc_discard_weight(AL,AR,C,h,Hl,Hr):

    def eff_ham(X):
        X = X.reshape(d,D,d,D)

        tensors = [AL, X, h, AL.conj()]
        indices = [(4,1,2), (5,2,-3,-4), (3,-1,4,5),(3,1,-2)]
        contord = [1,2,3,4,5]
        H1 = nc(tensors,indices,contord)

        tensors = [X, h]
        indices = [(1,-2,2,-4), (-1,-3,1,2)]
        contord = [1,2]
        H2 = nc(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(-1,-2,4,2), (5,2,1), (-3,3,4,5), (3,-4,1)]
        contord = [1,2,3,4,5]
        H3 = nc(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2,1),(-1,1,-3,-4)]
        H4 = nc(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1,-2,-3,1),(1,-4)]
        H5 = nc(tensors,indices)

        return (H1+H2+H3+H4+H5).ravel()

    two_site = nc([AL, C, AR], [(-1,-2,1),(1,2),(-3,2,-4)])

    H = spspla.LinearOperator((d*D*d*D,d*D*d*D), matvec=eff_ham)

    w, v = spspla.eigs(H, k=1, which='SR', v0=two_site.ravel(), tol=1e-12, return_eigenvectors=True)

    s = spla.svdvals(v[:,0].reshape(d*D,d*D))

    t = 0
    for i in range(D,d*D):
        t += s[i]**2
        
    return t

def left_ortho(A,X0,tol):

    def left_fixed_point(A,B):
        def left_transfer_op(X):
            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1,2,-2), (3, 2), (1, 3, -1)]
            contord = [2, 3, 1]
            return nc(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D*D,D*D), matvec=left_transfer_op)

        evals, evecs = spspla.eigs(E, k=1, which="LR", v0=X0, tol=tol)

        return evals[0], evecs[:,0].reshape(D,D)

    norm, l = left_fixed_point(A,A)

    A = A/np.sqrt(norm)

    l = 0.5*(l + l.T.conj())

    l = l/np.trace(l)

    w, v = spla.eigh(l)

    L = np.sqrt(np.diag(np.abs(w)))@v.T.conj()

    Li = spla.inv(L)

    AL = nc([L, A, Li], [(-2,1), (-1,1,2), (2,-3)])

    return AL, L

def right_ortho(A, X0, tol):
    A, L = left_ortho(np.transpose(A, (0, 2, 1)), X0, tol)
    A, L = np.transpose(A, (0, 2, 1)), np.transpose(L, (1, 0))
    return A, L

def HeffTerms(AL,AR,C,h,Hl,Hr,ep):

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(2, 7, 1), (3, 1, -2), (4, 5, 2, 3), (4, 7, 6), (5, 6, -1)]
    contord = [7, 2, 4, 1, 3, 6, 5]
    hl = nc(tensors, indices, contord)
    el = np.trace(hl @ C @ C.T.conj())

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(2, -1, 1), (3, 1, 7), (4, 5, 2, 3), (4, -2, 6), (5, 6, 7)]
    contord = [7, 3, 5, 1, 2, 6, 4]
    hr = nc(tensors, indices, contord)
    er = np.trace(C.T.conj() @ C @ hr)

    e = 0.5 * (el + er)

    hl -= el * np.eye(D)
    hr -= er * np.eye(D)

    hl = 0.5 * (hl + hl.T.conj())
    hr = 0.5 * (hr + hr.T.conj())

    Hl -= np.trace(Hl @ C @ C.T.conj()) * np.eye(D)
    Hr -= np.trace(C.T.conj() @ C @ Hr) * np.eye(D)

    def left_env(X):
        X = X.reshape(D, D)

        t = X@AL.transpose(1,0,2).reshape(D,d*D)
        XT = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,D)

        XR = np.trace(X @ C @ C.T.conj()) * np.eye(D)

        return (X - XT + XR).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        # tensors = [AR, AR.conj(), X]
        # indices = [(3, -1, 2), (3, -2, 1), (2, 1)]
        # contord = [2, 3, 1]
        # XT = nc(tensors, indices, contord)

        t = td(AR, X, axes=(2,0))
        t = td(t, AR.conj(), axes=([2,0], [2,0]))
        XT = t

        XL = np.trace(C.T.conj() @ C @ X) * np.eye(D)

        return (X - XT + XL).ravel()

    Ol = spspla.LinearOperator((D**2,D**2), matvec=left_env)
    Or = spspla.LinearOperator((D**2,D**2), matvec=right_env)

    Hl, _ = spspla.gmres(Ol, hl.ravel(), x0=Hl.ravel(), tol=ep/100) 
    Hr, _ = spspla.gmres(Or, hr.ravel(), x0=Hr.ravel(), tol=ep/100)

    Hl, Hr = Hl.reshape(D,D), Hr.reshape(D,D)

    Hl = 0.5 * (Hl + Hl.T.conj())
    Hr = 0.5 * (Hr + Hr.T.conj())

    # print('hl == hl+', spla.norm(hl - hl.T.conj()))
    # print('hr == hr+', spla.norm(hr - hr.T.conj()))

    # print('Hl == Hl+', spla.norm(Hl - Hl.T.conj()))
    # print('Hr == Hr+', spla.norm(Hr - Hr.T.conj()))

    # print('(L|hr)', np.trace(C.T.conj()@C@hr))
    # print('(hl|R)', np.trace(hl@C@C.T.conj()))

    # print('(L|Hr)', np.trace(C.T.conj()@C@Hr))
    # print('(Hl|R)', np.trace(Hl@C@C.T.conj()))

    return Hl, Hr, e

def Apply_HC(AL,AR,h,Hl,Hr,X):
    X = X.reshape(D, D)

    t = AL.reshape(d*D,D)@X@AR.transpose(1,0,2).reshape(D,d*D)
    t = h.reshape(d**2,d**2)@t.reshape(d,D,d,D).transpose(0,2,1,3).reshape(d**2,D*D)
    t = t.reshape(d,d,D,D).transpose(0,2,1,3).reshape(d*D,d*D)
    H1 = AL.conj().transpose(2,0,1).reshape(D,d*D)@t@AR.conj().transpose(0,2,1).reshape(d*D,D)

    H2 = Hl @ X
    H3 = X @ Hr

    return (H1 + H2 + H3).ravel()

def Apply_HAC(hL_mid,hR_mid,Hl,Hr,X):
    X = X.reshape(D, d, D)

    t = hL_mid.reshape(D*d,D*d)@X.reshape(D*d,D)
    H1 = t.reshape(D,d,D)

    t = X.reshape(D,d*D)@hR_mid.reshape(d*D,d*D).transpose(1,0)
    H2 = t.reshape(D,d,D)

    t = Hl@X.reshape(D,d*D)
    H3 = t.reshape(D,d,D)

    t = X.reshape(D*d,D)@Hr
    H4 = t.reshape(D,d,D)

    return (H1 + H2 + H3 + H4).ravel()

def calc_new_A(AL,AR,AC,C):

    Al = AL.reshape(d*D,D)
    Ar = AR.transpose(1,0,2).reshape(D,d*D)

    def calcnullspace(n):
        u,s,vh = spla.svd(n, full_matrices=True)

        right_null = vh.conj().T[:,D:]
        left_null = u.conj().T[D:,:]

        return left_null, right_null

    _, Al_right_null = calcnullspace(Al.T.conj())
    Ar_left_null, _  = calcnullspace(Ar.T.conj())

    Bl = Al_right_null.T.conj()@AC.transpose(1,0,2).reshape(d * D, D)
    Br = AC.reshape(D,d*D)@Ar_left_null.T.conj()

    epl = spla.norm(Bl)
    epr = spla.norm(Br)

    # print('new error left', epl)
    # print('new error right',epr)

    ulAC, plAC = spla.polar(AC.reshape(D * d, D), side='right')
    urAC, prAC = spla.polar(AC.reshape(D, d * D), side='left')

    ulC, plC = spla.polar(C, side='right')
    urC, prC = spla.polar(C, side='left')

    AL = (ulAC @ ulC.T.conj()).reshape(D, d, D).transpose(1, 0, 2)
    AR = (urC.T.conj() @ urAC).reshape(D, d, D).transpose(1, 0, 2)

    return epl, epr, AL, AR

def vumps(AL,AR,C,h,Hl,Hr,ep):
    AC = td(C, AR, axes=(1,1))

    Hl, Hr, e = HeffTerms(AL,AR,C,h,Hl,Hr,ep)

    tensors = [AL, h, AL.conj()]
    indices = [(2, 1, -3), (3, -2, 2, -4), (3, 1, -1)]
    contord = [1, 2, 3]
    hL_mid = nc(tensors, indices, contord)

    tensors = [AR, h, AR.conj()]
    indices = [(2, -4, 1), (-1, 3, -3, 2), (3, -2, 1)]
    contord = [1, 2, 3]
    hR_mid = nc(tensors, indices, contord)

    f = functools.partial(Apply_HC, AL, AR, h, Hl, Hr)
    g = functools.partial(Apply_HAC, hL_mid, hR_mid, Hl, Hr)

    H = spspla.LinearOperator((D*D,D*D), matvec=f)
    w, v = spspla.eigs(H, k=1, which='SR', v0=C.ravel(), tol=ep/100, return_eigenvectors=True)
    C = v[:,0].reshape(D,D)

    H = spspla.LinearOperator((D*d*D,D*d*D), matvec=g)
    w, v = spspla.eigs(H, k=1, which='SR', v0=AC.ravel(), tol=ep/100, return_eigenvectors=True)
    AC = v[:,0].reshape(D,d,D)

    epl, epr, AL, AR = calc_new_A(AL,AR,AC,C)

    x = calc_discard_weight(AL,AR,C,h,Hl,Hr)

    return AL, AR, C, Hl, Hr, e, epl, epr, x

def calc_entent(C):
    s = spla.svdvals(C)

    b = -np.log(s[0])

    entent = 0
    for i in range(np.size(s)):
        entent -= s[i]**2 * np.log(s[i]**2)

    return entent, b 

def calc_fidelity(X,Y):
    '''Presumes that MPS tensors X and Y are both properly normalized'''

    E = np.tensordot(X,Y.conj(),axes=(0,0)).transpose(0,2,1,3).reshape(D*D,D*D)

    evals = spspla.eigs(E, k=4, which='LM', return_eigenvectors=False)

    print(evals)
    
    return np.max(np.abs(evals))

def calc_stat_struc_fact(AL,AR,C,o1,o2,o3,N):
  
    q = np.linspace(0,np.pi,N)

    stat_struc_fact = []
    AC = td(AL, C, axes=(2,0))

    o1 = o1 - nc([AC, o1, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)
    o2 = o2 - nc([AC, o2, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)

    tensors = [AC, o1, o2, AC.conj()]
    indices =[(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc(tensors, indices, contord)
    print('s --> s1', s1)

    def left(X,o,Y):
        indices =[(2,1,-2), (3,2), (3,1,-1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices =[(2,-1,1), (3,2), (3,-2,1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC,o1,AL), right(AR,o2,AC)
    s3l, s3r = left(AL,o2,AC), right(AC,o1,AR)

    def left_env(X):
        X = X.reshape(D,D)

        tensors = [X, AR, AL.conj()]
        indices = [(1, 2), (3, 2, -2), (3, 1, -1)]
        contord = [2, 3, 1]
        XT = nc(tensors, indices, contord)

        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, AR.conj(), X]
        indices = [(3, -1, 2), (3, -2, 1), (2, 1)]
        contord = [2, 3, 1]
        XT = nc(tensors, indices, contord)

        return (X - np.exp(+1.0j * p) * XT).ravel()

    L1, R1 = np.random.rand(D, D) - .5, np.random.rand(D, D) - .5

    for i in range(N):
        p = q[i]

        left_env_op = spspla.LinearOperator((D*D, D*D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D*D, D*D), matvec=right_env)

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(), x0=L1.ravel(), tol=10**-12)
        R1, _ = spspla.gmres(right_env_op, s3r.ravel(), x0=R1.ravel(), tol=10**-12)

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * td(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * td(s3l, R1, axes=([1,0], [0,1]))

        s = s1+s2+s3

        stat_struc_fact.append(s.real)

    return q, np.array(stat_struc_fact)

def calc_momentum(AL,AR,C,o1,o2,o3,N):

    q = np.linspace(0,np.pi,N)

    momentum = []
    AC = td(AL, C, axes=(2,0))

    tensors = [AC, o1, o2, AC.conj()]
    indices =[(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc(tensors, indices, contord)
    print('n --> s1', s1)

    def left(X,o,Y):
        indices =[(2,1,-2), (3,2), (3,1,-1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices =[(2,-1,1), (3,2), (3,-2,1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC,o1,AL), right(AR,o2,AC)
    s3l, s3r = left(AL,o2,AC), right(AC,o1,AR)

    def left_env(X):
        X = X.reshape(D,D)

        tensors = [X, AR, o3, AL.conj()]
        indices = [(1, 2), (3, 2, -2), (4,3) , (4, 1, -1)]
        contord = [2, 3, 4, 1]
        XT = nc(tensors, indices, contord)

        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D,D)

        tensors = [AL, o3, AR.conj(), X]
        indices = [(3, -1, 2), (4,3), (4, -2, 1), (2, 1)]
        contord = [2, 3, 4, 1]
        XT = nc(tensors, indices, contord)

        return (X - np.exp(+1.0j * p) * XT).ravel()

    L1, R1 = np.random.rand(D, D) - .5, np.random.rand(D, D) - .5

    for i in range(N):
        p = q[i]

        left_env_op = spspla.LinearOperator((D*D, D*D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D*D, D*D), matvec=right_env)

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(), x0=L1.ravel(), tol=10**-12)
        R1, _ = spspla.gmres(right_env_op, s3r.ravel(), x0=R1.ravel(), tol=10**-12)

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * td(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * td(s3l, R1, axes=([1,0], [0,1]))

        s = s1+s2+s3
        
        momentum.append(s.real)

    return q, np.array(momentum)

energy = []
error = []
discard_weight = []

count, tol, ep = 0, 1e-12, 1e-2

d = 2
#D = 80 + int(sys.argv[1]) * 10
D = 15
N = 500

si, sx = np.array([[1, 0],[0, 1]]),    np.array([[0, 1],[1, 0]])
sy, sz = np.array([[0, -1j],[1j, 0]]), np.array([[1, 0],[0, -1]])
sp, sm, n = 0.5*(sx + 1.0j*sy), 0.5*(sx - 1.0j*sy), 0.5*(sz + np.eye(d))

x = 1
y = 1
z = 0

t  = 1
V  = 0
V2 = 0
##################################################

XYZ = -(x*np.kron(sx, sx) + y*np.kron(sy, sy) - z*np.kron(sz, sz)) #+ 0.5*(np.kron(sz, si) + np.kron(si, sz))

XY  = -x*np.kron(sx, sx) - y*np.kron(sy, sy)

TFI = -np.kron(sx, sx) - (1/2) * (np.kron(sz, si) + np.kron(si, sz))

tVV2 = -2 * t * (np.kron(sx, sx) + np.kron(sy, sy)) + V * np.kron(sz, sz)

####################################################

h = XYZ
h = h.reshape(d,d,d,d)

A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
C = np.random.rand(D, D) - 0.5
Hl, Hr = np.eye(D, dtype=A.dtype), np.eye(D, dtype=A.dtype)

AL, C = left_ortho(A, C, tol/100)
AR, C = right_ortho(AL, C, tol/100)

print('left iso', spla.norm(nc([AL, AL.conj()], [[3,1,-2], [3,1,-1]]) - np.eye(D)))
print('right iso', spla.norm(nc([AR, AR.conj()], [[3,-1,1], [3,-2,1]]) - np.eye(D)))
print('norm', nc([AL, AL.conj(), C, C.conj(), AR, AR.conj()], [[7,1,2],[7,1,3],[2,4],[3,5],[8,4,6],[8,5,6]]))
print('ALC - CAR', spla.norm(nc([AL,C],[[-1,-2,1],[1,-3]]) - nc([C,AR],[[-2,1], [-1,1,-3]])))

AL, AR, C, Hl, Hr, *_ = vumps(AL,AR,C,h,Hl,Hr,ep)

AL, C = left_ortho(AR, C, tol/100)
AR, C = right_ortho(AL, C, tol/100)

while ep > tol and count < 1000:
    print(count, end='\t')

    AL, AR, C, Hl, Hr, e, epl, epr, x = vumps(AL,AR,C,h,Hl,Hr,ep)

    ep = np.maximum(epl,epr)

    print('ep ', ep, end='\t')
    print('energy', e)

    energy.append(e)
    error.append(ep)
    discard_weight.append(x)
    
    count += 1

q, momentum = calc_momentum(AL, AR, C, sp, sm, -sz, N)

q, stat_struc_fact = calc_stat_struc_fact(AL,AR,C,n,n,None,N)

plt.plot(np.array(energy).real)
plt.show()

plt.plot(np.array(error))
plt.show()

plt.plot(np.array(discard_weight))
plt.show()

plt.plot(q, momentum)
plt.show()

plt.plot(q, stat_struc_fact)
plt.show()


# model = 'tVV2'

# path = '/home/baktay.j/vumps/data'

# filename = "energy_%s_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), energy)

# filename = "error_%s_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), error)

# filename = "discweight_%s_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), discard_weight)

# filename = "statstrucfact_%s_%.2f_%.2f_%i_.txt" % (model , V, V2, D)
# np.savetxt(os.path.join(path, filename), stat_struc_fact)

# filename = "momentum_%s_%.2f_%.2f_%i_.txt" % (model , V, V2, D)
# np.savetxt(os.path.join(path, filename), momentum)

# filename1 = "%s_AL_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# filename2 = "%s_AR_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# filename3 = "%s_C_%.2f_%.2f_%i_.txt" % (model, V, V2, D)

# open(os.path.join(path, filename1), 'a')
# for data_slice in AL:
#     np.savetxt(os.path.join(path, filename1), data_slice)

# open(os.path.join(path, filename2), 'a')
# for data_slice in AR:
#     np.savetxt(os.path.join(path, filename2), data_slice)

# np.savetxt(os.path.join(path, filename3), C)








