#cd Desktop/code/vumps 

import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt
import functools

from numpy import tensordot as td
from ncon import ncon as nc 

def calc_discard_weight(AL,AR,C,h,Hl,Hr):

    def eff_ham(X):
        X = X.reshape(d,D,d,D)

        tensors = [AL, AL, X, h, AL.conj(), AL.conj()]
        indices = [(7,1,2), (8,2,4), (9,4,-3,-4), (5,6,-1,7,8,9), (5,1,3), (6,3,-2)]
        contord = [1,2,3,4,5,6,7,8,9]
        H1 = nc(tensors,indices,contord)

        t = AL.transpose(1,0,2).reshape(D*d,D)@X.transpose(1,0,2,3).reshape(D,d*d*D)
        t = AL.transpose(1,0,2).reshape(D*d,D)@t.reshape(D,d*d*d*D)
        t = h.reshape(d**3,d**3)@t.reshape(D,d**3,d*D).transpose(1,0,2).reshape(d**3,D*d*D)
        t = AL.conj().transpose(2,0,1).reshape(D,d*D)@t.reshape(d,d**2,D,d*D).transpose(0,2,1,3).reshape(d*D,d*d*d*D)
        t = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,d*d*D)
        t = t.reshape(D,d,d,D).transpose(1,0,2,3)

        print(spla.norm(t - H1))

        tensors = [AL, X, h, AL.conj()]
        indices = [(4,1,2), (5,2,6,-4), (3,-1,-3,4,5,6), (3,1,-2)]
        contord = [1,2,3,4,5,6]
        H2 = nc(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(4,-2,5,2), (6,2,1), (-1,-3,3,4,5,6), (3,-4,1)]
        contord = [1,2,3,4,5,6]
        H3 = nc(tensors,indices,contord)

        tensors = [X, AR, AR, h, AR.conj(), AR.conj()]
        indices = [(-1,-2,7,4), (8,4,2), (9,2,1), (-3,5,6,7,8,9), (5,-4,3), (6,3,1)]
        contord = [1,2,3,4,5,6,7,8,9]
        H4 = nc(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2,1),(-1,1,-3,-4)]
        H5 = nc(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1,-2,-3,1),(1,-4)]
        H6 = nc(tensors,indices)

        return (H1+H2+H3+H4+H5).ravel()

    two_site = nc([AL, C, AR], [(-1,-2,1),(1,2),(-3,2,-4)])

    H = spspla.LinearOperator((d*D*d*D,d*D*d*D), matvec=eff_ham)
    w, v = spspla.eigs(H, k=1, which='SR', v0=two_site.ravel(), tol=1e-12, return_eigenvectors=True)

    u, s, vh = spla.svd(v[:,0].reshape(d*D,d*D))

    t = 0
    for i in range(D,d*D):
        t += s[i]**2
        
    return t

def left_ortho(A,X0,D,tol):

    def left_fixed_point(A,B):
        def left_transfer_op(X):

            t = X.reshape(D,D)@A.transpose(1,0,2).reshape(D,d*D)
            t = B.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,D)
            t = t.ravel()

            # tensors = [A, X.reshape(D, D), B.conj()]
            # indices = [(1,2,-2), (3, 2), (1, 3, -1)]
            # contord = [2, 3, 1]
            # nc(tensors,indices,contord).ravel()

            return t

        E = spspla.LinearOperator((D*D,D*D), matvec=left_transfer_op)

        evals, evecs = spspla.eigs(E, k=1, which="LR", v0=X0, tol=tol)

        return evals[0], evecs[:,0].reshape(D,D)

    norm, l = left_fixed_point(A,A)

    A = A/np.sqrt(norm)

    l = 0.5*(l + l.T.conj())

    l = l/np.trace(l)

    w, v = spla.eigh(l)
    l = v@np.diag(np.abs(w))@v.T.conj()

    L = spla.cholesky(l, lower=False)

    Li = spla.inv(L)

    AL = nc([L, A, Li], [(-2,1), (-1,1,2), (2,-3)])

    return AL, L

def right_ortho(A,X0,D,tol):

    def right_fixed_point(A,B):
        def right_transfer_op(X):
            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1, -1, 2), (2, 3), (1, -2, 3)]
            contord = [2, 3, 1]
            return nc(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D*D,D*D), matvec=right_transfer_op)

        evals, evecs = spspla.eigs(E, k=1, which="LR", v0=X0, tol=tol)

        return evals[0], evecs[:,0].reshape(D,D)

    norm, r = right_fixed_point(A,A)

    A = A/np.sqrt(norm)

    r = 0.5*(r + r.T.conj())

    r = r/np.trace(r)

    w, v = spla.eigh(r)
    r = v@np.diag(np.abs(w))@v.T.conj()

    R = spla.cholesky(r, lower=True)
    Ri = spla.inv(R)

    AR = nc([Ri, A, R], [(-2,1), (-1,1,2), (2,-3)])

    return AR, R

def HeffTerms(AL,AR,C,h,Hl,Hr,ep):

    tensors = [AL, AL, AL, h, AL.conj(), AL.conj(), AL.conj()]
    indices = [(4,7,8), (5,8,10), (6,10,-2), (1,2,3,4,5,6), (1,7,9), (2,9,11), (3,11,-1)]
    contord = [7,8,9,10,11,1,2,3,4,5,6]
    hl = nc(tensors,indices,contord)
    el = np.trace(hl @ C @ C.T.conj())

    tensors = [AR, AR, AR, h, AR.conj(), AR.conj(), AR.conj()]
    indices = [(4,-1,10), (5,10,8), (6,8,7), (1,2,3,4,5,6), (1,-2,11), (2,11,9), (3,9,7)]
    contord = [7,8,9,10,11,1,2,3,4,5,6]
    hr = nc(tensors,indices,contord)
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

        XT = td(X, AL, axes=(1,1))
        XT = td(XT, AL.conj(), axes=([0,1], [1,0])).transpose(1,0)

        XR = np.trace(X @ C @ C.T.conj()) * np.eye(D)

        return (X - XT + XR).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        XT = td(AR, X, axes=(2,0))
        XT = td(XT, AR.conj(), axes=([2,0], [2,0]))

        XL = np.trace(C.T.conj() @ C @ X) * np.eye(D)

        return (X - XT + XL).ravel()

    Ol = spspla.LinearOperator((D**2,D**2), matvec=left_env)
    Or = spspla.LinearOperator((D**2,D**2), matvec=right_env)

    Hl, _ = spspla.gmres(Ol, hl.ravel(), x0=Hl.ravel(), tol=ep/100) 

    Hr, _ = spspla.gmres(Or, hr.ravel(), x0=Hr.ravel(), tol=ep/100)

    Hl, Hr = Hl.reshape(D,D), Hr.reshape(D,D)

    Hl = 0.5 * (Hl + Hl.T.conj())
    Hr = 0.5 * (Hr + Hr.T.conj())

    return Hl, Hr, e

def Apply_HC(hl_mid,hr_mid,AL,AR,h,Hl,Hr,X):
    X = X.reshape(D, D)

    t = hl_mid.transpose(0,1,3,2).reshape(D*d*d,D)@X
    t = t.reshape(D*d,d*D)@AR.reshape(d*D,D)
    t = t.reshape(D,d*D)@AR.conj().transpose(0,2,1).reshape(d*D,D)
    H1 = t

    t = X@hr_mid.transpose(2,0,1,3).reshape(D,d*D*d)
    t = AL.transpose(1,0,2).reshape(D,d*D)@t.reshape(D,d,D,d).transpose(3,0,1,2).reshape(d*D,d*D)
    t = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,D)
    H2 = t

    H3 = Hl @ X
    H4 = X @ Hr
    return (H1 + H2 + H3 + H4).ravel()

def Apply_HAC(hl_mid, hr_mid, AL, AR, h, Hl, Hr ,X):
    X = X.reshape(D, d, D)

    t = hl_mid.reshape(D*d,D*d)@X.reshape(D*d,D)
    t = t.reshape(D,d,D)
    H1 = t

    t = AL.reshape(d*D,D)@X.reshape(D,d*D)
    t = t.reshape(d*D*d,D)@AR.transpose(1,0,2).reshape(D,d*D)
    t = t.reshape(d,D,d,d,D).transpose(1,4,0,2,3).reshape(D*D,d*d*d)@h.reshape(d*d*d,d*d*d).transpose(1,0)
    t = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D,D,d,d,d).transpose(0,2,3,1,4).reshape(D*d,d*D*d)
    t = t.reshape(D*d,D*d)@AR.conj().transpose(2,0,1).reshape(D*d,D)
    t = t.reshape(D,d,D)
    H2 = t

    t = X.reshape(D,d*D)@hr_mid.transpose(3,2,0,1).reshape(d*D,d*D)
    t = t.reshape(D,d,D)
    H3 = t

    t = Hl@X.reshape(D,d*D)
    t = t.reshape(D,d,D)
    H4 = t

    t = X.reshape(D*d,D)@Hr
    t = t.reshape(D,d,D)
    H5 = t

    return (H1 + H2 + H3 + H4 + H5).ravel()

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

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(4,7,8), (5,8,-3), (1,2,-2,4,5,-4), (1,7,9), (2,9,-1)]
    contord = [7,8,9,1,2,4,5]
    hl_mid = nc(tensors,indices,contord)

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(5,-3,8), (6,8,7), (-1,2,3,-4,5,6), (2,-2,9), (3,9,7)]
    contord = [7,8,9,2,3,5,6]
    hr_mid = nc(tensors,indices,contord)

    f = functools.partial(Apply_HC, hl_mid, hr_mid, AL, AR, h, Hl, Hr)
    g = functools.partial(Apply_HAC, hl_mid, hr_mid, AL, AR, h, Hl, Hr)

    H = spspla.LinearOperator((D*D,D*D), matvec=f)
    w, v = spspla.eigs(H, k=1, which='SR', v0=C.ravel(), tol=ep/100, return_eigenvectors=True)
    C = v[:,0].reshape(D,D)

    H = spspla.LinearOperator((D*d*D,D*d*D), matvec=g)
    w, v = spspla.eigs(H, k=1, which='SR', v0=AC.ravel(), tol=ep/100, return_eigenvectors=True)
    AC = v[:,0].reshape(D,d,D)

    epl, epr, AL, AR = calc_new_A(AL,AR,AC,C)

    x = calc_discard_weight(AL,AR,C,h,Hl,Hr)

    return AL, AR, C, Hl, Hr, e, epl, epr, x

def calc_stat_struc_fact(AL,AR,C,o1,o2,o3):
  
    N = 500
    q = np.linspace(0,np.pi,N)

    stat_struc_fact = []
    AC = td(AL, C, axes=(2,0))

    o1 = o1 - nc([AC, o1, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)
    o2 = o2 - nc([AC, o2, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)

    tensors = [AC, o1, o2, AC.conj()]
    indices =[(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc(tensors, indices, contord)

    def left(X,o,Y):
        indices =[(2,1,-2), (3,2), (3,1,-1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices =[(2,-1,1), (3,2), (3,-2,1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC,o1,AL), right(AR,o2,AC)
    s3l, s3r = left(AL,o2,AC), right(AC,o1,AR)

    def left_fixed_point(A,B):
        def left_transfer_op(X):

            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1,2,-2), (3, 2), (1, 3, -1)]
            contord = [2, 3, 1]
            return nc(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D*D,D*D), matvec=left_transfer_op)
        evals, evecs = spspla.eigs(E,k=1,which="LR", tol=10**-12)
        return evecs[:,0].reshape(D,D)

    def right_fixed_point(A,B):
        def right_transfer_op(X):
            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1, -1, 2), (2, 3), (1, -2, 3)]
            contord = [2, 3, 1]
            return nc(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D*D,D*D), matvec=right_transfer_op)
        evals, evecs = spspla.eigs(E,k=1,which="LR", tol=10**-12)
        return evecs[:,0].reshape(D,D)

    l_Erl, r_Erl = left_fixed_point(AR,AL), right_fixed_point(AR,AL)

    l_Erl, r_Erl = l_Erl/np.sqrt(np.trace(l_Erl@r_Erl)), r_Erl/np.sqrt(np.trace(l_Erl@r_Erl))

    l_Elr, r_Elr = left_fixed_point(AL,AR), right_fixed_point(AL,AR)

    l_Elr, r_Elr = l_Elr/np.sqrt(np.trace(l_Elr@r_Elr)), r_Elr/np.sqrt(np.trace(l_Elr@r_Elr))

    def left_env(X):
        X = X.reshape(D,D)

        tensors = [X, AR, AL.conj()]
        indices = [(1, 2), (3, 2, -2), (3, 1, -1)]
        contord = [2, 3, 1]

        XT = nc(tensors, indices, contord)
        XR = np.trace(X @ r_Erl)*l_Erl
        return (X - np.exp(-1.0j*p)*(XT-XR)).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, AR.conj(), X]
        indices = [(3, -1, 2), (3, -2, 1), (2, 1)]
        contord = [2, 3, 1]

        XT = nc(tensors, indices, contord)
        XL = np.trace(l_Elr @ X) * r_Elr
        return (X - np.exp(+1.0j*p)*(XT-XL)).ravel()

    for i in range(N):
        p = q[i]

        left_env_op = spspla.LinearOperator((D*D, D*D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D*D, D*D), matvec=right_env)

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(), tol=10**-12)
        R1, _ = spspla.gmres(right_env_op, s3r.ravel(), tol=10**-12)

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * td(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * td(s3l, R1, axes=([1,0], [0,1]))

        s = s1+s2+s3

        stat_struc_fact.append(s.real)

    return q, np.array(stat_struc_fact)

def calc_momentum(AL,AR,C,o1,o2,o3):
  
    N = 500
    q = np.linspace(0,np.pi,N)

    momentum = []
    AC = td(AL, C, axes=(2,0))

    o1 = o1 - nc([AC, o1, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)
    o2 = o2 - nc([AC, o2, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)
    o3 = o3 - nc([AC, o3, AC.conj()], [[3,1,4], [2,3], [2,1,4]])*np.eye(d)

    tensors = [AC, o1, o2, AC.conj()]
    indices =[(3,1,2), (4,3), (5,4), (5,1,2)]
    contord = [1,2,3,4,5]
    s1 = nc(tensors, indices, contord)

    def left(X,o,Y):
        indices =[(2,1,-2), (3,2), (3,1,-1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    def right(X,o,Y):
        indices =[(2,-1,1), (3,2), (3,-2,1)]
        return nc([X, o, Y.conj()], indices, [1,2,3])

    s2l, s2r = left(AC,o1,AL), right(AR,o2,AC)
    s3l, s3r = left(AL,o2,AC), right(AC,o1,AR)

    def left_fixed_point(A,B):
        def left_transfer_op(X):

            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1,2,-2), (3, 2), (1, 3, -1)]
            contord = [2, 3, 1]
            return nc(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D*D,D*D), matvec=left_transfer_op)
        evals, evecs = spspla.eigs(E,k=1,which="LR", tol=10**-12)
        return evecs[:,0].reshape(D,D)

    def right_fixed_point(A,B):
        def right_transfer_op(X):
            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1, -1, 2), (2, 3), (1, -2, 3)]
            contord = [2, 3, 1]
            return nc(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D*D,D*D), matvec=right_transfer_op)
        evals, evecs = spspla.eigs(E,k=1,which="LR", tol=10**-12)
        return evecs[:,0].reshape(D,D)

    l_Erl, r_Erl = left_fixed_point(AR,AL), right_fixed_point(AR,AL)

    l_Erl, r_Erl = l_Erl/np.sqrt(np.trace(l_Erl@r_Erl)), r_Erl/np.sqrt(np.trace(l_Erl@r_Erl))

    l_Elr, r_Elr = left_fixed_point(AL,AR), right_fixed_point(AL,AR)

    l_Elr, r_Elr = l_Elr/np.sqrt(np.trace(l_Elr@r_Elr)), r_Elr/np.sqrt(np.trace(l_Elr@r_Elr))

    # print(np.trace(l_Erl@r_Erl))
    # print(spla.norm(np.subtract(l_Erl, nc([l_Erl, AR, AL.conj()], [[2,1], [3,1,-2], [3,2,-1]]))))
    # print(spla.norm(np.subtract(r_Erl, nc([r_Erl, AR, AL.conj()], [[1,2], [3,-1,1], [3,-2,2]]))))

    # print('norm', np.trace(l_Elr@r_Elr))
    # print(spla.norm(np.subtract(l_Elr, nc([l_Elr, AL, AR.conj()], [[2,1], [3,1,-2], [3,2,-1]]))))
    # print(spla.norm(np.subtract(r_Elr, nc([r_Elr, AL, AR.conj()], [[1,2], [3,-1,1], [3,-2,2]]))))

    def left_env(X):
        X = X.reshape(D,D)

        tensors = [X, AR, o3, AL.conj()]
        indices = [(1, 2), (3, 2, -2), (4,3) , (4, 1, -1)]
        contord = [2, 3, 4, 1]

        XT = nc(tensors, indices, contord)
        XR = np.trace(X @ r_Erl)*l_Erl
        return (X - np.exp(-1.0j*p)*(XT-XR)).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, o3, AR.conj(), X]
        indices = [(3, -1, 2), (4,3), (4, -2, 1), (2, 1)]
        contord = [2, 3, 4, 1]

        XT = nc(tensors, indices, contord)
        XL = np.trace(l_Elr @ X) * r_Elr
        return (X - np.exp(+1.0j*p)*(XT-XL)).ravel()

    for i in range(N):
        p = q[i]; print(i)

        left_env_op = spspla.LinearOperator((D*D, D*D), matvec=left_env)
        right_env_op = spspla.LinearOperator((D*D, D*D), matvec=right_env)

        L1, _ = spspla.gmres(left_env_op, s2l.ravel(), tol=10**-12)
        R1, _ = spspla.gmres(right_env_op, s3r.ravel(), tol=10**-12)

        L1, R1 = L1.reshape(D,D), R1.reshape(D,D)

        s2 = np.exp(-1.0j*p) * td(L1, s2r, axes=([1,0], [0,1]))
        s3 = np.exp(+1.0j*p) * td(s3l, R1, axes=([1,0], [0,1]))

        s = s1+s2+s3
        
        momentum.append(s.real)

    return q, np.array(momentum)

def calc_entent(C):
    u, s, vh = spla.svd(C)

    entent = 0
    for i in range(np.size(s)):
        entent -= s[i]**2 * np.log(s[i]**2)

    return entent 

def pad_with(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = (np.random.rand() - 0.5) + 1j * (np.random.rand() - 0.5)
    vector[-pad_width[1]:] = (np.random.rand() - 0.5) + 1j * (np.random.rand() - 0.5)

def expand(A,deltaD):
    A_new = []
    for i in range(A.shape[0]):
        A_new.append(np.pad(A[i,:,:], ((0,deltaD), (0,deltaD)), pad_with))
    return np.array(A_new)

#######################################################################################

energy, error, discard_weight, entent = [], [], [], []

tol = 1e-12
ep  = 1e-2

d = 2
#D = 80 + int(sys.argv[1]) * 10
D = 15
deltaD = 2
count = 0

si, sx = np.array([[1, 0],[0, 1]]),    np.array([[0, 1],[1, 0]])
sy, sz = np.array([[0, -1j],[1j, 0]]), np.array([[1, 0],[0, -1]])
sp, sm, n = 0.5*(sx + 1.0j*sy), 0.5*(sx - 1.0j*sy), 0.5*(sz + np.eye(d))

x = 1
y = 1
z = 0

J = 1
g = 1

t = 2
V = 0
mu = 0

t2 = 0
tp = 0
V2 = 0

#######################################################################################

XYZ = x * np.kron(np.kron(sx, sx), si) + y * np.kron(np.kron(sy, sy), si) + z * np.kron(np.kron(sz, sz), si)

TFI = -(J*np.kron(si, np.kron(sx, sx)) + g*np.kron(si, np.kron(sz, si)))

nnn_ham = x * (np.kron(np.kron(sx, sz), sx) + np.kron(np.kron(sy, sz), sy)) - y * (np.kron(np.kron(sz, si), si) + np.eye(d**3))

tVV2 = -t*(np.kron(np.kron(sx, sx), si) + np.kron(np.kron(sy, sy), si)) + V*np.kron(np.kron(sz, sz), si)+ V2*np.kron(np.kron(sz, si), sz)

quasi_fermi = -t * (np.kron(np.kron(sx, sx), si) + np.kron(np.kron(sy, sy), si))
+ t2 * (np.kron(np.kron(sx, sz), sx) + np.kron(np.kron(sy, sz), sy)) 
+ V * np.kron(np.kron(sz, sz), si)
+ tp * (np.kron(sz, np.kron(sx, sx)) + np.kron(sz, np.kron(sy, sy)))

#########################################################################################

h = tVV2
h = h.reshape(d,d,d,d,d,d)

A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
C = np.random.rand(D, D) - 0.5 
Hl, Hr = np.eye(D, dtype=A.dtype), np.eye(D, dtype=A.dtype)

AL, C = left_ortho(A,C,D, tol/100)
AR, C = right_ortho(AL,C,D, tol/100)

AL, AR, C, Hl, Hr, *_ = vumps(AL,AR,C,h,Hl,Hr,ep)

AL, C = left_ortho(AR,C,D, tol/100)
AR, C = right_ortho(AL,C,D, tol/100)

while ep > tol and count < 400:

    print(count)

    AL, AR, C, Hl, Hr, e, epl, epr, x = vumps(AL,AR,C,h,Hl,Hr,ep)

    if np.maximum(epl, epr) < ep:
        ep = np.maximum(epl,epr)

    # print(ep)

    energy.append(e)
    error.append(ep)
    discard_weight.append(x)

    AL, C = left_ortho(AR,C,D, tol/100)

    count += 1

# q, stat_struc_fact = calc_stat_struc_fact(AL, AR, C, n, n, None)

# q, momentum = calc_momentum(AL, AR, C, sp, sm, -sz)

energy = np.array(energy)
plt.plot(energy.real)
plt.title('Energy: ' + 'D = ' + str(D) + ', d = ' + str(d) + ', z = ' + str(z))
plt.show()

discard_weight = np.array(discard_weight)
plt.plot(discard_weight)
plt.title('Energy: ' + 'D = ' + str(D) + ', d = ' + str(d) + ', z = ' + str(z))
plt.show()

# plt.plot(q, momentum)
# plt.show()

# model = 'tVV2'

# path = '/home/baktay.j/vumps/data'

# filename = "energy_%s_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), energy)

# filename = "error_%s_%.2f_%.2f_%i_.txt" % (model, V, V2, D)
# np.savetxt(os.path.join(path, filename), error)

# filename = "statstrucfact_%s_%.2f_%.2f_%i_.txt" % (model , V, V2, D)
# np.savetxt(os.path.join(path, filename), stat_struc_fact)

# filename = "momentum_%s_%.2f_%.2f_%i_.txt" % (model , V, V2, D)
# np.savetxt(os.path.join(path, filename), mom)

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




