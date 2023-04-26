import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla

def left_gauge(A, X0, tol, stol):
    d, D = A.shape[0], A.shape[1]

    def left_fixed_point(A, B):
        def left_transfer_op(X):
            tensors = [A, X.reshape(D, D), B.conj()]
            indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
            contord = [2, 3, 1]
            return nc.ncon(tensors,indices,contord).ravel()

        E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
        evals, evecs = spspla.eigs(E, k=1, which="LR", v0=X0, tol=tol)
        return evals[0], evecs[:,0].reshape(D, D)

    eval_LR, l = left_fixed_point(A, A)

    l = l + l.T.conj()
    l /= np.trace(l)

    A = A / np.sqrt(eval_LR)

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

def right_gauge(A, X0, tol, stol):
    A, L = left_gauge(np.transpose(A, (0, 2, 1)), X0, tol, stol)
    A, L = np.transpose(A, (0, 2, 1)), np.transpose(L, (1, 0))
    return A, L

def mix_gauge(A, X0, tol, stol):
	AL, C = left_gauge(A, X0, tol / 100, stol)
	AR, C = right_gauge(AL, C, tol / 100, stol)
	return AL, AR, C

def checks(AL, AR, C):
    print('left iso', 
    	spla.norm(nc.ncon([AL, AL.conj()], [[3,1,-2], [3,1,-1]]) 
    		        - np.eye(C.shape[0]))
    	)

    print('right iso', 
    	spla.norm(nc.ncon([AR, AR.conj()], [[3,-1,1], [3,-2,1]]) 
    		        - np.eye(C.shape[0])
                    )
        )
    
    print('norm', 
    	nc.ncon(
            [AL, AL.conj(), C, C.conj(), AR, AR.conj()],
            [[7, 1, 2], [7, 1, 3], [2, 4], [3, 5], [8, 4, 6], [8, 5, 6]]
            )
        )
    
    print('ALC - CAR', 
    	spla.norm(nc.ncon([AL, C],[[-1, -2, 1],[1, -3]]) 
    		    - nc.ncon([C, AR],[[-2, 1], [-1, 1, -3]]))
    	)

def HeffTerms_two(AL, AR, C, Hl, Hr, h, ep):
    d, D = AL.shape[1], C.shape[0]

    AL = AL.transpose(1, 0, 2)
    AR = AR.transpose(1, 0, 2)

    h = h.reshape(d, d, d, d)

    tensors = [AL, AL, h, AL.conj(), AL.conj()]
    indices = [(2, 7, 1), (3, 1, -2), (4, 5, 2, 3), (4, 7, 6), (5, 6, -1)]
    contord = [7, 2, 4, 1, 3, 6, 5]
    hl = nc.ncon(tensors, indices, contord)
    el = np.trace(hl @ C @ C.T.conj())
    print('el', el)

    tensors = [AR, AR, h, AR.conj(), AR.conj()]
    indices = [(2, -1, 1), (3, 1, 7), (4, 5, 2, 3), (4, -2, 6), (5, 6, 7)]
    contord = [7, 3, 5, 1, 2, 6, 4]
    hr = nc.ncon(tensors, indices, contord)
    er = np.trace(C.T.conj() @ C @ hr)
    print('er', er)

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

def HeffTerms_three(AL, AR, C, Hl, Hr, h, ep):
    d, D = AL.shape[1], C.shape[0]

    AL = AL.transpose(1, 0, 2)
    AR = AR.transpose(1, 0, 2)

    h = h.reshape(d, d, d, d, d, d)

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

def calc_expectations(AL, AR, C, O):
    AC = np.tensordot(AL, C, axes=(2, 0))

    if O.shape[0] == d:
        tensors = [AC, O, AC.conj()]
        indices = [(1, 3, 4), (2, 3), (1, 2, 4)]
        contord = [1, 4, 3, 2]
        expectation_value = nc.ncon(tensors, indices, contord)

    if O.shape[0] == d**2:
        pass
    return expectation_value

def fixed_points(A, B):
    D = A.shape[2]

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





