import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla

def left_gauge(A, X0, tol, stol):
    D = X0.shape[0]

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

def gauge_checks(AL, AR, C):
    AL, AR = AL.transpose(1, 0, 2), AR.transpose(1, 0, 2)

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

def mixed_canon_mps(d, D, tol, stol):
    A = (np.random.rand(d, D, D) - 0.5) + 1j * (np.random.rand(d, D, D) - 0.5)
    C = np.random.rand(D, D) - 0.5

    AL, AR, C = mix_gauge(A, C, tol, stol)
    return AL.transpose(1, 0, 2), AR.transpose(1, 0, 2), C
