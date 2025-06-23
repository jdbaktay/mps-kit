import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla

def nonuniform_mesh(npts_left, npts_mid, npts_right, k0, dk):
    k_left, k_right = k0 - dk, k0 + dk
    mesh = np.concatenate((
            np.linspace(0.0,     k_left,  npts_left, endpoint=False),
            np.linspace(k_left,  k_right, npts_mid,  endpoint=False),
            np.linspace(k_right, 1.0,     npts_right)
            ))
    return mesh

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

def calc_corr_length(A, X0, tol):
    D = X0.shape[0]

    def left_transfer_op(X):
        tensors = [A, X.reshape(D, D), A.conj()]
        indices = [(1, 2, -2), (3, 2), (1, 3, -1)]
        contord = [2, 3, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)

    # k must be LARGER THAN OR EQUAL TO 2
    evals = spspla.eigs(E, k=4, which="LM", v0=X0, tol=tol,
                                return_eigenvectors=False)

    print('argmax', np.argmax(evals), evals)
    return -1.0 / np.log(np.abs(evals[-2])), evals

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
