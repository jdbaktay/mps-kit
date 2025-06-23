import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import functools

def calc_expectation_val(o, AC, lfp):
    tensors = [lfp, AC, o, AC.conj()]
    indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
    return nc.ncon(tensors, indices)

def op_transfer_matrix(A, B, sz):
    D = A.shape[0]

    def left_transfer_op(X):
        tensors = [X.reshape(D, D), A, -sz, B.conj()]
        indices = [(4, 5), (5, 2, -2), (1, 2), (4, 1, -1)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    def right_transfer_op(X):
        tensors = [A, -sz, B.conj(), X.reshape(D, D)]
        indices = [(-1, 2, 4), (1, 2), (-2, 1, 5), (4, 5)]
        contord = [4, 5, 2, 1]
        return nc.ncon(tensors,indices,contord).ravel()

    E = spspla.LinearOperator((D * D, D * D), matvec=left_transfer_op)
    wl, lfp_AB = spspla.eigs(E, k=2, which='LM', tol=1e-14)
    print('wl', wl, np.abs(wl))
    print('phi', np.angle(wl))

    if np.angle(wl[0]) > 0:
        lfp_AB = lfp_AB[:,0].reshape(D, D)
        print(np.angle(wl[0]))
        phi = np.angle(wl[0])
    if np.angle(wl[0]) < 0:
        lfp_AB = lfp_AB[:,1].reshape(D, D)
        print(np.angle(wl[1]))
        phi = np.angle(wl[1])

    E = spspla.LinearOperator((D * D, D * D), matvec=right_transfer_op)
    wr, rfp_AB = spspla.eigs(E, k=2, which='LM', tol=1e-14)
    print('wr', wr, np.abs(wr))
    print('phi', np.angle(wr))

    if np.angle(wr[0]) > 0:
        rfp_AB = rfp_AB[:,0].reshape(D, D)
        print(np.angle(wr[0]))
    if np.angle(wr[0]) < 0:
        rfp_AB = rfp_AB[:,1].reshape(D, D)
        print(np.angle(wr[1]))

    norm = np.trace(lfp_AB @ rfp_AB)
    print('(l|r)', norm)

    # Normalize fixed points
    lfp_AB /= np.sqrt(norm)
    rfp_AB /= np.sqrt(norm)

    print('(l|r)', np.trace(lfp_AB @ rfp_AB))
    return lfp_AB, np.exp(1j * phi)

def spectral_fxn(AL, AR, C,
                 excit_energy, excit_states, mom_vec,
                 sp, sm, sz):

    D = AL.shape[0]
    d = AL.shape[1]

    AC = np.tensordot(AL, C, axes=(2, 0))

    VL = spla.null_space(AL.conj().reshape(D * d, D).T)
    VL = VL.reshape(D, d, (d - 1) * D)

    lz, phase = op_transfer_matrix(AL, AL, sz)

    def left_env(X):
        X = X.reshape(D, D)

        tensors = [X, AR, -sz, (phase * AR).conj()]
        indices = [(3, 4), (4, 2, -2), (1, 2), (3, 1, -1)]
        contord = [3, 4, 1, 2]
        XT = nc.ncon(tensors, indices, contord)
        return (X - np.exp(-1.0j * p) * XT).ravel()

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AL.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)

        if p == 0:
            XL = np.trace(X) * (C @ C.T.conj())
            return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()
        else:
            return (X - np.exp(+1.0j * p) * XT).ravel()

    Ap, Ah = [], []
    for i, p in enumerate(mom_vec):
        if p / np.pi == 0:
            sp -= calc_expectation_val(sp, AC, np.eye(D)) * np.eye(d)
            sm -= calc_expectation_val(sm, AC, np.eye(D)) * np.eye(d)
            print('p', p)

        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

            tensors = [lz, B, -sz, (phase * AC).conj()]
            indices = [(3, 4), (4, 2, -2), (1, 2), (3, 1, -1)]
            contord = [3, 4, 1, 2]
            left_vec = nc.ncon(tensors, indices, contord)

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            left_env_op = spspla.LinearOperator((D * D, D * D),
                                                matvec=left_env
                                                )

            right_env_op = spspla.LinearOperator((D * D, D * D),
                                                 matvec=right_env
                                                 )

            LB = spspla.gmres(left_env_op, left_vec.ravel(),
                                           x0=rand_init.ravel(),
                                           rtol=1e-14,
                                           atol=1e-14
                                           )[0].reshape(D, D)

            RB = spspla.gmres(right_env_op, right_vec.ravel(),
                                            x0=rand_init.ravel(),
                                            rtol=1e-14,
                                            atol=1e-14
                                            )[0].reshape(D, D)

            tensors = [lz, B, sp, AC.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sp1 = nc.ncon(tensors, indices, contord)

            tensors = [lz, AL, sp, AL.conj(), RB]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 6), (5, 6)]
            contord = [3, 4, 5, 6, 1, 2]
            sp2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, sp, AR.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sp3 = nc.ncon(tensors, indices, contord)

            tensors = [lz, B, sm, AC.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sm1 = nc.ncon(tensors, indices, contord)

            tensors = [lz, AL, sm, AL.conj(), RB]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 6), (5, 6)]
            contord = [3, 4, 5, 6, 1, 2]
            sm2 = nc.ncon(tensors, indices, contord)

            tensors = [LB, AR, sm, AR.conj()]
            indices = [(3, 4), (4, 2, 5), (1, 2), (3, 1, 5)]
            contord = [3, 4, 5, 1, 2]
            sm3 = nc.ncon(tensors, indices, contord)

            Ap.append(np.abs(sp1
                           + np.exp(+1j * p) * sp2
                           + np.exp(-1j * p) * sp3
                           )
                      )

            Ah.append(np.abs(sm1
                           + np.exp(+1j * p) * sm2
                           + np.exp(-1j * p) * sm3
                            )
                      )

    Ap = np.array(Ap).reshape(mom_vec.size, excit_states.shape[2])
    Ah = np.array(Ah).reshape(mom_vec.size, excit_states.shape[2])
    return Ap, Ah

def dynamic_struc_fact(AL, AR, C,
             excit_energy, excit_states,
             mom_vec, O):

    D = AL.shape[0]
    d = AL.shape[1]

    AC = np.tensordot(AL, C, axes=(2, 0))

    VL = spla.null_space(AL.conj().reshape(D * d, D).T)
    VL = VL.reshape(D, d, (d - 1) * D)

    def right_env(X):
        X = X.reshape(D, D)

        tensors = [AL, X, AL.conj()]
        indices = [(-1, 1, 2), (2, 3), (-2, 1, 3)]
        contord = [2, 3, 1]
        XT = nc.ncon(tensors, indices, contord)

        if p == 0:
            XL = np.trace(X) * (C @ C.T.conj())
            return (X - np.exp(+1.0j * p) * (XT - XL)).ravel()
        else:
            return (X - np.exp(+1.0j * p) * XT).ravel()

    O = (O
         - nc.ncon([AC, O, AC.conj()], [[1, 3, 4], [2, 3], [1, 2, 4]])
           * np.eye(d)
           )

    dsf = []
    for i, p in enumerate(mom_vec):
        for j in range(excit_states.shape[2]):
            X = excit_states[i,:,j].reshape((d - 1) * D, D)
            B = np.tensordot(VL, X, axes=(2, 0))

            tensors = [B, AC.conj()]
            indices = [(-1, 2, 1), (-2, 2, 1)]
            contord = [1, 2]
            right_vec = nc.ncon(tensors, indices, contord)

            rand_init = np.random.rand(D, D) - 0.5

            right_env_op = spspla.LinearOperator((D * D, D * D),
                                                 matvec=right_env
                                                 )

            RB = spspla.gmres(right_env_op, right_vec.ravel(),
                                            x0=rand_init.ravel(),
                                            rtol=1e-14,
                                            atol=1e-14
                                            )[0].reshape(D, D)

            tensors = [B, O, AC.conj()]
            indices = [(3, 2, 4), (1, 2), (3, 1, 4)]
            contord = [3, 4, 1, 2]
            t1 = nc.ncon(tensors, indices, contord)

            tensors = [AL, O, AL.conj(), RB]
            indices = [(3, 2, 4), (1, 2), (3, 1, 5), (4, 5)]
            contord = [4, 5, 3, 1, 2]
            t2 = nc.ncon(tensors, indices, contord)

            dsf.append(np.abs(t1 + np.exp(+1j * p) * t2))
    return np.array(dsf).reshape(mom_vec.size, excit_states.shape[2])
