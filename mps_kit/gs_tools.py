import numpy as np
import ncon as nc
import scipy.linalg as spla
import scipy.sparse.linalg as spspla

def HeffTerms_two(AL, AR, C, Hl, Hr, h, ep):
    d = AL.shape[0]
    D = C.shape[0]

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

def Apply_HC_two(AL, AR, Hl, Hr, h, D, d, X):
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

def Apply_HAC_two(hL_mid, hR_mid, Hl, Hr, D, d, X):
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

def HeffTerms_three(AL, AR, C, Hl, Hr, h, ep):
    d = AL.shape[0]
    D = C.shape[0]

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

    Hl, _ = spspla.gmres(Ol, hl.ravel(), x0=Hl.ravel(), rtol=ep/100, atol=ep/100)
    Hr, _ = spspla.gmres(Or, hr.ravel(), x0=Hr.ravel(), rtol=ep/100, atol=ep/100)

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

def Apply_HC_three(hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d, X):
    X = X.reshape(D, D)

    t = hl_mid.transpose(0, 1, 3, 2).reshape(D * d * d, D) @ X
    t = t.reshape(D * d, d * D) @ AR.reshape(d * D, D)
    H1 = t.reshape(D, d * D) @ AR.conj().transpose(0, 2, 1).reshape(d * D, D)

    t = X @ hr_mid.transpose(2, 0, 1, 3).reshape(D, d * D * d)
    t = t.reshape(D, d, D, d).transpose(3, 0, 1, 2).reshape(d * D, d * D)
    t = AL.transpose(1, 0, 2).reshape(D, d * D) @ t
    H2 = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t.reshape(D * d, D)

    H3 = Hl @ X
    H4 = X @ Hr
    return (H1 + H2 + H3 + H4).ravel()

def Apply_HAC_three(hl_mid, hr_mid, AL, AR, Hl, Hr, h, D, d, X):
    X = X.reshape(D, d, D)

    t = hl_mid.reshape(D * d, D * d) @ X.reshape(D * d, D)
    H1 = t.reshape(D, d, D)

    t = AL.reshape(d * D, D) @ X.reshape(D, d * D)
    t = t.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
    t = t.reshape(d, D, d, d, D).transpose(1, 4, 0, 2, 3).reshape(D * D, d * d * d)
    t = t @ h.reshape(d * d * d, d * d * d).transpose(1, 0)
    t = t.reshape(D, D, d, d, d).transpose(0, 2, 3, 1, 4).reshape(D * d, d * D * d)
    t = AL.conj().transpose(2, 1, 0).reshape(D, D * d) @ t
    t = t.reshape(D * d, D * d) @ AR.conj().transpose(2, 0, 1).reshape(D * d, D)
    H2 = t.reshape(D, d, D)

    t = X.reshape(D, d * D) @ hr_mid.transpose(3, 2, 0, 1).reshape(d * D,d * D)
    H3 = t.reshape(D, d, D)

    t = Hl @ X.reshape(D, d * D)
    H4 = t.reshape(D, d, D)

    t = X.reshape(D * d, D) @ Hr
    H5 = t.reshape(D, d, D)
    return (H1 + H2 + H3 + H4 + H5).ravel()

def calc_new_A(AL, AR, AC, C):
    d = AL.shape[0]
    D = C.shape[0]

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

def calc_discard_weight_two(AL, AR, C, h, Hl, Hr):
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

def calc_discard_weight_three(AL, AR, C, h, Hl, Hr):
    def eff_ham(X):
        X = X.reshape(d, D, d, D)

        t = AL.transpose(1,0,2).reshape(D*d,D)@X.transpose(1,0,2,3).reshape(D,d*d*D)
        t = AL.transpose(1,0,2).reshape(D*d,D)@t.reshape(D,d*d*d*D)
        t = h.reshape(d**3,d**3)@t.reshape(D,d**3,d*D).transpose(1,0,2).reshape(d**3,D*d*D)
        t = AL.conj().transpose(2,0,1).reshape(D,d*D)@t.reshape(d,d**2,D,d*D).transpose(0,2,1,3).reshape(d*D,d*d*d*D)
        t = AL.conj().transpose(2,1,0).reshape(D,D*d)@t.reshape(D*d,d*d*D)
        H1 = t.reshape(D,d,d,D).transpose(1,0,2,3)

        t = (AL.reshape(d * D, D)
                              @ X.transpose(1, 0, 2, 3).reshape(D, d * d * D))
        t = t.reshape(d, D, d, d, D).transpose(0, 2, 3, 1, 4)
        t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
        t = t.reshape(d, d, d, D, D).transpose(0, 3, 1, 2, 4)
        t = AL.conj().reshape(d * D, D).T @ t.reshape(d * D, d * d * D)
        H2 = t.reshape(D, d, d, D).transpose(1, 0, 2, 3)

        t = X.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
        t = t.reshape(d, D, d, d, D).transpose(0, 2, 3, 1, 4)
        t = h.reshape(d**3, d**3) @ t.reshape(d**3, D * D)
        t = t.reshape(d, d, d, D, D).transpose(0, 1, 3, 2, 4)
        t = (t.reshape(d * d * D, d * D)
                            @ AR.conj().transpose(0, 2, 1).reshape(d * D, D))
        H3 = t.reshape(d, d, D, D).transpose(0, 2, 1, 3)

        t = X.reshape(d * D * d, D) @ AR.transpose(1, 0, 2).reshape(D, d * D)
        t = (t.reshape(d * D * d * d, D)
                                @ AR.transpose(1, 0, 2).reshape(D, d * D))
        t = t.reshape(d * D, d, d, d, D).transpose(0, 4, 1, 2, 3)
        t = t.reshape(d * D * D, d**3) @ h.reshape(d**3, d**3).T
        t = t.reshape(d * D, D, d**2, d).transpose(0, 2, 3, 1)
        t = (t.reshape(d * D * d * d, d * D)
                            @ AR.conj().transpose(0, 2, 1).reshape(d * D, D))
        t = (t.reshape(d * D * d, d * D)
                            @ AR.conj().transpose(0, 2, 1).reshape(d * D, D))
        H4 = t.reshape(d, D, d, D)

        t = Hl @ X.transpose(1, 0, 2, 3).reshape(D, d * d * D)
        H5 = t.reshape(D, d, d, D).transpose(1, 0, 2, 3)

        t = X.reshape(d * D * d, D) @ Hr
        H6 = t.reshape(d, D, d, D)
        return (H1 + H2 + H3 + H4 + H5 + H6).ravel()

    two_site = nc.ncon([AL, C, AR], [(-1, -2, 1), (1, 2), (-3, 2, -4)])

    H = spspla.LinearOperator((d * D * d * D, d * D * d * D), matvec=eff_ham)

    w, v = spspla.eigs(H, k=1, which='SR', v0=two_site.ravel(), tol=1e-12, return_eigenvectors=True)

    s = spla.svdvals(v[:, 0].reshape(d * D, d * D))

    return sum(ss**2 for ss in s[D:])

def dynamic_expansion_two(AL, AR, C, Hl, Hr, h, delta_D):
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

def dynamic_expansion_three(AL, AR, C, Hl, Hr, h, delta_D):
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

        tensors = [AL, AL, X, h, AL.conj(), AL.conj()]
        indices = [(7, 1, 2), (8, 2, 4), (9, 4, -3, -4),
                   (5, 6, -1, 7, 8, 9), (5, 1, 3), (6, 3, -2)]
        contord = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        H1 = nc.ncon(tensors,indices,contord)

        tensors = [AL, X, h, AL.conj()]
        indices = [(4, 1, 2), (5, 2, 6, -4), (3, -1, -3, 4, 5, 6), (3, 1, -2)]
        contord = [1, 2, 3, 4, 5, 6]
        H2 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, h, AR.conj()]
        indices = [(4, -2, 5, 2), (6, 2, 1), (-1, -3, 3, 4, 5, 6), (3, -4, 1)]
        contord = [1, 2, 3, 4, 5, 6]
        H3 = nc.ncon(tensors,indices,contord)

        tensors = [X, AR, AR, h, AR.conj(), AR.conj()]
        indices = [(-1, -2, 7, 4), (8, 4, 2), (9, 2, 1),
                   (-3, 5, 6, 7, 8, 9), (5, -4, 3), (6, 3, 1)]
        contord = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        H4 = nc.ncon(tensors,indices,contord)

        tensors = [Hl, X]
        indices = [(-2, 1), (-1, 1, -3, -4)]
        H5 = nc.ncon(tensors,indices)

        tensors = [X, Hr]
        indices = [(-1, -2, -3, 1), (1, -4)]
        H6 = nc.ncon(tensors,indices)
        return (H1 + H2 + H3 + H4 + H5 + H6).ravel()

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
