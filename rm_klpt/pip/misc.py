from sage.all import *

import cypari2
pari = cypari2.Pari()

def weil_pairing_pari(P, Q, N):
    """
    Weil pairing using pari.
    """
    return pari.ellweilpairing(P.curve(), P, Q, N)

def discrete_log_pari(a, base, order):
    """
    Discrete log using pari.
    """
    x = pari.fflog(a, base, order)
    return ZZ(x)

def BiDLP(R, P, Q, N, ePQ=None):
    """
    Given a basis P,Q of E[N] finds a,b such that R = [a]P + [b]Q.
    Optional: include the pairing e(P,Q).
    """
    if ePQ:
        pair_PQ = ePQ
    else:
        pair_PQ = weil_pairing_pari(P, Q, N)

    pair_a = weil_pairing_pari(R, Q, N)
    pair_b = weil_pairing_pari(R, -P, N)
    a = discrete_log_pari(pair_a, pair_PQ, N)
    b = discrete_log_pari(pair_b, pair_PQ, N)

    return a, b

def nnorm(M):
    """
    Compute the norm of M in M_2(O0).
    """
    M_adj = M.parent()([theta.conjugate() for theta in M.transpose().list()])
    M1 = M*M_adj
    return ZZ(M1.determinant())

def eval_end(theta, P, D, valP=None):
    """
    Evaluate the endomorpshism theta in End(E0) on P. D contains E0 and the
    maps corresponding to i and j. Optionally, valP contains the precomputation
    of values [P, iP, jP, kP].
    """
    if P == 0:
        return P
    E0, qt_i, qt_j = D

    if not valP:
        iP = qt_i(P)
        jP = qt_j(P)
        kP = qt_i(jP)
        valP = [P, iP, jP, kP]

    ell = P.order()
    thetaP = sum([GF(ell)(c)*d for c, d in zip(list(theta), valP)])
    return thetaP

def eval_matrix(M, P, Q, D):
    a, b, c, d = M.list()
    P1 = eval_end(a, P, D) + eval_end(b, Q, D)
    Q1 = eval_end(c, P, D) + eval_end(d, Q, D)
    return P1, Q1

def kernel_to_ideal_1d(P, O, D):
    """
    Given a point P, an order O and an effective end ring D mapping
    End(P.curve()) to O, compute the ideal with kernel P.
    """
    B = O.quaternion_algebra()
    p = B.ramified_primes()[0]
    P.curve()._order = (p+1)**2
    ell = P.order()

    # Precompute images of P
    E0, qt_i, qt_j = D
    valP = [P, qt_i(P), qt_j(P), qt_i(qt_j(P))]

    # Complete P to a basis (P, Q = T(P)) of E[ell]
    while True:
        T = B([randint(1, 20) for _ in range(4)])
        Q = eval_end(T, P, D, valP)
        ePQ = pari.ellweilpairing(P.curve(), P, Q, ell)
        if ePQ != 1:
            break

    # Find <tau, ell> generating <P1> using T
    while True:
        tau = B([randint(1, 10) for _ in range(4)])
        R = eval_end(tau, P, D, valP)
        a, b = log(R, (P, Q))
        tau = tau - a - b*T
        assert eval_end(tau, P, D, valP) == 0
        if eval_end(tau, Q, D) != 0:
            return O*tau + O*ell
