import time
import logging

from sage.all import *

from .qmat_iso import QMatIso
from .ideals import (
    gen_equiv_prime_id,
    find_push_endo_short,
    pushforward,
    principal_generator,
    qgcd_short)
from .misc import eval_matrix, kernel_to_ideal_1d

proof.all(False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger_sh = logging.StreamHandler()
formatter = logging.Formatter('%(name)s [%(levelname)s] %(message)s')
logger_sh.setFormatter(formatter)
logger.addHandler(logger_sh)


def fill_square(I):
    """
    Given in input an ideal I, compute a matrix M in M_2(O) (where O is the
    left order of I) such that the first column generates I as a O-ideal, the
    second column generates O, and n(M) = n(I).

    Input:
    - I: a left O-ideal

    Output:
    - M: a matrix in M_2(O) where the first column generates I, the second
      generates O, and n(M) = n(I)
    """

    O = I.left_order()
    ell = I.norm()

    B = I.quaternion_algebra()
    i, j, k = B.gens()
    p = B.ramified_primes()[0]

    O0 = B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1-k)/2))
    if O != O0:
        raise NotImplementedError("only O0 supported for now")

    # Step 1: compute I_xi ~ I
    _t0 = time.time()
    logger.debug(f'{I.norm() % 4=}')
    if I.norm().is_prime() and I.norm() % 4 == 1:
        I_xi = I
        N_xi = I.norm()
        a = N_xi
    else:
        for I_xi, a in gen_equiv_prime_id(I):
            N_xi = ZZ(I_xi.norm())
            if N_xi % 4 == 1:
                break

    _t1 = time.time()
    logger.debug(f'Step 1 (I_xi): {_t1-_t0:.3f}s')

    # Step 2: compute I_chi
    l1, l2 = two_squares(N_xi)
    chi = l1 + l2*i
    I_chi = O*chi + O*N_xi

    assert I.left_order() == I_chi.left_order() == I_xi.left_order()
    _t2 = time.time()
    logger.debug(f'Step 2 (I_chi): {_t2-_t1:.3f}s')

    # Step 3: compute I_sigma matching the pushforward
    OmodN = QMatIso(O, N_xi)
    sigma = find_push_endo_short(I_xi, I_chi, OmodN=OmodN)

    s = ZZ(sigma.reduced_norm())

    I_sigma = O * sigma
    assert I_sigma.left_order() == I.left_order()

    _t3 = time.time()
    logger.debug(f'Step 3 (sigma): {_t3-_t2:.3f}s')

    # Step 4: fill the square
    I_gamma = pushforward(I_sigma, I_xi)

    # Compute c
    assert I_xi.right_order() == I_gamma.left_order()
    c_abar = principal_generator(ell * I_xi * I_gamma)
    c = (c_abar * a) / a.reduced_norm()

    assert s == ZZ(c.reduced_norm()) // ell

    # Compute b, d
    check = ell * chi * sigma
    # Fix automorphism
    if check != c_abar:
        chi *= (c_abar / check)

    x, y = qgcd_short(sigma.conjugate(), chi, O)

    b = x.conjugate()
    d = -y.conjugate()

    M = Matrix(O, 2, [a, b, c, d])
    _t4 = time.time()
    logger.debug(f'Step 4 (a, b, c, d): {_t4-_t3:.3f}s')
    logger.info(f'Building square total time: {_t4-_t0:.3f}s')
    return M


def kernel_to_matrix(K, O, ell, D):
    """
    Given in input a kernel (P1, P2) in E[l]^2, and O = End(E), compute a matrix
    M in M_2(O) such that M * (P1 P2)^T = 0 and n(M) = ell.

    Input:
    - K = (P1, P2): two points in E[l], forming the target kernel
    - O: the endomorphism ring of the starting curve
    - ell: the order of the points
    - D: an effective represetation of O. Currently, only O0 is supported, and
      hence D = (E0, i, j) is the starting curve E0 = x^3 + x together with the
      two maps i : (x, y) -> (-x, iy) and j : (x, y) -> (x^p, y^p). Can be
      replaced with any effective end ring, giving the action of a basis of O
      on E[ell]

    Output:
    - M: a matrix in M_2(O) such that M * K = 0 and n(M) = ell
    """

    _t0 = time.time()
    P1, P2 = K

    E = P1.curve()
    p = ZZ(E.base_ring().characteristic())
    E._order = (p+1)**2

    B = O.quaternion_algebra()
    i, j, k = B.gens()
    if O != B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1+k)/2)):
        raise NotImplementedError("only O0 supported for now")
    if not is_prime(ell):
        raise NotImplementedError("only ell prime supported")

    _, quat_i, quat_j = D
    valP = None

    # Compute the change of basis into (P, 0)
    P, Q = K
    if P == 0:
        MP = Matrix(O, 2, [0, 1, 1, 0])
        P = Q
    else:
        ePQ = pari.ellweilpairing(E, P, Q, ell)
        if ePQ == 1:
            # Case 1: Q is a multiple of P
            beta = B(Q.log(P))
        else:
            # Case 2: (P, Q) is a basis
            iP = quat_i(P)
            jP = quat_j(P)
            kP = quat_i(jP)
            valP = [P, iP, jP, kP]

            a1, b1 = iP.log((P, Q))
            a2, b2 = jP.log((P, Q))

            if b1 % ell != 0:
                b1_inv = inverse_mod(b1, ell)
                beta = B([-a1*b1_inv, b1_inv, 0, 0,])
            else:
                assert b2 % ell != 0
                b2_inv = inverse_mod(b2, ell)
                beta = B([-a2*b2_inv, 0, b2_inv, 0])
        MP = Matrix(O, 2, [1, 0, -beta, 1])

    assert eval_matrix(MP, K[0], K[1], D) == (P, 0)

    # Find the ideal I with kernel P
    I = kernel_to_ideal_1d(P, O, D)

    # Compute the matrix
    M = fill_square(I)
    assert eval_matrix(M, P, E(0), D) == (0, 0)

    # Compose with the change of basis
    M = M * MP
    assert eval_matrix(M, K[0], K[1], D) == (0, 0)
    _t1 = time.time()
    logger.info(f'Kernel to matrix total time: {_t1-_t0:.3f}s')

    return M
