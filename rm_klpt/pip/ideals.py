from sage.all import *

from .qmat_iso import QMatIso


def lll_reduced_basis(I):
    """
    Given an ideal I, computes an LLL-reduced basis of I.

    Input:
    - I: an ideal

    Output:
    - 4 elements of I forming an LLL-reduced basis
    """
    B = I.basis()
    M = []
    for a in B:
        M.append([QQ(2)*(a*b.conjugate()).reduced_trace() for b in B])
    G = Matrix(QQ, M)
    U = G.LLL_gram().transpose()
    return [sum(c*beta for c, beta in zip(row, B)) for row in U]


def gen_equiv_prime_id(I):
    """
    Given an ideal I, returns an equivalent ideal of (small) prime norm and the
    element generating it, as an iterable

    Input:
    - I: an ideal

    Yield:
    - J: an equivalent ideal to I of small prime norm
    - beta: the element such that I * beta.conjugate() / n(I) = J
    """
    I_basis = lll_reduced_basis(I)
    nI = I.norm()
    p = I.quaternion_algebra().ramified_primes()[0]

    # There is a 1 / log(p) chance of hitting a prime
    B = p.nbits()  # 8 * log(p)^3 primes expected

    while True:
        # Could be made deterministic
        c_i = [randint(-B, B) for _ in range(4)]
        beta = sum([c_i[i] * I_basis[i] for i in range(4)])

        nJ = ZZ(beta.reduced_norm() / nI)
        if nJ.is_prime():
            J = I * beta.conjugate() * (1 / nI)
            yield J, beta


def equiv_prime_id(I):
    """
    Given an ideal I, returns an equivalent ideal of (small) prime norm and the
    element generating it, as an iterable

    Input:
    - I: an ideal

    Output:
    - J: an equivalent ideal to I of small prime norm
    - beta: the element such that I * beta.conjugate() / n(I) = J
    """
    return next(gen_equiv_prime_id(I))


def ideal_generator(I):
    """
    Given an ideal I of norm N finds an element alpha such that I = (alpha, N)

    Input:
    - I: an ideal

    Output:
    - alpha such that I = (alpha, n(I))
    """
    p = I.quaternion_algebra().ramified_primes()[0]
    bound = max(ceil(10*log(p, 2)), 100)
    while True:
        alpha = sum([b * randint(-bound, bound) for b in I.basis()])
        if gcd(alpha.reduced_norm(), I.norm()**2) == I.norm():
            return alpha


def principal_generator(I):
    """
    Given a principal ideal I, computes a generator

    Input:
    - I: an ideal

    Output:
    - alpha: a generator of I if I is principal, None otherwise
    """
    N = I.norm()
    for gen in lll_reduced_basis(I):
        if gen.reduced_norm() == N:
            return gen

    raise ValueError('ideal not principal')


def decompose_in_ideal(alpha, sigma, N, O, N_fac=None):
    """
    Given alpha in I = (sigma, N), write alpha = x * sigma + y * N with
    x and y in O.

    Input:
    - alpha: target quaternion element in I
    - sigma, N: generators of I; N can be either an integer or a sage
      factorization [(p_i, e_i)]
    - O: left order of I
    - fac_N: optional, factorization of N; otherwise, N is factored on the fly

    Output:
    - x, y: elements of O such that alpha = x * sigma + y * N
    """
    if not N_fac:
        N_fac = factor(N)

    B = O.quaternion_algebra()
    p = B.ramified_primes()[0]

    OmodN = QMatIso(O, N, n_fac=N_fac)
    mat_sigma = OmodN.quat_to_mat(sigma)
    mat_alpha = OmodN.quat_to_mat(alpha)

    # Solve x * sigma = alpha mod N
    mat_x = mat_sigma.solve_left(mat_alpha)
    x = OmodN.mat_to_quat(mat_x)

    y = (alpha - x * sigma) / N
    assert x in O and y in O
    assert x*sigma + y*N == alpha
    return x, y


def pushforward(I, J):
    """
    Compute the pushforward of I by J

    Input:
    - I, J: two left O-ideals

    Output:
    - K = [I]_* J
    """
    if I.left_order() != J.left_order():
        raise ValueError('different left orders')
    if gcd(I.norm(), J.norm()) != 1:
        raise ValueError('pushforward not defined: norms are not coprime')

    O = J.right_order()
    return J.conjugate()*I + O*I.norm()


def compute_isomorphism(O1, O2):
    """
    Given two isomorphic maximal orders O1 and O2 computes the element alpha
    such that alpha * O1 * alpha^-1 = O2

    Input:
    - O1, O2: two isomorphic maximal orders

    Output:
    - alpha such that alpha * O1 * alpha^-1 = O2
    """
    I = O1 * O2
    I *= I.norm().denominator()
    alpha = principal_generator(I)
    if not alpha:
        raise ValueError('orders are not isomorphic')
    return alpha


def match_and_mul(I, J):
    """
    Given two ideals I and J such that the right order of I is isomorphic to
    the left order of J, find the isomorphism phi and then compute the product
    I * phi(J)

    Input:
    - I, J: two ideals such that the right order of I is isomorphic to the left
      order of J

    Output:
    - I * phi(J), where phi is the isomorphism O_L(J) -> O_R(I)
    """
    alpha = compute_isomorphism(J.left_order(), I.right_order())
    Jcon = alpha**(-1) * J * alpha
    return I * Jcon


def find_push(I, J, OmodN=None):
    """
    Given two ideals I and J find an ideal I_theta such that J is the
    pushforward of I by I_theta.

    Input:
    - I, J: two ideals of the same norm
    - OmodN: optional, a QMatIso object representing the isomorphism O_L(J) /
      NO_L(J) -> M_2(N) where N = n(J)

    Output:
    - theta such that [I_theta]_* I = J
    """
    N = I.norm()
    if J.norm() != N:
        raise ValueError(f'n(I) = {N} different from n(J) = {J.norm()}')

    # Make the left orders match
    K = I.left_order() * J.left_order()
    K *= K.norm().denominator()  # connecting ideal

    for K, _ in gen_equiv_prime_id(K):
        if gcd(ZZ(K.norm()), N) == 1:
            break

    I_prime = pushforward(I, K)

    alpha = compute_isomorphism(I_prime.left_order(), J.left_order())
    I_prime = alpha**(-1) * I_prime * alpha

    theta = find_push_endo(I_prime, J, OmodN=OmodN)
    I_theta = alpha * theta * alpha**(-1)
    return K * I_theta


def find_push_endo(I, J, OmodN=None, prime_out=False):
    """
    Given two left O-ideals I and J find theta in O such that J is the
    pushforward of I by (the principal ideal of) theta.

    Input:
    - I, J: two left O-ideals of the same norm
    - OmodN: optional, a QMatIso object representing the isomorphism O_L(J) /
      NO_L(J) -> M_2(N) where N = n(J)
    - prime_out: default false, whether to require the output theta to have
      prime norm

    Output:
    - theta in O such that [I_theta]_* I = J
    """
    O = I.left_order()
    if J.left_order() != O:
        raise ValueError('left orders do not match')

    N = ZZ(I.norm())
    if ZZ(J.norm()) != N:
        raise ValueError('norms do not match')
    ZN = Integers(N)

    if not OmodN:
        OmodN = QMatIso(O, N)

    alpha_I = ideal_generator(I)
    alpha_J = ideal_generator(J)

    mat_I = OmodN.project(alpha_I)
    mat_J = OmodN.project(alpha_J)

    K_I = mat_I.right_kernel().basis()[0]
    K_J = mat_J.right_kernel().basis()[0]

    system = Matrix(ZN, [
        [K_I[0], K_I[1], 0, 0],
        [0, 0, K_I[0], K_I[1]]
    ])

    theta_0 = system.solve_right(K_J)

    theta = Matrix(ZN, [[0, 0], [0, 0]])
    while True:
        if not theta.is_singular():
            # Valid theta: try to lift
            if prime_out:
                theta = OmodN.lift_prime(theta)
            else:
                theta = OmodN.lift(theta)
            if theta:
                return theta

        theta = theta_0 + system.right_kernel().random_element()
        theta = Matrix(ZN, [[theta[0], theta[1]], [theta[2], theta[3]]])


def find_push_endo_short(I, J, OmodN=None, prime_out=False):
    """
    Given two left O-ideals I and J find theta in O such that J is the
    pushforward of I by (the principal ideal of) theta, with n(theta) =
    O(sqrt(n(I)p)).

    Input:
    - I, J: two left O-ideals of the same norm
    - OmodN: optional, a QMatIso object representing the isomorphism O_L(J) /
      NO_L(J) -> M_2(N) where N = n(J)
    - prime_out: default false, whether to require the output theta to have
      prime norm

    Output:
    - theta in O such that [I_theta]_* I = J
    """
    O = I.left_order()
    if J.left_order() != O:
        raise ValueError('left orders do not match')

    N = ZZ(I.norm())
    if ZZ(J.norm()) != N:
        raise ValueError('norms do not match')
    ZN = Integers(N)
    B = O.quaternion_algebra()

    if not OmodN:
        OmodN = QMatIso(O, N)

    alpha_I = ideal_generator(I)
    alpha_J = ideal_generator(J)

    mat_I = OmodN.project(alpha_I)
    mat_J = OmodN.project(alpha_J)

    K_I = mat_I.right_kernel().basis()[0]
    K_J = mat_J.right_kernel().basis()[0]

    # Map ker(I) -> ker(J)
    system = Matrix(ZN, [
        [K_I[0], K_I[1], 0, 0],
        [0, 0, K_I[0], K_I[1]]
    ])

    ker_basis = system.right_kernel().basis()
    theta_0 = system.solve_right(K_J)

    # Build the lattice
    L_basis = [N * bi for bi in O.basis()]

    M0 = Matrix(ZN, 2, theta_0)
    L_basis.append(OmodN.lift(M0))
    for kk in ker_basis:
        Mk = Matrix(ZN, 2, kk + theta_0)
        L_basis.append(OmodN.lift(Mk))

    # Sanity
    # V = span([vector(QQ, list(v)) for v in L_basis[4:]], ZZ)
    # assert V.rank() == 3

    # Find a basis of the lattice
    vecs = [vector(QQ, list(v)) for v in L_basis]
    L = span(vecs, ZZ)
    L_basis = [B(v) for v in L.basis()]

    # LLL
    G = matrix(QQ, 4)

    for i in range(4):
        bi = L_basis[i]
        for j in range(4):
            bj = L_basis[j]

            # This is equivalent to Re(bi * conjugate(bj))
            val = ((bi + bj).reduced_norm() -
                   bi.reduced_norm() - bj.reduced_norm()) / 2
            G[i, j] = val
    U = G.LLL_gram().transpose()

    # Sanity
    # cov = G.determinant() * 16
    # assert cov == B.ramified_primes()[0]**2 * N**2

    red_basis = [sum(c*beta for c, beta in zip(row, L_basis)) for row in U]

    # Sanity
    # beta = red_basis[0]
    # J_prime = pushforward(I, O*beta)
    # alpha = compute_isomorphism(J_prime.left_order(), J.left_order())
    # J_prime = alpha**(-1)*J_prime*alpha
    # assert J_prime.conjugate().is_left_equivalent(J.conjugate())

    if not prime_out:
        # This should never happen I think, but is easy to fix
        for beta in red_basis:
            if beta.reduced_norm() % N != 0:
                return beta

    # Generate a short prime output
    bd = B.ramified_primes()[0].nbits()  # 8 * log(p)^3 primes expected
    while True:
        c_i = [randint(-bd, bd) for _ in range(4)]
        beta = sum([c_i[i] * red_basis[i] for i in range(4)])

        n_beta = ZZ(beta.reduced_norm())
        if n_beta.is_prime():
            return beta


def qgcd(a, b, O):
    """
    Given two elements a, b in O find x, y such that 1 = xa + yb.

    Input:
    - a, b: two quaternions
    - O: an order

    Output:
    - x, y: elements in O such that 1 = xa + yb
    """
    one, u, v = xgcd(a.reduced_norm(), b.reduced_norm())
    if one != 1:
        raise ValueError("norm of a and b not coprime")

    x = u * a.conjugate()
    y = v * b.conjugate()

    return x, y


def qgcd_short(a, b, O):
    """
    Given two elements a, b in O find short x, y such that 1 = xa + yb.

    Input:
    - a, b: two quaternions
    - O: an order

    Output:
    - x, y: elements in O of small norm such that 1 = xa + yb
    """
    # Find first solution
    B = O.quaternion_algebra()
    x, y = qgcd(a, b, O)

    # Construct the lattice of homogeneous solutions
    L0_basis = [vector(QQ, list(bi)) for bi in O.basis()]
    L1_basis = [vector(QQ, list(bi * b * a**-1)) for bi in O.basis()]

    L = span(L0_basis, ZZ).intersection(span(L1_basis, ZZ))
    L_basis = [B(v) for v in L.basis()]

    # Sanity: elements of the lattice can be subtracted from x, y
    # for z in L_basis:
    #     assert (x - z) * a - 1 in O * b

    # LLL
    G = matrix(QQ, 4)
    for i in range(4):
        bi = L_basis[i]
        for j in range(4):
            bj = L_basis[j]
            val = ((bi + bj).reduced_norm() -
                   bi.reduced_norm() - bj.reduced_norm()) / 2
            G[i, j] = val

    U = G.LLL_gram().transpose()
    red_basis = [sum(c*beta for c, beta in zip(row, L_basis)) for row in U]

    # Babai rounding
    M = Matrix(QQ, 4)
    for i in range(4):
        M.set_column(i, list(red_basis[i]))
    v = vector(QQ, list(x))

    coeffs = M**-1 * v
    assert M * coeffs == v
    coeffs = [int(i) for i in coeffs]

    c_v = sum(red_basis[i] * coeffs[i] for i in range(4))
    x1 = x - c_v

    assert x1 * a - 1 in O * b
    y1 = (1 - x1 * a) * b**-1
    assert y1 in O
    assert x1 * a + y1 * b == 1

    return x1, y1
