from .qmat_iso_q import QMatModq
from sage.all import *
proof.all(False)


def mat_crt(res, mod):
    """
    CRT for matrices.

    Input:
    - res: a list of 2x2 matrices M_i defined over Zn_i
    - mod: the list of the n_i

    Output:
    - a matrix M such that M mod n_i == M_i
    """
    a_i = [ZZ(Mi[0, 0]) for Mi in res]
    b_i = [ZZ(Mi[0, 1]) for Mi in res]
    c_i = [ZZ(Mi[1, 0]) for Mi in res]
    d_i = [ZZ(Mi[1, 1]) for Mi in res]
    a, b, c, d = [crt(x, mod) for x in [a_i, b_i, c_i, d_i]]
    N = product(mod)
    return Matrix(Integers(N), 2, [a, b, c, d])


def quat_crt(res, mod):
    """
    CRT for quaternion elements.

    Input:
    - res: a list of 4-tuples, each representing the coordinates of a
      quaternion theta_i in the fixed basis mod n_i
    - mod: the list of the n_i

    Output:
    - the 4-tuple of coordinates of the quaternion element theta such that
      theta mod n_i = theta_i
    """
    a_i = [ZZ(ri[0]) for ri in res]
    b_i = [ZZ(ri[1]) for ri in res]
    c_i = [ZZ(ri[2]) for ri in res]
    d_i = [ZZ(ri[3]) for ri in res]
    a, b, c, d = [crt(x, mod) for x in [a_i, b_i, c_i, d_i]]
    return (a, b, c, d)


class QMatIso:

    def __init__(self, O, n, n_fac=None, O_basis=None):
        """
        Given a maximal order O of a quaternion algebra B_(p, infty) and an
        integer n compute the isomorphism between O/nO and M_2(Z/nZ).

        Input:
        - O: the target order
        - n: the modulus
        - n_fac: optional, the factorization of n
        - O_basis: optional, a fixed basis for O; warning: calling O.basis() on
          O = <1, i, (i+j)/2, (1-k)/2> does not return this as a basis

        Example usage:
        ```
        p = 2**10*3**10*5**2*7*11 - 1
        B = QuaternionAlgebra(-1, -p)
        i,j,k = B.gens()
        N = 503 * 509

        ZN = Integers(N)
        O0 = B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1-k)/2))

        OmodN = QMatIso(O0, N)
        alpha = O0.random_element()
        M = OmodN.quat_to_mat(alpha)
        ```
        """

        B = O.quaternion_algebra()
        i, j, k = B.gens()

        if not n_fac:
            n_fac = factor(n)

        self.B = B
        self.O = O
        if O_basis:
            self.O_basis = O_basis
        else:
            self.O_basis = O.basis()  # Must stay fixed
        self.n = n
        self.n_fac = list(n_fac)

        p = B.ramified_primes()[0]
        self.iso_mod_q = [QMatModq(O, q_i, e_i, self.O_basis)
                          for q_i, e_i in self.n_fac]
        self.mods = [ZZ(q_i**e_i) for q_i, e_i in self.n_fac]

    def quat_to_mat(self, alpha):
        """
        Given a quaternion alpha in O return the image of alpha in M_2(Z/nZ)

        Input:
        - alpha: a quaternion in O

        Output:
        - M: a matrix in M_2(Z/nZ), the image of alpha
        """
        if not alpha in self.O:
            raise ValueError(f"{alpha} not in the order")

        # Recover the matrix mod q_i
        Mi = [Omodq.quat_to_mat(alpha) for Omodq in self.iso_mod_q]
        M = mat_crt(Mi, self.mods)

        return M

    def mat_to_quat(self, M):
        """
        Given a matrix M in M_2(Zn) return a lift of M in O.

        Input:
        - M: a matrix in M_2(Z/nZ)

        Output:
        - alpha: a quaternion in O such that a mod n = M
        """
        # Recover the coordinates w.r.t. the basis
        elt_i = [Omodq.mat_to_quat(M) for Omodq in self.iso_mod_q]
        elt = quat_crt(elt_i, self.mods)
        alpha = sum(self.O_basis[i] * ZZ(elt[i]) for i in range(4))

        if not alpha in self.O:
            raise RuntimeError('the output is not in the order')

        return alpha

    def project(self, alpha):
        return self.quat_to_mat(alpha)

    def lift(self, M):
        return self.mat_to_quat(M)

    def lift_prime(self, M):
        """
        Given a matrix M in M_2(Zn) return a lift of M in O of prime norm.
        Since this is not always possible, after a fixed number of iterations
        return False.

        Input:
        - M: a matrix in M_2(Z/nZ)

        Output:
        - alpha: a quaternion in O such that a mod n = M and n(A) is prime
        """
        elt_i = [Omodq.mat_to_quat(M) for Omodq in self.iso_mod_q]
        elt = quat_crt(elt_i, self.mods)
        alpha = sum(self.O_basis[i] * ZZ(elt[i]) for i in range(4))

        alphas = []
        for _ in range(1000):
            # Should be more than enough for most sizes
            alphas.append(alpha.reduced_norm())
            if ZZ(alpha.reduced_norm()).is_prime():
                return alpha
            alpha += self.n
        breakpoint()
        return None


if __name__ == "__main__":
    rr = randint(1, 1000)
    print(f'{rr=}')
    set_random_seed(rr)

    p = 2**10*3**10*5**2*7*11 - 1

    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()

    # Smooth N
    N = 2**10 * 3**10 * 11 * 37
    # # Non-smooth N
    # N = 503 * 509

    ZN = Integers(N)
    O0 = B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1-k)/2))

    OmodN = QMatIso(O0, N)

    # Quaternion to matrices (mod n)
    for _ in range(10):
        alpha = O0.random_element()
        M = OmodN.quat_to_mat(alpha)
        alpha1 = B(OmodN.mat_to_quat(M))
        assert (alpha - alpha1)/N in O0

    # Generic order
    A = [randint(-100, 100) for _ in range(4)]
    alpha = sum([ai*bi for ai, bi in zip(A, O0.basis())])
    assert alpha in O0
    I = O0 * alpha + O0 * factor(alpha.reduced_norm())[-1][0]
    O1 = I.right_order()

    O1modN = QMatIso(O1, N)

    # Quaternion to matrices
    for _ in range(10):
        alpha = O1.random_element()
        M = O1modN.quat_to_mat(alpha)
        alpha1 = B(O1modN.mat_to_quat(M))
        assert (alpha - alpha1)/N in O1

    # Linearity
    for _ in range(10):
        alpha = O1.random_element()
        beta = O1.random_element()

        Ma = O1modN.quat_to_mat(alpha)
        Mb = O1modN.quat_to_mat(beta)

        assert Ma + Mb == O1modN.quat_to_mat(alpha + beta)
        assert Ma * Mb == O1modN.quat_to_mat(alpha * beta)

    print(f'Ok')

    # Matrices to quaternions

    # Linearity for matrices

    # Linearity for quaternions
