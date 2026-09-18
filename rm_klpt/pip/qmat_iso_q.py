from sage.all import *
proof.all(False)

def _lll_reduced_basis(I):
    B = I.basis()
    M = []
    for a in B:
        M.append([QQ(2)*(a*b.conjugate()).reduced_trace() for b in B])
    G = Matrix(QQ, M)
    U = G.LLL_gram().transpose()
    return [sum(c*beta for c, beta in zip(row, B)) for row in U]

def hensel_solve(b, q, e):
    """
    Use Hensel lifting to solve the equation x^2 + y^2 - y = -b modulo q^e.
    """
    # TODO: actually use hensel, also maybe check derivatives
    Zn = Integers(q**e)
    c1 = None
    for x in range(1000):
        c2 = Zn(x)
        y = -b - c2**2 + c2
        if y.is_square() and (c2 != 0 or y != 0):
            c1 = sqrt(y)
            break

        c2 = Zn.random_element()
        y = -b - c2**2 + c2
        if y.is_square() and (c2 != 0 or y != 0):
            c1 = sqrt(y)
            break

    if c1 == None:
        # TODO: proper solution
        raise ValueError('solution not found for {b = } {q = } {e = }')

    assert c1**2 + c2**2 - c2 == -b
    return c1, c2

def find_coords(alpha, O_basis, denoms=False):
    """
    Given in input an element alpha in an order O and a basis of such order,
    compute the (integer) coordinates of alpha in the given basis.

    Input:
    - alpha: a quaternion element in O
    - O_basis: a basis for O
    - denoms: optional, default False; whether to accept an output with
      denominator (N.B. denominators means alpha is not in O)

    Output:
    - c1, c2, c3, c4: coordinates of alpha in the given basis
    """
    # Columns of M are the coefficients of the basis
    M = Matrix(QQ, 4)
    for i in range(4):
        for j in range(4):
            M[i, j] = O_basis[j][i]

    alpha = vector(QQ, 4, list(alpha))
    v = M.solve_right(alpha)

    if denoms:
        return v

    if not all(vi in ZZ for vi in v):
        raise ValueError(f'{alpha = } not in O = ({O_basis})')

    return [ZZ(vi) for vi in v]

class QMatModq:
    """
    Contstruct the isomorphism for prime powers
    """
    def __init__(self, O, q, e=1, O_basis=None):
        """
        Given a maximal order O of a quaternion algebra B_(p, infty) and a
        prime power n = q^e compute the isomorphism between O/nO and M_2(Z/nZ).

        Input:
        - O: the target order
        - q: the base prime of the modulus
        - e: the exponent of the modulus; if not given it is set as 1
        - O_basis: optional, a specific basis for O

        """
        B = O.quaternion_algebra()
        i, j, k = B.gens()

        self.B = B
        self.O = O
        if not O_basis:
            self.O_basis = O.basis()
        else:
            self.O_basis = O_basis

        self.q = q
        self.e = e

        p = B.ramified_primes()[0]
        self.p = p
        if p % 4 != 3:
            st = 'implemented only for p = 3 mod 4; replace the starting order'
            raise NotImplementedError(st)

        n = q**e
        Zn = Integers(n)

        self.n = n
        self.Zn = Zn

        # Start solving for O0: matrices for 1, i, (i+j)/2, (1+k)/2
        M1_0 = Matrix(Zn, 2, [1, 0, 0, 1])
        M2_0 = Matrix(Zn, 2, [0, 1, -1, 0])
        assert M2_0**2 == -M1_0

        # (i+j)/2 has norm (p+1)/4 and trace zero
        b = Zn((p+1)/4)

        # Find c1, c2 s.t. c1^2 + c2^2 - c2 = -b mod n
        c1, c2 = hensel_solve(b, q, e)

        M3_0 = Matrix(Zn, 2, [c1, c2, c2-1, -c1])

        assert M3_0**2 == -((p+1)//4) * M1_0
        assert M3_0*M2_0 + M2_0*M3_0 + M1_0 == 0

        # Now (1+k)/2 = 1 + i*(i+j)/2
        M4_0 = M1_0 + M2_0 * M3_0
        assert M4_0 == Matrix(Zn, 2, [c2, -c1, -c1, -c2+1])

        O0_basis = (B(1), i, (i+j)/2, (1+k)/2)
        O0 = B.maximal_order(order_basis=O0_basis)

        if self.O_basis == O0_basis:
            self.M_basis = [M1_0, M2_0, M3_0, M4_0]

        else:
            # Compute a connecting ideal between O0 and O
            I = O0 * O
            I *= I.norm().denominator()
            assert I.left_order() == O0 and I.right_order() == O

            N_I = ZZ(I.norm())

            # Find an element defining the conjugation mod n
            I_B = _lll_reduced_basis(I)
            while True:
                alpha = sum(randint(1, 20) * I_B[i] for i in range(4))
                if gcd(alpha.reduced_norm() / N_I, q) == 1:
                    break

            # Find the corresponding images
            self.M_basis = []
            O0_M = [M1_0, M2_0, M3_0, M4_0]

            om1, om2, om3, om4 = self.O_basis
            for om_i in [om1, om2, om3, om4]:
                gamma_i = alpha * om_i * alpha**(-1)

                v_i = find_coords(gamma_i, O0.basis(), denoms=True)

                d_i = v_i.denominator()
                if gcd(d_i, q) != 1:
                    print(f'{v_i = } this should not happen')
                    breakpoint()
                    raise ValueError('cannot invert denominators')

                v_i = (v_i * d_i).change_ring(ZZ)
                d_inv = inverse_mod(ZZ(d_i), n)

                M_i = sum(v_i[i] * O0_M[i] for i in range(4)) * d_inv

                assert M_i.det() == (om_i.reduced_norm()) % q**e
                assert M_i.trace() == (om_i.reduced_trace()) % q**e
                self.M_basis.append(M_i)

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

        beta = find_coords(alpha, self.O_basis)
        mat = sum(self.Zn(beta[i]) * self.M_basis[i] for i in range(4))
        return mat

    def mat_to_quat(self, M):
        """
        Given a matrix M in M_2(Zn) return a lift of M in O.

        Input:
        - M: a matrix in M_2(Z/nZ)

        Output:
        - alpha: a quaternion in O such that a mod n = M, as a list of
          coefficients
        """
        M1, M2, M3, M4 = self.M_basis
        a1, a2, a3, a4 = M1.list()
        b1, b2, b3, b4 = M2.list()
        c1, c2, c3, c4 = M3.list()
        d1, d2, d3, d4 = M4.list()

        A = Matrix(self.Zn, 4, [
            a1, b1, c1, d1,
            a2, b2, c2, d2,
            a3, b3, c3, d3,
            a4, b4, c4, d4
        ])
        v = vector(self.Zn, 4, M.list())
        elt = A.solve_right(v).list()
        return elt

    def project(self, alpha):
        return self.quat_to_mat(alpha)

    def lift(self, M):
        return self.mat_to_quat(M)

