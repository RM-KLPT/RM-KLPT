from sage.matrix.constructor import matrix
from sage.matrix.special import block_matrix
from sage.modules.free_module_element import vector
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.finite_rings.finite_field_constructor import GF


class SplitAlgebra:
    r"""
    Class for a splitting of an algebra isomorphic to a matrix algebra over a finite field.

    Input :

    -A : A FiniteDimensionalAlgebra over a finite field
    -to_A: A map from your intended algebra to A.
    -from_A: A map from A to your intended algebra (e.g a lift)
    """

    def __init__(self, A, to_A=None, from_A=None):
        self.A = A
        self._to_A = to_A
        self._from_A = from_A
        self.n = ZZ(A.degree()).nth_root(2)
        self.k = A.base_ring()
        q = self.k.cardinality()
        while True:
            a = A.random_element()
            min_poly = a.minimal_polynomial()
            if min_poly.is_irreducible() and min_poly.degree() == self.n:
                break
        system = matrix([(a * c - c * a**q).vector() for c in A.basis()])
        while True:
            y = A(system.left_kernel().random_element())
            if y.is_invertible():
                break
        norming_power = ZZ((q**self.n - 1) / (q - 1))
        _, N = is_scalar(A, y**self.n)
        b = norm_equation_in_induced_field(a, N, norming_power)
        if not self.k.is_prime_field():
            p = self.k.characteristic()
            for i in range(self.k.degree()):
                if b**norming_power == y**self.n:
                    break
                else:
                    b = b**p
        yb = ~b * y
        cyclic_pres_basis = matrix(
            [(a**i * (yb) ** j).vector() for i in range(self.n) for j in range(self.n)]
        )
        aK = self.k.extension(min_poly, "aK").gen()
        frob = aK**q
        a_matrix = matrix([(aK**i).list() for i in range(1, self.n + 1)])
        y_matrix = matrix([((frob) ** i).list() for i in range(self.n)])

        self._A_to_mat = ~cyclic_pres_basis * matrix(
            [
                (a_matrix**i * y_matrix**j).list()
                for i in range(self.n)
                for j in range(self.n)
            ]
        )
        self._mat_to_A = ~self._A_to_mat

    def to_M(self, a):
        """
        Maps an element of the domain of ``self.to_A`` to a matrix.
        """
        if self._to_A is not None:
            a = self._to_A(a)
        return matrix(self.k, self.n, self.n, a.vector() * self._A_to_mat)

    def from_M(self, mat):
        """
        Maps a matrix to an element of the codomain of ``self.from_A``.
        """
        a = self.A(vector(mat) * self._mat_to_A)
        if self._from_A is not None:
            a = self._from_A(a)
        return a

    def check_isom(self):
        """
        Check that the isomorphism represented by ``self`` is indeed an isomorphism
        of algebras.
        """
        check_hom = all(
            matrix(self.k, self.n, self.n, a.vector() * self._A_to_mat)
            * matrix(self.k, self.n, self.n, b.vector() * self._A_to_mat)
            == matrix(self.k, self.n, self.n, (a * b).vector() * self._A_to_mat)
            for a in self.A.basis()
            for b in self.A.basis()
        )
        check_iso = (
            matrix(
                self.k, [a.vector() * self._A_to_mat for a in self.A.basis()]
            ).nullity()
            == 0
        )
        return check_hom and check_iso


def norm_equation_in_induced_field(a, N, norming_power):
    """
    Solve the norm equation N(x) = N for the relative field extension
    induced by a.

    Necessary if the base field of the algebra is not prime because Sage does not really handle relative finite field extensions.
    """
    k = a.parent().base_ring()
    if k.is_prime_field():
        min_poly = a.minimal_polynomial()
        K = k.extension(min_poly, name="aK")
        b = K(N).nth_root(norming_power)
    else:
        min_poly = absolute_minimal_polynomial(a)
        p = min_poly.base_ring().cardinality()
        K = GF(order=(p, min_poly.degree()), modulus=min_poly, name="aK")
        k_modulus = k.modulus()
        z = k_modulus.roots(ring=K, multiplicities=False)[0]
        NK = sum(c * z**i for i, c in enumerate(N.list()))
        b = NK.nth_root(norming_power)
    return sum(c * a**i for i, c in enumerate(b.list()))


def absolute_minimal_polynomial(elem):
    """
    Return the minimal polynomial of an element of A over the prime subfield
    of the base field of A.
    """
    A = elem.parent()
    k = A.base_ring()
    if k.is_prime_field():
        return elem.minimal_polynomial()
    n = k.degree() * A.degree()
    k = k.subfield(1)
    elem_pow = A.one()
    mat = matrix(k, 0, n, [])
    while mat.nullity() == 0:
        mat = block_matrix(
            [[mat], [matrix([sum([c.list() for c in elem_pow.vector()], [])])]]
        )
        elem_pow *= elem
    pol = PolynomialRing(k, "x")
    return pol(mat.left_kernel().basis()[0].list())


def is_scalar(A, x):
    """
    Check that an element of ``A`` is a scalar and returns
    the corresponding field element.
    """
    key, coef = A.one().monomial_coefficients().popitem()
    s = x[key] / coef
    if x == s * A.one():
        return True, s
    else:
        return False, None
