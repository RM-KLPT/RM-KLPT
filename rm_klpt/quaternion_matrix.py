# ***********************************************************************
# Copyright (C) 2026 Mickaël Montessinos
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ***********************************************************************
from itertools import batched
from copy import copy

from sage.misc.cachefunc import cached_function
from sage.algebras.quatalg.quaternion_algebra import QuaternionAlgebra_ab
from sage.algebras.finite_dimensional_algebras.finite_dimensional_algebra import (
    FiniteDimensionalAlgebra,
)
from sage.structure.element import Matrix
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import QuadraticField
from sage.rings.number_field.number_field import NumberField
from sage.matrix.constructor import matrix
from sage.matrix.special import companion_matrix
from sage.matrix.special import zero_matrix
from sage.matrix.special import block_matrix
from sage.matrix.special import diagonal_matrix
from sage.matrix.special import identity_matrix
from sage.matrix.special import elementary_matrix
from sage.arith.functions import lcm
from sage.arith.misc import gcd
from sage.modules.free_module_element import vector
from sage.misc.prandom import randint
from sage.rings.finite_rings.finite_field_constructor import GF
from sage.arith.misc import factor

from finite_algebras import SplitAlgebra
from pip.pip import fill_square


class QuaternionMatrix:
    """
    A wrapper class for matrices over a rational quaternion algebra.

    TODO: It would make more sense to have a class that inherits the Sage
    matrix classes, but this is simpler for a first implementation.

    INPUT:

    -arg0: a matrix with base ring a rational quaternion algebra
            or a rational quaternion algebra

    -arg1: if ``arg0`` is not a matrix, an integer or a list of list of elements
            of ``arg0``

    -arg2: if ``arg1`` is an integer, an integer

    -arg3: if ``arg1`` and ``arg2`` are integers, a list of elements of ``arg0`` or a list of rational numbers.
    """

    def __init__(self, arg0, arg1=None, arg2=None, arg3=None):
        if isinstance(arg0, Matrix):
            self._from_matrix(arg0)
        elif isinstance(arg0, QuaternionAlgebra_ab):
            if arg0.base_ring() != QQ:
                raise NotImplementedError(
                    "The base ring should be a rational\
                quaternion algebra."
                )
            if arg2 is None:
                self._from_matrix(matrix(arg0, arg1))
            else:
                if len(arg3) == arg1 * arg2 * 4:
                    arg3 = [arg0(v) for v in batched(arg3, 4)]
                self._from_matrix(matrix(arg0, arg1, arg2, arg3))
        else:
            raise ValueError("Invalid input.")

    def _from_matrix(self, matrix):
        """
        Ancillary function for generating an object form a matrix of
        quaternions
        """
        if not isinstance(matrix.base_ring(), QuaternionAlgebra_ab):
            raise TypeError(
                "The matrix's base ring must be a quaternion\
            algebra."
            )
        if matrix.base_ring().base_ring() != QQ:
            raise NotImplementedError(
                "The base ring must be defined over the rationals."
            )
        self._data = matrix
        self._base_ring = matrix.base_ring()
        self._splitting_field = QuadraticField(self._base_ring.invariants()[1])

    def __repr__(self):
        return self._data.__repr__()

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, key, newvalue):
        self._data[key] = newvalue

    def __mul__(self, other):
        if isinstance(other, QuaternionMatrix):
            return QuaternionMatrix(self._data * other._data)
        return QuaternionMatrix(self._data * other)

    def __rmul__(self, other):
        if isinstance(other, QuaternionMatrix):
            return QuaternionMatrix(other._data * self._data)
        return QuaternionMatrix(other * self._data)

    def __add__(self, other):
        return QuaternionMatrix(self._data + other._data)

    def __radd__(self, other):
        if isinstance(other, QuaternionMatrix):
            return self + other
        if other == 0:
            return self
        raise TypeError(f"Cannot add QuaternionMatrix with nonzero {type(other)}")

    def __neg__(self):
        return QuaternionMatrix(-self._data)

    def __sub__(self, other):
        return QuaternionMatrix(self._data - other._data)

    def __eq__(self, other):
        return self._data == other._data

    def __invert__(self):
        return self.inverse()

    def _adjunct(self):
        self._assert_square()
        a = self.base_ring().invariants()[0]
        K = self._splitting_field
        s1 = matrix(
            K, self.nrows(), self.ncols(), [K([c[0], c[2]]) for c in self.list()]
        )
        s2 = matrix(
            K, self.nrows(), self.ncols(), [K([c[1], c[3]]) for c in self.list()]
        )
        return block_matrix([[s1, a * s2.conjugate()], [s2, s1.conjugate()]])

    def _is_adjunction_hermitian(self):
        return self.base_ring().invariants()[0] == -1

    def _unadjunct(self, ad):
        self._assert_square()
        n = self.nrows()
        s1 = ad[0:n, 0:n]
        s2 = ad[n : 2 * n, 0:n]
        coefs = [
            self.base_ring()([a[0], b[0], a[1], b[1]])
            for (a, b) in zip(s1.list(), s2.list())
        ]
        return QuaternionMatrix(self.base_ring(), n, n, coefs)

    def base_ring(self):
        """
        Return the base ring of ``self``.
        """
        return self._base_ring

    def list(self):
        """
        Return a list of the coefficients of ``self``.
        """
        return self._data.list()

    def nrows(self):
        """
        Return the number of rows of ``self``.
        """
        return self._data.nrows()

    def ncols(self):
        """
        Return the number of collumns of ``self``.
        """
        return self._data.ncols()

    def is_square(self):
        """
        Return `true` if ``self`` is square.
        """
        return self.nrows() == self.ncols()

    def _assert_square(self):
        """
        Raise an error if ``self`` is not square.
        """
        if not self.is_square():
            raise ValueError("self is not square.")

    def matrix_space(self):
        """
        Return the matrix space of 2x2 matrices over ``self.base_ring()``
        Equivalent to ``MatrixSpace(self.base_ring(), 2, 2)``.
        """
        return self._data.matrix_space()

    def new_matrix(self):
        """
        Return a zero QuaternionMatrix.
        """
        return QuaternionMatrix(self._data.new_matrix())

    def identity_matrix(self):
        """
        Return the identity matrix in the same ring as ``self``.
        """
        self._assert_square()
        n = self.nrows()
        return QuaternionMatrix(identity_matrix(self.base_ring(), n))

    def elementary_matrix(self, i, j, c):
        """
        Return the matrix with same base ring, row number and column number
        as ``self``, with zero coefficients everywhere except at ``i,j``,
        where the coefficient is ``c`` instead.
        """
        mat = self.new_matrix()
        mat[i, j] = c
        return mat

    def scalar_matrix(self, b):
        """
        Return the scalar matrix ``b`` lying in the same space as self.

        INPUT:

        -b: element of ``self.base_ring()``
        """
        self._assert_square()
        n = self.nrows()
        return QuaternionMatrix(diagonal_matrix([b] * n, n, self.base_ring()))

    def conjugate_transpose(self):
        """
        Return the conjugate-transpose of ``self``.
        """
        return QuaternionMatrix(self._data.conjugate_transpose())

    def rational_space_basis(self):
        r"""
        Return a `\Q`-basis of the matrix space of ``self``.
        """
        return [
            QuaternionMatrix(a * b)
            for a in self.matrix_space().basis()
            for b in self.base_ring().basis()
        ]

    def rational_coefficients(self):
        """
        Return a list of rational coefficients of self.
        The output is coherent with the basis returned by
        ``self.rational_space_basis()``.

        OUTPUT: list
        """
        return [a for x in self.list() for a in x.coefficient_tuple()]

    def reduced_norm(self):
        """
        Return the reduced norm of self.
        """
        return QQ(self._adjunct().determinant())

    def reduced_trace(self):
        """
        Return the reduced trace of self.
        """
        return QQ(self._adjunct().trace())

    def reduced_characteristic_polynomial(self):
        """
        Return the reduced characteristic polynomial of self.
        """
        return self._adjunct().characteristic_polynomial().change_ring(QQ)

    def minimal_polynomial(self):
        """
        Return the minimal polynomial of self.
        """
        return self._adjunct().minimal_polynomial().change_ring(QQ)

    def inverse(self):
        """
        Return the inverse of self
        """
        self._assert_square()
        return self._unadjunct(self._adjunct().inverse())

    def is_rational_sqrt(self):
        """
        Check if ``self**2`` is a rational number
        """
        self._assert_square()
        M2 = self._data**2
        d = M2[0, 0]
        return (M2.is_scalar() and d in QQ), QQ(d)

    def denominator(self, order=None):
        """
        Return the smallest rational integer ``d`` such that all coefficients of ``d * self`` lie
        in ``order``.

        Uses ``self.base_ring().maximal_order()`` if ``order`` is ``None``.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        if order is None:
            B = self.base_ring()
            order = B.maximal_order()
        coefs = [order.basis_matrix().solve_left(vector(c)) for c in self.list()]
        return lcm([c.denominator() for c in sum(list(map(list, coefs)), [])])

    def numerator(self, order=None):
        """
        Return ``self.denominator(order) * self``

        INPUT:

        -order: a maximal order in ``self.base_ring()``.
        """
        return self.denominator(order) * self

    def mod_ell(self, ell, order):
        """
        Return the reduction of ``self`` modulo ``n`` with respect to ``order``
        """
        if not ell.is_prime():
            raise NotImplementedError("Only implemented for prime modulus")
        if not self.is_integral(order):
            raise ValueError("The matrix must be integral with respect to the order.")
        if ell.divides(self.base_ring().discriminant()):
            raise ValueError("ell must not divide the discriminant of the algebra.")
        splitting = quaternion_order_mod_ell(order, ell)
        return block_matrix([[splitting.to_M(c) for c in r] for r in self._data.rows()])

    def is_integral(self, order=None):
        """
        Checks if the coefficients of ``self`` are contained in ``order``.
        """
        if order is None:
            order = self.base_ring().maximal_order()
        return all(c in order for c in self.list())

    def is_hermitian(self):
        """
        Checks if ``self`` is a hermitian matrix.
        """
        self._assert_square()
        return self == self.conjugate_transpose()

    def is_positive_definite(self):
        """
        Checks if ``self`` is a positive definite hermitian matrix.
        """
        a, _ = self.base_ring().invariants()
        if a == -1:
            return self._adjunct().is_positive_definite()
        else:
            raise NotImplementedError(
                "Positive definiteness check currently implemented only for 2x2 matrices and base rings with -1 as first invariant."
            )

    def is_iko_matrix(self, order=None):
        """
        Check is ``self`` is an IKO-matrix. Than is, integral, hermitian and positive definite.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        if not self._is_adjunction_hermitian():
            raise NotImplementedError(
                "This is currently only implemented if the fist invariant of the base ring is -1"
            )
        return self.is_integral() and self.is_positive_definite()

    def rosati(self, g1, g2=None):
        """
        Return the image of ``self`` by the Rosati map with respect to ``g1``
        and ``g2`` (a generalisation of the Rosati involution)

        Uses ``g1`` for ``g2`` if ``g2`` is ``None``.

        INPUT:

        -g1, g2 : quaternion matrices that are IKO matrices
        """
        if g2 is None:
            g2 = g1
        return g2.inverse() * self.conjugate_transpose() * g1

    def conjugating_matrix(self, other):
        """
        Return a matrix P such that `P * self * P^-1 = other` if it exists.

        WARNING: Doesn't check that the input matrices are conjugated, and may
        loop forever if they aren't.

        INPUT:

        -other: a quaternion matrix that is conjugated to ``self``
        """
        self._assert_square()
        n = self.nrows()
        if self == other:
            return self.identity_matrix()
        id = self.identity_matrix()
        tensor = [(id, self), (-other, id)]
        rows = [
            b.apply_tensor_endomorphism(tensor) for b in self.rational_space_basis()
        ]
        system = matrix([mat.rational_coefficients() for mat in rows])
        ker = system.left_kernel()
        while True:
            v = sum(randint(-n, n) * e for e in ker.basis())
            coefs = [self.base_ring()(b) for b in batched(v, 4)]
            P = QuaternionMatrix(self.base_ring(), n, n, coefs)
            if P.reduced_norm() != 0:
                return P

    def apply_tensor_endomorphism(self, tensor):
        """
        Return ``tensor(self)``, where ``tensor`` is a list of pairs of
        matrices defining an element in the envelopping algebra of quaternion
        matrices.

        INPUT:
        - tensor: a list of pairs of QuaternionMatrices
        """
        return sum([t[0] * self * t[1] for t in tensor], self.new_matrix())

    def _centralizer_quats(self):
        """
        Return a basis of the centralizer of ``self`` as a standard basis of a `Q(self)`-quaternion algebra.
        Assumes that ``self`` has an irreducible minimal polynomial of degree ``self.nrows()``.
        """
        pol = self.minimal_polynomial()
        ref = ideal_embedding(self.base_ring(), NumberField(pol, "aK").ideal(1))
        isom = self.conjugating_matrix(ref)
        isomi = ~isom
        return [
            isomi * (b * self.identity_matrix()) * isom
            for b in self.base_ring().basis()
        ]

    def centralizer(self):
        r"""
        Compute a `\mathbb{Q}`-basis of the centralizer of ``self``.
        """
        pol = self.minimal_polynomial()
        if pol.is_irreducible() and pol.degree() == self.nrows():
            return self._centralizer_quats()
        iden = self.identity_matrix()
        system = matrix(
            [
                b.apply_tensor_endomorphism(
                    [(self, iden), (iden, -self)]
                ).rational_coefficients()
                for b in self.rational_space_basis()
            ]
        )
        ker = system.left_kernel().basis()
        return [
            QuaternionMatrix(self.base_ring(), self.nrows(), self.ncols(), list(e))
            for e in ker
        ]

    def coordinates_in_rational_subspace(self, basis):
        r"""
        Return the coordinates of ``self`` with respect to `\Q`-basis ``basis``
        of a subspace of ``self.matrix_space()``.

        INPUT:

        -basis: iterable of quaternion matrices

        OUTPUT:

        -vector or None
        """
        mat = matrix([b.rational_coefficients() for b in basis])
        if mat.nullity() != 0:
            raise ValueError("Input should be a basis of a subspace.")
        target = vector(self.rational_coefficients())
        try:
            return mat.solve_left(target)
        except ValueError:
            return None

    def matrix_space_order_Z_basis(self, order=None):
        r"""
        Return a `\Z`-basis of the matrix space of ``self``
        over ``order``.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        if order is None:
            B = self.base_ring()
            order = B.maximal_order()
        Obasis = [
            QuaternionMatrix(a * M)
            for M in self.matrix_space().basis()
            for a in order.basis()
        ]
        return Obasis

    def principal_ideal_Z_basis(self, order=None):
        r"""
        Return a `\Z`-basis of the principal right `M_2(order)`-ideal generated
        by ``self``.

        If ``order`` is ``None``, ``self.base_ring().maximal_order()`` is used
        instead.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        Obasis = self.matrix_space_order_Z_basis(order)
        return [self * b for b in Obasis]

    def is_integer(self):
        r"""
        Check if ``self`` is an integer.

        That is, a scalar matrix with coefficients in `\Z`.
        """
        check = self._data.is_scalar() and self[0, 0] in ZZ
        N = None
        if check:
            N = ZZ(self[0, 0])
        return check, N

    def is_rm_matrix(self, order=None):
        r"""
        Checks that the `QQ`-algebra generated by ``self`` is a totally real number field of degree ``self.nrows()`` and that the ring of integers of this field only contains integral matrices with respect to ``order``.

        If order is None, ``self.base_ring().maximal_order()`` is used instead.

        INPUT:

        -order: a maximal order of ``self.base_ring()`` or None

        OUTPUT:

        -a boolean
        -the number field generated by ``self`` if ``self`` is an rm matrix
        -a `ZZ`-basis of the ring of integers of this number field embedded as quaternion matrices.
        """
        self._assert_square()
        n = self.nrows()
        chi = self.minimal_polynomial()
        if chi.degree() != n:
            return False
        K = NumberField(chi)
        a = K.gen()
        if not K.is_totally_real():
            return False
        int_basis_matrices = [
            sum([c * a**i for i, c in b.list().enumerate()], self.new_matrix())
            for b in K.ring_of_integers().basis()
        ]
        if not all(b.is_integral(order) for b in int_basis_matrices):
            return False
        return True, K, int_basis_matrices

    def is_integral_unit(self, order=None):
        """
        Check if ``self`` is a unit in ``M_2(order)``.

        If ``order`` is ``None``, use ``self.base_ring().maximal_order()`` instead.

        INPUT:

        -QuaternionOrder or None
        """
        return self.is_integral(order) and self.reduced_norm().abs() == 1


def quaternion_order_norm_equation(order, N):
    """
    Return an element of reduced norm ``N`` in quaternion order ``order``.

    INPUT:

    -order: a maximal order in a rational quaternion algebra
    -N: A positive integer
    """
    s = order.quadratic_form().solve(N)
    if all(c in ZZ for c in s):
        return True, order(s * order.basis_matrix())
    else:
        return False, None


def _col_padding(mat):
    """
    Pads a matrix with more rows than collumns into a square matrix.

    Uses zeroes for the padding.

    INPUT:

    -mat: A matrix
    """
    pad = zero_matrix(mat.base_ring(), mat.nrows(), mat.nrows() - mat.ncols())
    return block_matrix([[mat, pad]], subdivide=False)


def lattice_subspace_intersection(lattice_basis, subspace_basis):
    r"""
    Compute a `\Z`-basis of the intersection of a `\Z`-lattice and a
    `\Q`-subspace of quaternion matrices.

    INPUT:

    -lattice_basis: an iterable of a basis of the `\Z`-lattice.
    -subspace_basis: an iterable of a basis of the `\Q`-subspace.

    OUTPUT:

    -list

    ALGORITHM:

    See Lemma 3.1 in Eﬃcient reductions among lattice problems by
    Daniele Micciancio.
    """
    L = matrix([v.rational_coefficients() for v in lattice_basis])
    G = matrix([v.rational_coefficients() for v in subspace_basis])
    H = G.right_kernel_matrix().transpose()
    H *= H.denominator()
    M = _col_padding(L * H)
    _, U, _ = M.smith_form(transformation=True, integral=True)
    n = len(lattice_basis)
    m = len(subspace_basis)
    res_vectors = (U * L).rows()[n - m :]
    B = lattice_basis[0].base_ring()
    m = lattice_basis[0].nrows()
    n = lattice_basis[0].ncols()
    return [QuaternionMatrix(B, m, n, tuple(v)) for v in res_vectors]


@cached_function
def quaternion_order_mod_ell(order, ell):
    r"""
    Output a local splitting of ``order`` at ``ell``.
    That is, an isomorphism to `M_2(\mathbb{Z}/\ell \mathbb{Z})`.
    """
    k = GF(ell)
    basis_matrix = order.basis_matrix()
    tables = [
        (basis_matrix * e.matrix() * ~basis_matrix).change_ring(k)
        for e in order.basis()
    ]
    A = FiniteDimensionalAlgebra(k, tables, assume_associative=True, assume_unital=True)

    def to_A(x):
        return A(vector(x.coefficient_tuple()) * ~basis_matrix)

    def from_A(x):
        return sum(ZZ(c) * b for c, b in zip(x.vector(), order.basis()))

    return SplitAlgebra(A, to_A, from_A)


def lift_GF_matrix_to_order(matrix, order):
    """
    Lift `2 \times 2` ``matrix`` over a finite field back to ``order``
    """
    ell = matrix.base_ring().cardinality()
    if not ell.is_prime():
        raise ValueError("The matrix must a prime field for base ring.")
    splitting = quaternion_order_mod_ell(order, ell)
    B = order.quaternion_algebra()
    return QuaternionMatrix(
        B,
        [
            [
                splitting.from_M(matrix.submatrix(i, j, 2, 2))
                for j in range(0, matrix.ncols(), 2)
            ]
            for i in range(0, matrix.nrows(), 2)
        ],
    )


def principalize_quaternion_matrix_ideal(order, gens):
    """
    Interface to the PIP solver
    """
    gens = copy(gens)
    n = gens[0].ncols()
    gen_matrix = matrix([gen.rational_coefficients() for gen in gens])
    order_matrix = matrix(
        [
            gen.rational_coefficients()
            for gen in gens[0].matrix_space_order_Z_basis(order)
        ]
    )
    gen_over_order = (
        (gen_matrix * ~order_matrix)
        .change_ring(ZZ)
        .hermite_form(include_zero_rows=False)
    )
    N = gen_over_order.determinant().nth_root(4)
    ideal_basis_matrix = gen_over_order * order_matrix
    ideal_Z_basis = [
        QuaternionMatrix(order.quaternion_algebra(), n, n, r)
        for r in (ideal_basis_matrix).rows()
    ]
    if gcd(N, order.discriminant()) != 1:
        gram_matrix = matrix(
            [
                [
                    sum(a.pair(b) for a, b in zip(left.list(), right.list()))
                    for left in ideal_Z_basis
                ]
                for right in ideal_Z_basis
            ]
        )
        U = gram_matrix.LLL_gram()
        ideal_basis_matrix = U.transpose() * ideal_basis_matrix
        reduced_ideal_basis = [
            QuaternionMatrix(order.quaternion_algebra(), n, n, r)
            for r in (ideal_basis_matrix).rows()
        ]
        while True:
            A = sum(randint(0, 1) * gen for gen in reduced_ideal_basis)
            NA = A.reduced_norm() // N
            if gcd(NA, order.discriminant()) == 1:
                break
        twisted = True
        gen_matrix = A.conjugate_transpose()
    else:
        twisted = False
        NA = N
        gen_matrix = QuaternionMatrix(
            block_matrix([[gen._data] for gen in ideal_Z_basis])
        )
    factors = sum([[ell] * n for ell, n in factor(NA)], [])
    res = gens[0].identity_matrix()
    for ell in factors:
        k = GF(ell)
        v = gen_matrix.mod_ell(ell, order).right_kernel().basis()[0]
        ws = (k ** (len(v))).span([v]).complement().basis()
        comp_mat = block_matrix(
            [[matrix([w for w in ws])], [zero_matrix(k, 1, len(v))]]
        )
        temp_gen = lift_GF_matrix_to_order(comp_mat, order)
        M = principalize_ideal_prime_norm(temp_gen, ell, order)
        gen_matrix = gen_matrix * ~M
        res = M * res
    if twisted:
        return ~res.conjugate_transpose() * A
    else:
        return res


def principalize_ideal_prime_norm(A, ell, order):
    """
    Use the PIP solver on an ideal of prime norm.
    """
    B = order.quaternion_algebra()
    k = GF(ell)
    A_ell = A.mod_ell(ell, order)
    if not A_ell.nullity() == 1:
        raise ValueError("A mod ell should have nullity equal to one.")
    w = A_ell.right_kernel().basis()[0]
    i, b = next(
        (i, vector(b))
        for i, b in enumerate(batched(list(w), n=2))
        if any(e != 0 for e in b)
    )
    pi = elementary_matrix(B, A.nrows(), row1=0, row2=i)
    splitting = quaternion_order_mod_ell(order, ell)
    b_mat = -identity_matrix(B, A.nrows())
    b_mat[i, i] = 1
    for j in range(A.nrows()):
        if j != i:
            b_mat[j, i] = splitting.from_M(_matrix_from_to(b, w[2 * j : 2 * j + 2]))
    I2 = order.left_ideal([splitting.from_M(matrix(k, 2, 2, [-b[1], b[0], 0, 0])), ell])
    M2 = fill_square(I2)
    M = QuaternionMatrix(
        block_matrix(
            [
                [M2, zero_matrix(B, 2, A.ncols() - 2)],
                [zero_matrix(B, A.nrows() - 2, 2), identity_matrix(B, A.nrows() - 2)],
            ]
        )
    )
    return M * b_mat * pi


def _matrix_from_to(v1, v2):
    """
    Compute a 2x2 matrix `M` such that `M v_1 = v_2`
    """
    if v1[0] != 0:
        return matrix([[v2[0] / v1[0], 0], [v2[1] / v1[0], 0]])
    else:
        return matrix([[0, v2[0] / v1[1]], [0, v2[1] / v1[1]]])


def ideal_embedding(B, ideal):
    r"""
    Output a matrix generating the regular representation of ``\mathbb{Z}_K``,
    with respect to a basis of ``ideal``, where ``K`` is ``ideal.number_field()``.
    """
    basis_matrix = matrix([b.list() for b in ideal.basis()])
    K = ideal.number_field()
    return QuaternionMatrix(
        (basis_matrix * K.gen().matrix() * ~basis_matrix).change_ring(B)
    )
