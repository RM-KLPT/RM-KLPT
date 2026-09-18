# *********************************************************************** Copyright (C) 2009 William Stein <wstein@gmail.com>
# Copyright (C) 2009 Jonathan Bober <jwbober@gmail.com>
# Copyright (C) 2014 Julian Rueth <julian.rueth@fsfe.org>
# Copyright (C) 2021 Peter Bruin <P.J.Bruin@math.leidenuniv.nl>
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

"""
Several portions of this file are directly adapted or copied from
the quaternion_algebra module of Sagemath.
"""

from itertools import product

from sage.categories.algebras import Algebras
from sage.misc.cachefunc import cached_method
from sage.misc.misc_c import prod
from sage.rings.integer_ring import ZZ
from sage.structure.element import RingElement
from sage.structure.parent import Parent
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.functions.other import ceil
from sage.misc.functional import log
from sage.misc.functional import sqrt
from sage.algebras.finite_dimensional_algebras.finite_dimensional_algebra import (
    FiniteDimensionalAlgebra,
)
from sage.misc.prandom import randint
from sage.modules.free_quadratic_module_integer_symmetric import IntegralLattice

from hnf import PseudoMatrix
from quaternion_ideal import QuaternionFractionalIdeal_nf
from represent_integer import RepresentIntegerSolver
from nf_utility import (
    narrow_class_group_support,
    totally_positive_units_up_to_squares,
    zeta_min_1,
    local_generator,
    small_lift,
    reduce_mod,
)
from finite_algebras import SplitAlgebra


class QuaternionOrder_nf(Parent):
    r"""
    An order in a quaternion algebra defined over a number field.
    """

    def __init__(self, A, gens, check=True) -> None:
        r"""
        INPUT:

        - ``A`` -- a quaternion algebra
        - ``gens`` -- list of integral quaternions in ``A`` or PseudoMatrix
        - ``check`` -- whether to do type and other consistency checks
        """
        if check:
            # right data type
            if not isinstance(gens, (list, tuple, PseudoMatrix)):
                raise TypeError("basis must be a list or tuple or a PseudoMatrix")
        K = A.base_ring()
        if not isinstance(gens, PseudoMatrix):
            gens = PseudoMatrix(
                [K.ideal(1) for _ in gens],
                matrix([A(x).coefficient_tuple() for x in gens]),
            )
        self.__pseudo_matrix, _ = gens.pseudo_hermite_form()
        self.__integral_basis = [
            A(list(v)) for v in self.__pseudo_matrix.integral_basis()
        ]

        if check:
            if self.__pseudo_matrix.matrix().nrows() != 4:
                raise ValueError("lattice must have rank 4")

            if not self.__pseudo_matrix.lattice_contains(vector(A.one())):
                raise ValueError("lattice does not contain 1")

            for a in self.__integral_basis:
                for b in self.__integral_basis:
                    if not self.__pseudo_matrix.lattice_contains(vector(a * b)):
                        raise ValueError("lattice is not a ring")

        self.__quaternion_algebra = A

        if K.degree() == 2:
            _, p = A.invariants()
            self.__solver = RepresentIntegerSolver(K, ZZ(-p))
        Parent.__init__(
            self,
            base=K.ring_of_integers(),
            facade=(A,),
            category=Algebras(ZZ).Facade().FiniteDimensional(),
        )

    def _element_constructor_(self, x):
        r"""
        Construct an element of this quaternion order from ``x``,
        or throw an error if ``x`` is not contained in the order.

        EXAMPLES::

            sage: Q.<i,j,k> = QuaternionAlgebra(-1,-19)
            sage: O = Q.quaternion_order([1,i,j,k])
            sage: O(1+i)
            1 + i
            sage: O(1/2)
            Traceback (most recent call last):
            ...
            TypeError: 1/2 does not lie in Order of Quaternion Algebra (-1, -19)
            with base ring Rational Field with basis (1, i, j, k)

        TESTS:

        Test for :issue:`32364`::

            sage: 1/5 in O
            False
            sage: j/2 in O
            False
        """
        y = self.quaternion_algebra()(x)
        if y not in self.unit_ideal():
            raise TypeError(f"{x!r} does not lie in {self!r}")
        return y

    def one(self):
        r"""
        Return the multiplicative unit of this quaternion order.

        EXAMPLES::

            sage: QuaternionAlgebra(-1,-7).maximal_order().one()
            1
        """
        return self.quaternion_algebra().one()

    def gens(self) -> tuple:
        r"""
        Return generators for ``self``.
        """
        return self.__integral_basis

    def ngens(self):
        r"""
        Return the number of generators (which is 4).
        """
        return len(self.__integral_basis)

    def gen(self, n):
        r"""
        Return the `n`-th generator.

        INPUT:

        - ``n`` -- integer between 0 and 3, inclusive
        """
        return self.__integral_basis[n]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, QuaternionOrder_nf)
            and self.quaternion_algebra() == other.quaternion_algebra()
            and all(b in other for b in self.integral_basis())
            and all(b in self for b in other.integral_basis())
        )

    def __contains__(self, elem) -> bool:
        return self.pseudo_matrix().lattice_contains(elem.coefficient_tuple())

    def __hash__(self) -> int:
        r"""
        Compute the hash of ``self``.
        """
        return hash((self.__quaternion_algebra, self.__integral_basis))

    def integral_basis(self):
        r"""
        Return fix choice of basis for this quaternion order.
        """
        return self.__integral_basis

    def pseudo_basis(self):
        """
        Return fix choice of pseudo-basis for this quaternion order.
        """
        ideals, vec = self.pseudo_matrix().pseudo_basis()
        quats = [self.quaternion_algebra()(list(v)) for v in vec]
        return ideals, quats

    def pseudo_matrix(self):
        return self.__pseudo_matrix

    def quaternion_algebra(self):
        r"""
        Return ambient quaternion algebra that contains this quaternion order.
        """
        return self.__quaternion_algebra

    def _repr_(self) -> str:
        r"""
        Return string representation of this order.
        """
        return f"Order of {self.quaternion_algebra()} with pseudo basis {self.pseudo_basis()}"

    def random_element(self, *args, **kwds):
        r"""
        Return a random element of this order.

        The args and kwds are passed to the random_element method of
        the integer ring, and we return an element of the form
        """
        ideals, quats = self.pseudo_basis()
        return sum(ideal.random_element() * quat for ideal, quat in zip(ideals, quats))

    def intersection(self, other):
        r"""
        Return the intersection of this order with other.

        INPUT:

        - ``other`` -- a quaternion order in the same ambient quaternion algebra

        OUTPUT: a quaternion order
        """
        if not isinstance(other, QuaternionOrder_nf):
            raise TypeError("other must be a QuaternionOrder_nf")

        A = self.quaternion_algebra()
        if other.quaternion_algebra() != A:
            raise ValueError(
                "self and other must be in the same ambient quaternion algebra"
            )

        intersection_pm = self.pseudo_matrix().lattice_intersection(
            other.pseudo_matrix()
        )
        return QuaternionOrder_nf(A, intersection_pm)

    def discriminant(self):
        r"""
        Return the discriminant of this order.

        This is defined as
        `\sqrt{ det ( Tr(e_i \bar{e}_j ) ) }`, where `\{e_i\}` is the
        basis of the order.

        OUTPUT: rational number

        EXAMPLES::

            sage: QuaternionAlgebra(-11,-1).maximal_order().discriminant()
            11
            sage: S = BrandtModule(11, 5).order_of_level_N()
            sage: S.discriminant()
            55
            sage: type(S.discriminant())
            <... 'sage.rings.rational.Rational'>
        """
        ideals, gens = self.pseudo_basis()
        L = [[x.pair(y) for y in gens] for x in gens]
        return matrix(self.base().number_field(), 4, 4, L).determinant().sqrt() * prod(
            ideals
        )

    def is_maximal(self) -> bool:
        r"""
        Check whether the order of ``self`` is maximal in the ambient quaternion algebra.

        Only implemented for quaternion algebras over number fields; for reference,
        see Theorem 15.5.5 in [Voi2021]_.
        """
        return self.discriminant() == self.quaternion_algebra().discriminant()

    def left_ideal(self, gens, check=True, is_basis=False):
        r"""
        Return the left ideal of this order generated by the given generators.

        INPUT:

        - ``gens`` -- list of elements of this quaternion order

        - ``check`` -- boolean (default: ``True``)

        - ``is_basis`` -- boolean (default: ``False``); if ``True`` then
          ``gens`` must be a `\ZZ`-basis of the ideal
        """
        if isinstance(gens, RingElement):
            gens = [gens]
        if not is_basis:
            gens = sum([[b * gen for b in self.integral_basis()] for gen in gens], [])
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(), gens, left_order=self, check=check
        )

    def right_ideal(self, gens, check=True, is_basis=False):
        r"""
        Return the right ideal of this order generated by the given generators.

        INPUT:

        - ``gens`` -- list of elements of this quaternion order

        - ``check`` -- boolean (default: ``True``)

        - ``is_basis`` -- boolean (default: ``False``); if ``True`` then
          ``gens`` must be a `\ZZ`-basis of the ideal
        """
        if isinstance(gens, RingElement):
            gens = [gens]
        if not is_basis:
            gens = sum([[gen * b for b in self.integral_basis()] for gen in gens], [])
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(), gens, right_order=self, check=check
        )

    @cached_method
    def unit_ideal(self):
        r"""
        Return the unit ideal in this quaternion order.

        EXAMPLES::

            sage: R = QuaternionAlgebra(-11,-1).maximal_order()
            sage: I = R.unit_ideal(); I
            Fractional ideal (1/2 + 1/2*i, 1/2*j - 1/2*k, i, -k)
        """
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(),
            self.pseudo_matrix(),
            left_order=self,
            right_order=self,
            check=False,
        )

    @cached_method
    def commutator_ideal(self):
        r"""
        Return the commutator ideal of this order, i.e., the ideal
        generated by elements of the form `\alpha\beta - \beta\alpha`
        where `\alpha,\beta` lie in this order.
        """
        return self.left_ideal(
            [a * b - b * a for i, a in enumerate(self.gens()) for b in self.gens()[i:]]
        )

    def __mul__(self, other):
        r"""
        Every order equals its own unit ideal. Overload ideal multiplication
        and scaling to orders.
        """
        return self.unit_ideal() * other

    def __rmul__(self, other):
        return other * self.unit_ideal()

    def __add__(self, other):
        r"""
        Every order equals its own unit ideal. Overload ideal addition
        to orders.
        """
        return self.unit_ideal() + other

    def quadratic_form(self, gen=None):
        r"""
        Return the normalized quadratic form associated to this quaternion order.

        OUTPUT: quadratic form
        """
        return self.unit_ideal().quadratic_form(gen)

    def conjugate(self, x):
        """
        Return the order obtained by conjugating ``self`` by ``x``.
        """
        return QuaternionOrder_nf(
            self.quaternion_algebra(), [x * b * ~x for b in self.gens()]
        )

    def is_unit(self, u):
        """
        Check whether ``u`` is a unit of ``self``.
        """
        R = self.base_ring()
        return u in self and R(u.reduced_norm()).is_unit()

    def from_integral_coordinates(self, v):
        """
        Recover an element of ``self`` from its coordinates with
        respect to ``self.integral_basis()``.
        """
        return self.unit_ideal().from_integral_coordinates(v)

    def projective_unit_group(self):
        """
        Output the group of units of ``self`` modulo
        the units of the base ring of ``self``.
        """
        K = self.base_ring().number_field()
        L = self.quadratic_form(K(1))
        traces = {K(u).trace() for u in totally_positive_units_up_to_squares(K)}
        bound = max(traces) + 1
        short_vectors = L.short_vector_list_up_to_length(bound)
        units = []
        for trace in traces:
            candidates = [
                self.from_integral_coordinates(v) for v in short_vectors[trace]
            ]
            units += [c for c in candidates if self.is_unit(c)]
        res = []
        for u in units:
            ui = ~u
            if all(not (ui * v).is_constant() for v in res):
                res.append(u)
        return res

    def mass(self):
        r"""
        Return the number of left (or right) ideal classes of ``self``

        WARNING:

        Currently assumes without checking that ``self`` is an Eichler order of level `p`, where `p` is
        the second invariant of ``self.quaternion_algebra()``, and the quaternion algebra is definite and
        has discriminant either `1` or `p`.
        """
        K = self.quaternion_algebra().base_ring()
        p = ZZ(-self.quaternion_algebra().invariants()[1])
        f = K.ideal(p).factor()
        if len(f) == 1:
            phipsi = p + 1
        else:
            phipsi = (p - 1) ** 2
        h = K.class_number()
        zeta = zeta_min_1(K)
        pow_two = ZZ(2) ** (1 - K.degree())
        return pow_two * zeta * h * phipsi

    def two_sided_ideal_classes(self):
        r"""
        Outputs a generator of representatives of the classes of two sided ideals of self.

        WARNING:
        Assumes that self is an Eichler order of square-free discriminant.
        """
        Q = self.quaternion_algebra()
        K = Q.base_ring()
        gens = [self.unit_ideal() * ideal.ideal() for ideal in K.class_group()]
        facto = self.discriminant().factor()
        if any(f[1] != 1 for f in facto):
            raise NotImplementedError(
                "computation only implemented for Eichler orders with squarefree discriminant."
            )
        gens = gens + [
            self.commutator_ideal() + self.unit_ideal() * p for p, _ in facto
        ]
        exponent_vectors = product([0, 1], repeat=len(gens))
        output = []
        for exp_vec in exponent_vectors:
            ideal = prod(
                gen if e == 1 else self.unit_ideal() for gen, e in zip(gens, exp_vec)
            )
            if all(not ideal.is_isomorphic(other)[0] for other in output):
                output.append(ideal)
                yield ideal

    def local_basis(self, prime):
        """
        Output a basis of self localized at ``prime``.
        """
        ideals, quats = self.pseudo_basis()
        return [
            local_generator(ideal, prime) * quat for ideal, quat in zip(ideals, quats)
        ]

    @cached_method
    def local_splitting(self, prime):
        """
        Output a splitting from ``self`` mod ``prime`` to a matrix algebra
        over the residue field of ``prime``.
        """
        if prime.divides(self.discriminant()):
            raise NotImplementedError(
                "Local splittings only implemented away from the discriminant"
            )
        local_basis = self.local_basis(prime)
        local_matrix = matrix([e.coefficient_tuple() for e in local_basis])
        k = prime.residue_field()
        R = self.base_ring()
        tables = [
            (local_matrix * e.matrix() * ~local_matrix).change_ring(k)
            for e in local_basis
        ]
        A = FiniteDimensionalAlgebra(
            k, tables, assume_associative=True, assume_unital=True
        )

        def to_A(x):
            vec = vector(x.coefficient_tuple()) * ~local_matrix
            return A(vec)

        def from_A(x):
            return sum(R(c) * b for c, b in zip(x.vector(), local_basis))

        return SplitAlgebra(A, to_A, from_A)

    def _left_primitive_ideals(self, prime):
        """
        Outputs the list of the primitive left ``self``-ideals of norm ``prime``.
        """
        k = prime.residue_field()
        splitting = self.local_splitting(prime)
        assert splitting.check_isom()
        matrices = [matrix(k, 2, 2, [1, 0, 0, 0])] + [
            matrix(k, 2, 2, [x, 1, 0, 0]) for x in k
        ]
        return [
            self.left_ideal(splitting.from_M(mat)) + self.unit_ideal() * prime
            for mat in matrices
        ]

    def random_primitive_ideal(self, prime):
        """
        Outputs a random primitive left ideal of ``self`` of norm ``prime``.
        """
        k = prime.residue_field()
        splitting = self.local_splitting(prime)
        toss = randint(0, k.cardinality())
        if toss:
            mat = matrix(k, 2, 2, [1, 0, 0, 0])
        else:
            mat = matrix(k, 2, 2, [k.random_element(), 1, 0, 0])
        return self.left_ideal(splitting.from_M(mat)) + self.unit_ideal() * prime

    def random_ideal(self):
        """
        Walk randomly ``length`` steps on the graph of ideals of
        norm supported by the elements of ``prime`` starting from ``self``.
        """
        d = ceil(sqrt(self.discriminant().norm()))
        n = ceil((3 / 4) * log(d, 2) + 5)
        K = self.quaternion_algebra().base_ring()
        ell = (ZZ(2) ** n).next_prime()
        while K.ideal(ell).is_prime():
            ell = ell.next_prime()
        primes = [f[0] for f in K.ideal(ell).factor()]
        ideal = self.unit_ideal()
        order = self
        for prime in primes:
            new_ideal = order.random_primitive_ideal(prime)
            order = new_ideal.right_order()
            ideal = ideal * new_ideal
        return ideal

    def connecting_ideal(self, other):
        """
        Output a left ideal of ``self`` with ``other`` as its right order.
        """
        res = self.unit_ideal() * other.unit_ideal()
        return res * res.norm().denominator()

    def is_conjugated(self, other):
        """
        Checks whether ``self`` and ``other`` are conjugated.
        """
        con = self.connecting_ideal(other)
        return any((ts * con).is_principal() for ts in self.two_sided_ideal_classes())

    def left_ideal_classes(self):
        r"""
        Outputs a generator of representatives of the classes of two sided ideals of self.

        WARNING:
        - Assumes that self is an Eichler order of square-free discriminant.
        """
        target_mass = self.mass()
        new_orders = [(self, self.unit_ideal())]
        ideals = []
        K = self.base_ring().number_field()
        support = narrow_class_group_support(K)
        if support == set():
            support = {K.ideal(2).factor()[0][0]}
        for ideal in self.two_sided_ideal_classes():
            yield ideal
            ideals.append(ideal)
        total_mass = ZZ(len(ideals)) / ZZ(len(self.projective_unit_group()))
        while total_mass < target_mass:
            order, connecting_ideal = new_orders.pop()
            for prime in support:
                neighbours = order._left_primitive_ideals(prime)
                candidates = [connecting_ideal * neighbour for neighbour in neighbours]
                for i, candidate in enumerate(candidates):
                    if all(
                        not candidate.is_isomorphic(ideal, side="left")[0]
                        for ideal in ideals
                    ):
                        new_order = candidate.right_order()
                        new_orders.append((new_order, candidate))
                        two_sided_classes = list(new_order.two_sided_ideal_classes())
                        for suffix in two_sided_classes:
                            ideal = candidate * suffix
                            ideals.append(ideal)
                            yield ideal
                        total_mass += len(two_sided_classes) / len(
                            new_order.projective_unit_group()
                        )

    def is_standard(self):
        r"""
        Check whether `self` is an order of discriminant ``p`` in
        `B_{p, \infty}` which contains the standard quaternion basis.
        """
        Q = self.quaternion_algebra()
        K = Q.base_ring()
        q, p = Q.invariants()
        return (
            q == -1
            and Q.gen(0) in self
            and Q.gen(1) in self
            and self.discriminant() == K.ideal(p)
        )

    def represent_integer(self, M):
        """
        Output an element of ``self`` of norm ``M``.
        """
        if not self.is_standard():
            raise NotImplementedError("Only implemented for standard orders")
        Q = self.quaternion_algebra()
        _, p = Q.invariants()
        _, sol = self.__solver.solve_four_squares(M)
        return Q(sol)

    def strong_approximation(self, N, C, D, ell, exponent):
        """
        Output an element of ``self`` of norm ``ell**exponent``
        or ``ell**(exponent + 1) which is equal to ``C * j + D* k``
        modulo ``N``.
        """
        B = self.quaternion_algebra()
        _, mp = B.invariants()
        R = self.base_ring()
        K = R.number_field()
        basis_matrix = matrix([b.vector() for b in R.basis()])
        p = -mp
        N_ideal = K.ideal(N)
        F = N_ideal.residue_field()

        # Picking parameters lambda and M for the equation
        e = ceil(log(ZZ(p) * N.norm() ** exponent, ell))
        M = ell**e
        lam_sq = ell**e / (p * (C**2 + D**2))
        if N_ideal.residue_symbol(lam_sq, 2) != 1:
            lam_sq *= ell
            M *= ell
        Flam_sq = F(lam_sq)
        Flam = sqrt(Flam_sq)
        lam = small_lift(N_ideal, R(Flam))

        # Building the lattice
        c1 = -2 * p * lam * C
        c2 = -2 * p * lam * D

        target = M - p * lam**2 * (C**2 + D**2)
        assert target in N_ideal
        target = K(target / N)

        c2_inv = c2.inverse_mod(N_ideal)

        initial_solution = N * vector([0, reduce_mod(N_ideal, target * c2_inv)])

        cvp_target = vector([lam * C, lam * D]) - initial_solution

        lattice_basis = [
            [N * b, N * reduce_mod(N_ideal, -b * c1 * c2_inv)] for b in R.basis()
        ] + [[K(0), N**2 * b] for b in R.basis()]

        def integral_vector(vec):
            return vector(
                sum([list(basis_matrix.solve_left(c.vector())) for c in vec], [])
            ).change_ring(ZZ)

        cvp_target_vec = integral_vector(cvp_target)
        lattice_rational_basis = matrix(
            [integral_vector(b) for b in lattice_basis]
        ).LLL()

        space_basis = [[b, 0] for b in R.basis()] + [[0, b] for b in R.basis()]
        space_gram = matrix(
            [
                [(b[0] * c[0] + b[1] * c[1]).trace() for b in space_basis]
                for c in space_basis
            ]
        )

        # Trying different Gram matrices for the ambient space. Leave only one line uncommented.
        lattice = IntegralLattice(space_gram, basis=lattice_rational_basis)
        # lattice = IntegralLattice(identity_matrix(QQ, 4), basis=lattice_rational_basis)

        # Extracting a solution from short lattice vectors
        def check_vector(z, t):
            z_term = lam * C - (initial_solution[0] + z)
            t_term = lam * D - (initial_solution[1] + t)
            S = R((M - p * (z_term**2 + t_term**2)) / N**2)
            if S.is_totally_positive() and S.is_prime():
                done, lift = self.__solver.solve_two_squares(S)
                if done:
                    x, y = lift
                    return True, B([N * x, N * y, z_term, t_term])
            return False, None

        def reconstruct_vector(v):
            return K(v[0:2] * basis_matrix), K(v[2:4] * basis_matrix)

        count = 1
        for close_vector in lattice.enumerate_close_vectors(cvp_target_vec):
            if count == 2000:
                print(
                    "Strong approximation is struggling to find a solution. It would be best to restart the computation with a larger exponent."
                )
            z, t = reconstruct_vector(close_vector)
            done, sol = check_vector(z, t)
            if done:
                return sol
        raise ValueError("Exhausted close vectors before a solution was found.")
