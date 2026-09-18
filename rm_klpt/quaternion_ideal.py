# ***********************************************************************
# Copyright (C) 2009 William Stein <wstein@gmail.com>
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

from nf_utility import (
    is_ideal_narrow_principal,
    totally_positive_units_up_to_squares,
    ideal_sqrt,
    quat_mod_I,
    small_lift,
)
from hnf import PseudoMatrix

from sage.misc.cachefunc import cached_method
from sage.algebras.quatalg.quaternion_algebra import QuaternionFractionalIdeal

from sage.rings.integer_ring import ZZ
from sage.rings.ideal import Ideal_fractional
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.functions.other import ceil
from sage.misc.functional import log
from sage.quadratic_forms.quadratic_form import QuadraticForm
from sage.matrix.special import block_matrix
from sage.rings.number_field.number_field_element import NumberFieldElement
from sage.rings.number_field.number_field_ideal import NumberFieldIdeal


class QuaternionFractionalIdeal_nf(QuaternionFractionalIdeal):
    r"""
    A fractional ideal in a quaternion algebra over a number field.

    INPUT:

    - ``left_order`` -- a quaternion order or ``None``

    - ``right_order`` -- a quaternion order or ``None``

    - ``Q`` -- a quaternion algebra over a number field.

    - ``gens`` -- tuple or vectors of elements of  ``Q`` whose
      `\mathcal{O}_K`-span is an ideal

    - ``check`` -- boolean (default: ``True``); if ``False``, do no type
      checking.
    """

    def __init__(self, Q, gens, left_order=None, right_order=None, check=True):
        K = Q.base_ring()
        self.__base_ring = K.maximal_order()
        if check:
            from quaternion_order import QuaternionOrder_nf

            if left_order is not None and not isinstance(
                left_order, QuaternionOrder_nf
            ):
                raise TypeError("left_order must be a quaternion order or None")
            if right_order is not None and not isinstance(
                right_order, QuaternionOrder_nf
            ):
                raise TypeError("right_order must be a quaternion order or None")
            if not isinstance(gens, (list, tuple, PseudoMatrix)):
                raise TypeError("gens must be a list or tuple or a PseudoMatrix")
        if not isinstance(gens, PseudoMatrix):
            gens = PseudoMatrix(
                [K.ideal(1) for _ in gens],
                matrix([gen.coefficient_tuple() for gen in gens]),
            )
        self.__pseudo_matrix, _ = gens.pseudo_hermite_form()
        integral_basis = sum(
            [
                [b * Q(list(row)) for b in ideal.basis()]
                for ideal, row in zip(
                    self.__pseudo_matrix.ideals(), self.__pseudo_matrix.matrix().rows()
                )
            ],
            [],
        )
        self.__left_order = left_order
        self.__right_order = right_order
        Ideal_fractional.__init__(self, Q, integral_basis)

    def __add__(self, other):
        mat = block_matrix(
            [[self.pseudo_matrix().matrix()], [other.pseudo_matrix().matrix()]]
        )
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(),
            PseudoMatrix(
                self.pseudo_matrix().ideals() + other.pseudo_matrix().ideals(), mat
            ),
        )

    def __mul__(self, other):
        if isinstance(other, QuaternionFractionalIdeal_nf):
            ideals_left, quats_left = self.pseudo_basis()
            ideals_right, quats_right = other.pseudo_basis()
            ideals = [left * right for left in ideals_left for right in ideals_right]
            mat = matrix(
                [
                    (left * right).coefficient_tuple()
                    for left in quats_left
                    for right in quats_right
                ]
            )
        elif isinstance(other, NumberFieldIdeal):
            pm = self.pseudo_matrix()
            ideals = [other * ideal for ideal in pm.ideals()]
            mat = pm.matrix()
        elif isinstance(other, NumberFieldElement):
            return self * other.parent().ideal(other)
        elif other in self.quaternion_algebra():
            return self * self.right_order().left_ideal(other)
        else:
            raise NotImplementedError(
                f"Cannot multiply QuaternionFractionalIdeal_nf with {type(other)}."
            )
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(), PseudoMatrix(ideals, mat)
        )

    def __rmul__(self, other):
        if isinstance(other, QuaternionFractionalIdeal_nf):
            return other.__mul__(self)
        elif isinstance(other, NumberFieldIdeal):
            return self * other
        elif isinstance(other, NumberFieldElement):
            return self * other.parent().ideal(other)
        elif other in self.quaternion_algebra():
            return self.left_order().right_ideal(other) * self
        else:
            raise NotImplementedError(
                f"Cannot multiply QuaternionFractionalIdeal_nf with {type(other)}."
            )

    def __invert__(self):
        return self.conjugate() * ~self.norm()

    def intersection(self, other):
        """
        Output the intersection of ``self`` and ``other``.
        """
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(),
            self.pseudo_matrix().lattice_intersection(other.pseudo_matrix()),
        )

    def colon(self, other, side="left"):
        """
        Comute the colon ``[self:other]_L`` or
        ``[self:other]_R``, depending on the value of ``side``.
        """
        side = _invert_side(side)
        if side == "left":
            ids = [g * self for g in other.gens()]
        else:
            ids = [self * g for g in other.gens()]
        res = ids[0]
        for id in ids[1:]:
            res = res.intersection(id)
        return res

    def quaternion_algebra(self):
        """
        Return the ambient quaternion algebra that contains this fractional
        ideal.

        This is an alias for `self.ring()`.
        """
        return self.ring()

    def base_ring(self):
        """
        Return the maximal order of the base field of the ambient algebra of
        ``self``.

        Equivalent to ``self.quaternion_algebra().base_ring().maximal_order()``
        """
        return self.__base_ring

    def _compute_order(self, side="left"):
        r"""
        Used internally to compute either the left or right order
        associated to an ideal in a quaternion algebra.

        INPUT:

        - ``side`` -- ``'left'`` or ``'right'``

        OUTPUT: the left order if ``side='left'``; the right order if
        ``side='right'``

        ALGORITHM: Let `b_1, b_2, b_3, b_3` be an integral basis for this
        fractional ideal `I`, and assume we want to compute the left
        order of `I` in the quaternion algebra `Q`.  Then
        multiplication by `b_i` on the right defines a map `B_i:Q \to
        Q`.  We have

        .. MATH::

           R = B_1^{-1}(I) \cap B_2^{-1}(I) \cap B_3^{-1}(I)\cap B_4^{-1}(I).

        This is because

        .. MATH::

           B_n^{-1}(I) = \{\alpha \in Q : \alpha b_n \in I \},

        and

        .. MATH::

           R = \{\alpha \in Q : \alpha b_n \in I, n=1,2,3,4\}.
        """
        if side == "left":
            action = "right"
        elif side == "right":
            action = "left"
        else:
            raise ValueError("side must be 'left' or 'right'")
        Q = self.quaternion_algebra()

        M = [(~b).matrix(action=action) for b in self.integral_basis()]
        pm = self.pseudo_matrix()
        invs = [
            PseudoMatrix(pm.ideals(), pm.matrix() * m).pseudo_hermite_form()[0]
            for m in M
        ]
        res = invs[0]
        for inv in invs[1:]:
            res = res.lattice_intersection(inv)
        from quaternion_order import QuaternionOrder_nf

        return QuaternionOrder_nf(Q, res)

    def left_order(self):
        """
        Return the left order associated to this fractional ideal.

        OUTPUT: an order in a quaternion algebra
        """
        if self.__left_order is None:
            self.__left_order = self._compute_order(side="left")
        return self.__left_order

    def right_order(self):
        """
        Return the right order associated to this fractional ideal.

        OUTPUT: an order in a quaternion algebra
        """
        if self.__right_order is None:
            self.__right_order = self._compute_order(side="right")
        return self.__right_order

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, QuaternionFractionalIdeal_nf)
            and self.quaternion_algebra() == other.quaternion_algebra()
            and all(b in other for b in self.integral_basis())
            and all(b in self for b in other.integral_basis())
        )

    def __repr__(self) -> str:
        """
        Return string representation of this quaternion fractional ideal.
        """
        return f"Fractional ideal {self.pseudo_matrix()}"

    def random_element(self, *args, **kwds):
        """
        Return a random element in the rational fractional ideal ``self``.

        The ``args`` and ``kwds`` are passed to the ``random_element`` method
        of the base ring.
        """
        return sum(
            self.__base_ring.random_element(*args, **kwds) * g for g in self.gens()
        )

    def integral_basis(self):
        """
        Return a basis for this fractional ideal.

        OUTPUT: tuple
        """
        return self.gens()

    def pseudo_basis(self):
        """
        Return a pseudo-basis for this fractional ideal.
        """
        Q = self.quaternion_algebra()
        pb = [Q(list(r)) for r in self.__pseudo_matrix.matrix().rows()]
        return self.__pseudo_matrix.ideals(), pb

    def __hash__(self) -> int:
        """
        Return the hash of ``self``.
        """
        return hash(self.gens())

    def pseudo_matrix(self):
        """
        Return the pseudo matrix of ``self``
        """
        return self.__pseudo_matrix

    def from_integral_coordinates(self, v):
        """
        Return an element of ``self`` from its coordinates with
        respect to ``self.integral_basis()``.
        """
        return sum(e * b for e, b in zip(list(v), self.integral_basis()))

    def integral_gram_matrix(self, scaling=1):
        r"""
        Return the gram matrix of self as a `\ZZ`-module with respect
        to the quadratic form defined as trace of reduced norm.

        OUTPUT: matrix over `\ZZ`
        """
        basis = self.integral_basis()
        return matrix(
            ZZ, [[(a.pair(b) * scaling).trace() for b in basis] for a in basis]
        )

    @cached_method
    def reduced_integral_basis(self):
        r"""
        Return a basis of the ideal as a `\ZZ`-module, which is LLL-reduced
        with respect to the quadratic form defined as the trace of the
        reduced norm

        OUTPUT: tuple
        """
        if not self.quaternion_algebra().is_totally_definite():
            raise TypeError("the quaternion algebra must be totally definite")

        U = self.integral_gram_matrix().LLL_gram().transpose()
        return tuple(
            sum(c * g for c, g in zip(row, self.integral_basis())) for row in U
        )

    def norm(self):
        """
        Return the reduced norm of this fractional ideal.

        OUTPUT: ideal of the base field of the ambient algebra
        """
        R = self.__left_order or self.__right_order or self.left_order()
        R_pm = R.pseudo_matrix()
        basis_wrt_order = R_pm.matrix().solve_left(self.pseudo_matrix().matrix())
        res = (
            basis_wrt_order.det()
            * prod(self.pseudo_matrix().ideals())
            * ~prod(R_pm.ideals())
        )
        return ideal_sqrt(res)

    def quadratic_form(self, gen=None):
        """
        Return the norm form of self, assuming that ``self.norm()``
        is principal.
        """
        K = self.base_ring().number_field()
        if gen is None:
            check, gen = is_ideal_narrow_principal(self.norm())
            if not check:
                raise ValueError("The ideal must have narrow principal norm")
        elif not K.ideal(gen) == self.norm():
            raise ValueError("Invalid norm ideal generator.")

        return QuadraticForm(self.integral_gram_matrix(scaling=~gen))

    def is_principal(self):
        """
        Check whether ``self`` is principal and output
        a generator if it is.
        """
        check, gen = is_ideal_narrow_principal(self.norm())
        if not check:
            return False, None
        K = self.quaternion_algebra().base_ring()
        candidates = totally_positive_units_up_to_squares(K)

        for c in candidates:
            L = self.quadratic_form(gen * c)
            short_vectors = L.short_vector_list_up_to_length(K.degree() + 1, True)
            if not short_vectors[-1] == []:
                gen = short_vectors[-1][0]
                return True, sum(
                    c * e for c, e in zip(list(gen), self.integral_basis())
                )
        return False, None

    def conjugate(self):
        """
        Output the conjugate ideal of ``self``.
        """
        ids, pb = self.pseudo_basis()
        mat = matrix([b.conjugate().coefficient_tuple() for b in pb])
        return QuaternionFractionalIdeal_nf(
            self.quaternion_algebra(),
            PseudoMatrix(ids, mat),
            left_order=self.__right_order,
            right_order=self.__left_order,
        )

    def is_isomorphic(self, other, side="left"):
        """
        Check whether ``self`` is isomorphic to ``other`` as
        ``side``-ideals.
        """
        if side == "right":
            if not self.right_order() == other.right_order():
                return False, None
            test_ideal = self * ~other
        else:
            if not self.left_order() == other.left_order():
                return False, None
            test_ideal = ~self * other
        check, gen = test_ideal.is_principal()
        if check:
            return True, gen
        return False, None

    def __contains__(self, x):
        """
        Checks if x is an element of self.
        """
        if x not in self.quaternion_algebra():
            return False
        return self.pseudo_matrix().lattice_contains(x.coefficient_tuple())

    def equivalent_ideal_prime_norm(self, residue_condition=[], excluded=[]):
        r"""
        Return a fractional ideal of prime norm with the same right order as
        ``self``, such that ell is not a square modulo the output.

        If left is True, the left order is preserved instead of the right.
        order.

        INPUT:

        -ell: integer
        -left: boolean

        OUTPUT

        - The right-equivalent fractional ideal of prime norm
        - The norm of the new ideal, as a totally positive number
        - The element used to generate the output ideal by rescaling.
        """
        K = self.quaternion_algebra().base_ring()
        correction = (~K.class_group()(self.norm())).ideal()
        basis = (self * correction).reduced_integral_basis()
        N = self.norm() * correction**2
        K = self.ring().base_ring()
        m = 1
        for coords in product(range(-m, m + 1), repeat=8):
            delta = sum(c * b for c, b in zip(coords, basis))
            dnorm = delta.reduced_norm() / N
            if (
                dnorm.is_prime()
                and all(
                    ell not in dnorm and dnorm.residue_symbol(ell, 2) != 1
                    for ell in residue_condition
                )
                and all(dnorm.is_coprime(e) for e in excluded)
            ):
                break
        res = (self * delta.conjugate()) * ~(self.norm() * correction)
        return res, dnorm, delta

    def ideal_mod_constraint(self, gamma):
        r"""
        Return ``C, D`` in `\ZZ_K` such that ``gamma (j * C + k * D) in I``,
        where ``j`` and ``k`` are elements of the standard basis of
        ``self.ring()``.

        Input:
        - quaternion
        """
        N = self.norm()
        _, j, k = self.quaternion_algebra().gens()
        mat = matrix([quat_mod_I(N, e).list() for e in self.integral_basis()])
        system = (
            matrix([quat_mod_I(N, gamma * x).list() for x in [j, k]])
            * mat.right_kernel_matrix().transpose()
        )
        C, D = system.kernel().basis()[0]
        return small_lift(N, C), small_lift(N, D)

    def klpt(self, ell, check_order=False, strong_approximation_exponent=1.51):
        """
        Compute a left-equivalent ideal of norm a power of ell.

        Requires that one of the orders of ``self`` be standard.
        """
        if check_order and not self.left_order().is_standard():
            if self.right_order().is_standard():
                J, factor = self.conjugate().klpt(ell)
                return J.conjugate(), factor.conjugate()
            else:
                raise NotImplementedError(
                    "One of the orders of self needs to be standard."
                )
        _, p = self.quaternion_algebra().invariants()
        p = ZZ(-p)
        JK, N, eta = self.equivalent_ideal_prime_norm(
            [ell], excluded=[t[0] for t in self.right_order().discriminant().factor()]
        )
        _, N = is_ideal_narrow_principal(N)
        e0 = max(
            [
                0,
                ceil(
                    log(p, ell)
                    + log(ceil(log(p) ** 2), ell) / 2
                    - log(N.norm(), ell) / 2
                ),
            ]
        )
        M = N * ell**e0
        gamma = self.left_order().represent_integer(M)
        C, D = JK.ideal_mod_constraint(gamma)
        nu = self.right_order().strong_approximation(
            N, C, D, ell, exponent=strong_approximation_exponent
        )
        _, N_self = is_ideal_narrow_principal(self.norm())
        factor = ~eta * nu.conjugate() * gamma.conjugate()
        return JK * (nu.conjugate() * gamma.conjugate() / N), factor


def _invert_side(side):
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError("side should be 'left' or 'right'")
