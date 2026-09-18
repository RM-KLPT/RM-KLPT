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

from sage.algebras.quatalg.quaternion_algebra import QuaternionAlgebra
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix
from sage.rings.number_field.number_field import NumberField
from sage.misc.cachefunc import cached_method

from quaternion_matrix import (
    QuaternionMatrix,
    lattice_subspace_intersection,
    principalize_quaternion_matrix_ideal,
    ideal_embedding,
)
from quaternion_ideal import QuaternionFractionalIdeal_nf
from quaternion_order import QuaternionOrder_nf
from nf_utility import full_sign_coverage


class RM_Context:
    r"""
    Endomorphism ring data for a real multiplication

    INPUT:

    - a: An integral 'n \times n' matrix over a rational quaternion algebra `B` with prime
        discriminant, whose minimal polynomial is irreducible, has degree `n`, and
        generates a totally real number field.
    - O0: A maximal order of `B` which contains the standard quaternion basis
        of `B` and such that ``a`` lies in `M_n(O0)`.
    """

    def __init__(self, a, O0=None):
        self.B = a.base_ring()
        n = a.nrows()
        pol = a.minimal_polynomial()
        if not pol.is_irreducible() or not pol.degree() == n:
            raise ValueError(
                "a must have an irreducible minimal polynomial of degree n"
            )
        self.K = NumberField(pol, "aK")
        mq, mp = self.B.invariants()
        self.p = ZZ(-mp)
        self.BK = QuaternionAlgebra(self.K, mq, mp)
        self.a = a

        if O0 is None:
            O0 = self.B.maximal_order()
        self.O0 = O0

        self.BK_matrix_K_basis = a.centralizer()
        pows = [a.identity_matrix()]
        for i in range(1, n):
            pows.append(a * pows[-1])
        self.K_matrix_basis = pows

        if not all(
            self.K_to_matrix(b).is_integral(O0)
            for b in self.K.ring_of_integers().basis()
        ):
            raise ValueError("a is not an RM with respect to O0")

        self.BK_matrix_Q_basis = [
            b * alpha for b in self.BK_matrix_K_basis for alpha in self.K_matrix_basis
        ]
        m_2_O_basis = a.matrix_space_order_Z_basis(O0)
        m_2_O_cap_BK = lattice_subspace_intersection(
            m_2_O_basis, self.BK_matrix_Q_basis
        )

        BK_order_gen = [self.matrix_to_BK(x) for x in m_2_O_cap_BK]
        self.O_BK = QuaternionOrder_nf(self.BK, BK_order_gen)

    def __eq__(self, other):
        return self.a == other.a and self.O0 == other.O0

    def matrix_to_BK(self, mat):
        """
        Convert a matrix that commutes with ``self.a`` into a quaternion in
        ``self.BK``.

        INPUT:

        -mat: QuaternionMatrix
        """
        v = mat.coordinates_in_rational_subspace(self.BK_matrix_Q_basis)
        K_coordinates = [self.K(v) for v in batched(v, n=self.K.degree())]
        return self.BK(K_coordinates)

    def matrix_to_K(self, mat):
        """
        Convert a matrix lying in `span(I_2, self.a)` into an element of
        ``self.K``.

        INPUT:

        -mat: QuaternionMatrix
        """
        v = mat.coordinates_in_rational_subspace(self.K_matrix_basis)
        return self.K(v)

    def K_to_matrix(self, x):
        """
        Convert an element of ``self.K`` into a matrix lying in
        `span(I_2, self.a)`.
        """
        x = self.K(x)
        return sum(c * e for c, e in zip(x.list(), self.K_matrix_basis))

    def BK_to_matrix(self, x):
        """
        Convert an element of ``self.BK`` into a matrix that commutes with
        ``self.a``.
        """
        return sum(
            [
                self.K_to_matrix(c) * mat
                for c, mat in zip(x.coefficient_tuple(), self.BK_matrix_K_basis)
            ],
            self.a.new_matrix(),
        )

    def hom(self, other):
        """
        Return an integral basis of the hom module from ``self`` to ``other``. That is, set of matrices `M` in `M_n(self.O0)`
        that `other.a * M = M * self.a`.
        """
        if self.O0 != other.O0:
            raise NotImplementedError(
                "Homomorphisms currently only supported if self and other are integral with respect to the same quaternion order."
            )
        idem = self.a.identity_matrix()
        system = matrix(
            [
                b.apply_tensor_endomorphism(
                    [(other.a, idem), (idem, -self.a)]
                ).rational_coefficients()
                for b in self.a.rational_space_basis()
            ]
        )
        B = self.a.base_ring()
        subspace_basis = [
            QuaternionMatrix(B, self.a.nrows(), self.a.ncols(), list(e))
            for e in system.left_kernel().basis()
        ]
        lattice_basis = self.a.matrix_space_order_Z_basis(self.O0)
        hom_basis = lattice_subspace_intersection(lattice_basis, subspace_basis)
        gram_matrix = matrix(
            [
                [
                    sum(left.pair(right) for left, right in zip(a.list(), b.list()))
                    for a in hom_basis
                ]
                for b in hom_basis
            ]
        )
        U = gram_matrix.LLL_gram()
        reduced_basis = [
            sum(u * e for u, e in zip(r, hom_basis)) for r in U.transpose().rows()
        ]
        return reduced_basis

    def codomain(self, gamma):
        """
        Return the codomain of a homomorphism of RM contexts.
        """
        return RM_Context(gamma * self.a * ~gamma, self.O0)

    def domain(self, gamma):
        """
        Return the codomain of a homomorphism of RM contexts.
        """
        return RM_Context(~gamma * self.a * gamma, self.O0)

    def ideal(self, gamma):
        """
        Computes the OK-basis of the intersection between M_2(O) * gamma and
        Cent(a) embedded as an ideal in B ⊗ K.

        INPUT:

        -gamma: QuaternionMatrix
        """
        other = self.codomain(gamma)
        ideal_basis = [b * gamma for b in other.hom(self)]
        return QuaternionFractionalIdeal_nf(
            self.BK, [self.matrix_to_BK(b) for b in ideal_basis], left_order=self.O_BK
        )

    def lift_ideal(self, ideal):
        """
        Output a homomorphism of real multiplications which
        induced ``ideal`` when ``self`` is the codomain.
        """
        p_ideal = self.K.ideal(self.p)
        if not ideal.norm().is_coprime(p_ideal):
            ideal, _, _ = ideal.equivalent_ideal_prime_norm(
                excluded=[prime for prime, _ in p_ideal.factor()]
            )
        gens = [
            left * self.BK_to_matrix(right)
            for right in ideal.integral_basis()
            for left in self.a.matrix_space_order_Z_basis(self.O0)
        ]
        return principalize_quaternion_matrix_ideal(self.O0, gens)

    def isomorphy_classes(self):
        """
        Output a generators on the isomorphy classes of real multiplications
        with the same parameters as ``self``.
        """
        for ideal in self.O_BK.left_ideal_classes():
            gamma = self.lift_ideal(ideal)
            yield self.codomain(gamma), gamma

    def base_ring(self):
        """
        Output the base quaternion algebra of ``self``.
        """
        return self.a.base_ring()

    def rm_algebra(self):
        """
        Output the endomorphism algebra of ``self``
        as a quaternion algebra.
        """
        return self.BK

    def endomorphism_order(self):
        return self.O_BK

    def K_degree(self, gamma):
        """
        Compute the K-degree of RM-isogeny ``gamma`` from ``self``.

        INPUT:
        -gamma: QuaternionMatrix
        """
        return self.ideal(gamma).norm()

    def some_hom(self, other):
        """
        Return some arbitrary element of self.hom(other)

        INPUT:

        -other: RM_Context
        """
        return self.hom(other)[0]

    def is_isomorphic(self, other):
        """
        Check whether ``self`` is isomorphic to ``other``.
        """
        delta = self.some_hom(other)
        ideal = self.ideal(delta)
        return ideal.is_principal()[0]

    def conjugate(self):
        """
        Output the real multiplication conjugate to ``self``
        """
        return RM_Context(self.a.conjugate_transpose(), self.O0)

    @cached_method
    def polarisation_module(self):
        """
        Output a pseudo-basis of the polarisation module of ``self``.
        """
        symmetric_space_basis = [
            self.a.elementary_matrix(i, i, 1) for i in range(self.a.nrows())
        ] + [
            self.a.elementary_matrix(i, j, c)
            + self.a.elementary_matrix(j, i, c.conjugate())
            for i in range(self.a.nrows())
            for j in range(i + 1, self.a.nrows())
            for c in self.B.basis()
        ]
        idem = self.a.identity_matrix()
        other = self.conjugate()
        system = matrix(
            [
                b.apply_tensor_endomorphism(
                    [(other.a, idem), (idem, -self.a)]
                ).rational_coefficients()
                for b in symmetric_space_basis
            ]
        )
        subspace_basis = [
            sum([c * b for c, b in zip(e, symmetric_space_basis)], self.a.new_matrix())
            for e in system.left_kernel().basis()
        ]
        lattice_basis = self.a.matrix_space_order_Z_basis(self.O0)
        integral_basis = lattice_subspace_intersection(lattice_basis, subspace_basis)
        elem = integral_basis[0]
        gen = elem
        for u in full_sign_coverage(self.K):
            if gen.is_positive_definite():
                break
            gen = elem * self.K_to_matrix(u)
        assert gen.is_positive_definite()
        ideal_basis = [self.matrix_to_K(~gen * b) for b in integral_basis]
        return gen, self.K.ideal(ideal_basis)


def ideal_RM(ideal, O0):
    B = O0.quaternion_algebra()
    a = ideal_embedding(B, ideal)
    return RM_Context(a, O0)


def standard_RM(K, O0):
    r"""
    Output a real multiplication with endomorphism ring
    `O_0 \otimes \mathbb{Z}_K`
    """
    return ideal_RM(K.ideal(1), O0)
