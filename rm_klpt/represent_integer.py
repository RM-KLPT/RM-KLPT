# ***********************************************************************
# Copyright (C) 2026 Mingjie Chen, Mickaël Montessinos
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
Skeleton for representing elements by

    x^2 + y^2 + p(z^2 + w^2) = M

over a real quadratic field ``K = Q(sqrt(d))`` with ``d > 0`` square-free.
"""

from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring import polygen
from sage.functions.other import ceil
from sage.functions.other import floor
from sage.misc.functional import log
from sage.misc.prandom import randint
from cypari2.handle_error import PariError


class RepresentIntegerSolver:
    """
    Skeleton solver for

        x^2 + y^2 + p(z^2 + w^2) = M.

    INPUT:
    -d: a positive squarefree integer
    """

    def __init__(self, K, p):
        self.p = p
        self.d = ZZ(K.gen() ** 2)
        self.K = K
        self.threshold = ceil(log(p) ** 2) * p**2
        self.OK = self.K.maximal_order()
        if self.d % 4 == 1:
            self.omega = (1 + self.K.gen()) / 2
        else:
            self.omega = self.K.gen()
        self.embeddings = self.K.real_embeddings()
        t = polygen(self.K)
        self.norm_field = self.K.extension(t**2 + 1, name="i")

    def solve_two_squares(self, T):
        """
        Try to solve ``a^2 + b^2 = T``.

        INPUT:

        -T: A totally positive element of ``self.OK``

        OUTPUT:
        -``True`` if there is a solution, ``False`` otherwise
        -``(self.OK, self.OK)`` if there is a solution, ``None`` otherwise.
        """
        T = self.K(T)
        if not self.admissible_prime_Z(T):
            return False, None
        try:
            is_norm, sol = T.is_norm(self.norm_field, element=True)
        except PariError:
            raise ValueError(
                f"We were trying to solve the norm equation with target {T}."
            )
        if not is_norm:
            return False, None
        x, y = sol.list()
        return True, (x, y)

    def admissible_prime_Z(self, Z):
        """
        Check whether ``Z`` is prime and satisfies the quadratic residue
        condition for being a sum of two squares.

        INPUT:
        -``Z``: element of ``self.OK``
        """
        Z = self.K(Z)
        if not Z.is_totally_positive():
            return False
        if not Z.is_prime():
            return False
        try:
            return self.K.ideal(Z).residue_symbol(-1, 2) == 1
        except ValueError:
            return False

    @staticmethod
    def _strict_integer_interval(lower, upper):
        """
        Return the integers ``n`` satisfying ``lower < n < upper``.

        INPUT:
        -``lower, upper``: real numbers in increasing order
        """
        return floor(lower) + 1, ceil(upper) - 1

    def Z_constraints(self, M):
        """
        Return the parallelogram in ``(z1, z2)``-space defined by

            Z = z1 + z2 * omega,

        together with the constraints that both ``Z`` and ``M - pZ`` are
        totally positive.

        INPUT:
        -M: A totally positive element of ``self.OK``
        """
        M = self.K(M)
        assert M.is_totally_positive()

        sigma1, sigma2 = self.embeddings
        omega1 = sigma1(self.omega)
        omega2 = sigma2(self.omega)
        bound1 = sigma1(M) / self.p
        bound2 = sigma2(M) / self.p

        # The four boundary lines are:
        # z1 + omega1 * z2 = 0,
        # z1 + omega1 * z2 = M1 / p,
        # z1 + omega2 * z2 = 0,
        # z1 + omega2 * z2 = M2 / p.
        vertices = [
            (0, 0),
            (-omega2 * bound1 / (omega1 - omega2), bound1 / (omega1 - omega2)),
            (
                (omega1 * bound2 - omega2 * bound1) / (omega1 - omega2),
                (bound1 - bound2) / (omega1 - omega2),
            ),
            (omega1 * bound2 / (omega1 - omega2), -bound2 / (omega1 - omega2)),
        ]

        return vertices

    def sample_Z(self, M):
        """
        Yield admissible ``Z`` by rejection sampling from the smallest
        axis-aligned rectangle containing the admissible parallelogram.
        """
        vertices = self.Z_constraints(M)
        z1_values = [vertex[0] for vertex in vertices]
        z2_values = [vertex[1] for vertex in vertices]
        z1_min = ceil(min(z1_values))
        z1_max = floor(max(z1_values))
        z2_min = ceil(min(z2_values))
        z2_max = floor(max(z2_values))
        if z1_min > z1_max or z2_min > z2_max:
            return

        while True:
            z1 = randint(z1_min, z1_max)
            z2 = randint(z2_min, z2_max)
            Z = self.K(z1) + self.K(z2) * self.omega
            if not self.K(Z).is_totally_positive():
                continue
            if not self.K(self.K(M) - self.p * Z).is_totally_positive():
                continue
            yield Z

    def solve_four_squares(self, M, verbose=False):
        """
        Try to solve ``x^2 + y^2 + p(z^2 + w^2) = M``, where ``p`` is a
        rational prime.

        INPUT:

        -M: Totally positive element of ``self.OK``
        -verbose: Boolean
        """
        M = self.K(M)
        if M.norm() < self.threshold:
            raise ValueError(
                f"Norm of M is {M.norm()}, which is below the threshold of {self.threshold}"
            )
        seen_Z = set()
        tested_Z_count = 0

        for Z in self.sample_Z(M):
            if Z in seen_Z:
                continue
            seen_Z.add(Z)
            tested_Z_count += 1

            has_zw, zw = self.solve_two_squares(Z)
            if not has_zw:
                continue

            rhs = M - self.p * Z

            has_xy, xy = self.solve_two_squares(rhs)
            if has_xy:
                if verbose:
                    print(f"success: Z = {Z}, rhs = {rhs}")
                    print(f"tested_Z_count = {tested_Z_count}, ")
                x, y = xy
                z, w = zw
                assert x**2 + y**2 + self.p * (z**2 + w**2) == M
                return True, (x, y, z, w)

        if verbose:
            print(f"tested_Z_count = {tested_Z_count}, ")

        return False, None
