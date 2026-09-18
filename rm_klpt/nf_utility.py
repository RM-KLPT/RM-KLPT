from itertools import product, chain, combinations

from sage.misc.cachefunc import cached_function
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.matrix.special import identity_matrix
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.misc.functional import round
from sage.arith.functions import lcm
from sage.functions.other import ceil
from sage.rings.number_field.unit_group import UnitGroup
from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.rings.finite_rings.finite_field_constructor import FiniteField
from sage.misc.misc_c import prod
from sage.misc.functional import log, sqrt
from sage.rings.qqbar import AA

from sage.modules.free_quadratic_module_integer_symmetric import IntegralLattice


def ideal_sqrt(ideal):
    """
    Computes the square-root of an ideal, provided that it exists.

    WARNING:
    Uses factorisation, slow for very large ideals.

    TODO:
    Is there a better algorithm?
    """
    factors = ideal.factor()
    if not all(ZZ(2).divides(f[1]) for f in factors):
        raise ValueError("ideal is not a square")
    return prod([f[0] ** (f[1] // 2) for f in factors], ideal.number_field().ideal(1))


def field_units_sign_matrix(K):
    """
    Output the matrix of the "sign" map from the group of ideals
    to the product of ``K.degree()`` copies of ``GF(2)``.
    """
    assert K.is_totally_real()
    embs = K.embeddings(AA)
    units = [K(-1)] + UnitGroup(K).fundamental_units()
    return matrix(
        FiniteField(2), [[1 if emb(u) < 0 else 0 for emb in embs] for u in units]
    )


def totally_positive_units_up_to_squares(K):
    """
    Output a list of generators of the group of totally positive
    units of ``K`` factored by the subgroup of square units.
    """
    mat = field_units_sign_matrix(K)
    basis = [-1] + UnitGroup(K).fundamental_units()

    tot_pos_basis = [
        prod(u ** ZZ(e) for u, e in zip(basis, v)) for v in mat.left_kernel().basis()
    ]
    exponent_vectors = product([0, 1], repeat=len(tot_pos_basis))
    return [
        prod(u ** ZZ(e) for u, e in zip(tot_pos_basis, exponent_vector))
        for exponent_vector in exponent_vectors
    ]


def units_up_to_totally_positive(K):
    """
    Outputs a list of generators of the group of units of ``K``
    factored by the subgroup of totally positive units.
    """
    mat = field_units_sign_matrix(K)
    basis = [-1] + UnitGroup(K).fundamental_units()

    not_tot_pos_basis = [
        prod(u ** ZZ(e) for u, e in zip(basis, v))
        for v in mat.left_kernel().complement().basis()
    ]
    exponent_vectors = product([0, 1], repeat=len(not_tot_pos_basis))
    return [
        prod(u ** ZZ(e) for u, e in zip(not_tot_pos_basis, exponent_vector))
        for exponent_vector in exponent_vectors
    ]


def is_ideal_narrow_principal(ideal):
    """
    Check that an ideal admits a totally positive generator,
    and output the generator if it exists.

    OUTPUT:

    - a boolean
    - None or a field element.
    """
    if not ideal.is_principal():
        return False, None
    g = ideal.gens_reduced()[0]
    K = ideal.number_field()
    F = FiniteField(2)
    sign_vector = vector(F, [1 if emb(g) < 0 else 0 for emb in K.embeddings(AA)])
    try:
        sign_factor_vec = (
            field_units_sign_matrix(K).solve_left(sign_vector).change_ring(ZZ)
        )
    except ValueError:
        return False, None
    sign_factor = (-1) ** sign_factor_vec[0] * prod(
        u**e
        for u, e in zip(UnitGroup(K).fundamental_units(), list(sign_factor_vec)[1:])
    )
    return True, sign_factor * g


def zeta_min_1(K):
    r"""
    Computes the exact value of `\zeta_K(-1)` as a rational number.
    """
    n = K.degree()
    p_candidates = [
        p
        for p in range(1, 2 * K.degree() + 2)
        if ZZ(p).is_prime() and ZZ(p - 1).divides(2 * n)
    ]
    q_candidates = [(p ** ((2 * n).valuation(p) + 1), p) for p in p_candidates]
    q_values = [2]
    pol = PolynomialRing(K, "x")
    for q, p in q_candidates:
        while q >= p:
            facto = pol.cyclotomic_polynomial(2 * q).factor()
            if any(P.degree() == 2 for P, _ in facto):
                q_values.append(q)
                break
            q //= p
    Q = lcm(q_values)
    return round((K.zeta_function(prec=ceil(log(Q, 2) + 2))(-1) * Q).real()) / Q


@cached_function
def quat_mod_I_basis(B, ideal):
    r"""
    Return matrices over a finite field which correspond to elements of the
    basis of ``B`` modulo ``ideal``.

    INPUT:
    -B: A quaternion algebra over a number field K.
    -ideal: A fractional ideal in K.

    OUTPUT:
    -Elements of `M_2(\Z_K / ideal)` that are images of the quaternion basis of
    ``B`` by the projection to `O / ideal`, with `O` any maximal order
    containing the basis.

    TODO:
    Somwehat obsolete. Should probably use the SplitAlgebra class and take any order
    as input.
    """
    if not ideal.is_prime() or ideal in B.ramified_primes():
        raise ValueError("The ideal must be a prime that does not ramify in B")
    a, b = B.invariants()
    if a in ideal:
        raise ValueError(
            "The first invariant of the quaternion algebra may\
        not be contained in the ideal"
        )
    F = ideal.residue_field()
    i2 = F(a)
    j2 = F(b)
    i2inv = ~i2
    imod = matrix(F, 2, 2, [0, i2, 1, 0])
    alpha = None
    for z in F:
        if z.is_zero():
            pass
        c = j2 + i2inv * z**2
        if c.is_square():
            alpha = c.sqrt()
            break

    jmod = matrix(F, 2, 2, [alpha, z, -z * i2inv, -alpha])
    kmod = imod * jmod
    return [identity_matrix(F, 2), imod, jmod, kmod]


def quat_mod_I(ideal, x):
    r"""
    Return the image of `x mod ideal` as a matrix over a finite field.

    INPUT:
    -ideal: fractional ideal in the base field of the algebra of ``x``
    -x: quaternion

    OUTPUT:
    -A matrix in `M_2(\Z_K / ideal)`

    TODO:
    see ``quat_mod_I_basis``.
    """
    B = x.parent()
    red_basis = quat_mod_I_basis(B, ideal)
    F = ideal.residue_field()
    return sum([F(c) * e for c, e in zip(x.coefficient_tuple(), red_basis)])


def local_generator(ideal, prime):
    """
    Output a generator of ``ideal`` in the base ring localized at ``prime``.
    """
    v = ideal.valuation(prime)
    return [a for a in ideal.gens() if a.valuation(prime) == v][0]


def pari_data_to_ideal(K, data):
    """
    Convert the output of a PARI method to an ideal.
    """
    basis = K.pari_zk()
    if isinstance(data, list):
        data = matrix([data])
    else:
        data = data.transpose()
    return K.ideal([sum(e * b for e, b in zip(r, basis)) for r in data.rows()])


def narrow_class_group_generators(K):
    """
    Compute a list of ideals of K that generate its narrow class group.
    """
    ideals_data = K.pari_bnf().bnfnarrow().sage()[2]
    return [pari_data_to_ideal(K, data) for data in ideals_data]


def narrow_class_group_support(K):
    """
    Compute a list of prime ideals that generate the narrow class group
    of K
    """
    gens = narrow_class_group_generators(K)
    return {t[0] for gen in gens for t in gen.factor()}


def inbetween(left, right):
    """
    Output a rational number between real numbers
    ``left`` and ``right``.
    """
    scale = (~(right - left)).ceil()
    return (left * scale).ceil() / scale


@cached_function
def full_sign_coverage(K):
    """
    Output a list of elements of ``K`` with each combination of signs at the
    real places of ``K``.
    Assumes that ``K`` is totally real.

    Algorithm 2 from THE ANISOTROPIC PART OF A QUADRATIC FORM OVER A NUMBER FIELD by Koprowski and Rothkegel
    """
    n = K.degree()
    P = K.defining_polynomial()
    roots = [sigma(K.gen()) for sigma in K.embeddings(AA)]
    roots.sort()
    bounds = (
        [roots[0].floor()]
        + [inbetween(roots[i], roots[i + 1]) for i in range(len(roots) - 1)]
        + [roots[-1].ceil()]
    )
    factors = [
        (K.gen() - bounds[i]) * (K.gen() - bounds[i + 1])
        for i in range(len(bounds) - 1)
    ]
    subsets = chain.from_iterable(combinations(list(range(n)), r) for r in range(n + 1))
    res_frac = [prod(factors[i] for i in subset) for subset in subsets]
    return [x * x.denominator() for x in res_frac]


def small_lift(ideal, residue, count=10):
    """
    Outputs a reduced lift of ``residue`` from the residue field of ``ideal``.
    The reduction is done by a solving a CVP
    """
    R = ideal.number_field().ring_of_integers()
    R_bm = matrix([b.list() for b in R.basis()])
    bad_lift = R_bm.solve_left(vector(R(residue)))
    lattice_basis = matrix([R_bm.solve_left(vector(b)) for b in ideal.basis()]).LLL()
    lattice = IntegralLattice(identity_matrix(QQ, 2), lattice_basis)
    close_vectors_gen = lattice.enumerate_close_vectors(bad_lift)
    close_vectors = [next(close_vectors_gen) for _ in range(count)]
    close_vectors.sort(key=lambda v: (bad_lift - v).norm())
    return sum(c * e for c, e in zip(bad_lift - close_vectors[0], R.basis()))


def reduce_mod(ideal, target, count=10):
    """
    Outputs a small element of the ring of integers equivalent to
    ``target`` modulo ``ideal``, which is assumed to be prime.
    """
    k = ideal.residue_field()
    return small_lift(ideal, k(target), count)
