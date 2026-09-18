from sage.all import *
proof.all(False)

import time

import pip
import misc

import logging
logging.getLogger('pip').setLevel(logging.INFO)

def random_ideal_prime_norm(N, O):
    # Generate a random O-ideal of given (prime) norm
    B = O.quaternion_algebra()
    FN = GF(N)

    while True:
        g1, g2, g3 = [randint(0, N-1) for _ in range(3)]
        gamma = B([0, g1, g2, g3])
        n_gamma = gamma.reduced_norm()
        if kronecker(-n_gamma, N) == 1:
            gamma += ZZ(FN(-n_gamma).sqrt())
            break

    # Scaling by a random beta mod N
    while True:
        x, y, z, w = [randint(0, N-1) for _ in range(4)]
        beta = B([x, y, z, w])
        if gcd(beta.reduced_norm(), N) == 1:
            break
    return O * (gamma * beta) + O * N



rnd = randint(1, 2**32)
set_random_seed(rnd)

print('-'*40)
print(f'Testing kernel_to_matrix: {rnd = }')
print('-'*40)
print()

tot_time = 0
for _ in range(10):

    ell = random_prime(2**20)
    while ell == 2:
        ell = random_prime(2**20)

    c = 1
    while True:
        p = c*ell*2**240 - 1
        if is_prime(p):
            break
        c += 1

    print()
    print(f'{p = } {ell = }')

    Fp2, ii = GF((p, 2), name='ii', modulus=[1,0,1]).objgen()

    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O0 = B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1+k)/2))

    E0 = EllipticCurve(Fp2, [1,0])
    for quat_i in E0.automorphisms():
        if quat_i.order() == 4:
            break # (x, y) -> (-x, iy)
    quat_j = E0.frobenius_isogeny()
    D = (E0, quat_i, quat_j)

    # Generate the kernel (P, Q)
    cof = (p+1)//ell
    P = Q = E0(0)
    while P == 0:
        P = E0.random_point() * cof
    while Q == 0:
        Q = E0.random_point() * cof
    K = (P, Q)

    # Testing kernel point -> isogeny

    t0 = time.time()
    M = pip.kernel_to_matrix(K, O0, ell, D)
    t1 = time.time()
    tot_time += t1 - t0

    good = misc.nnorm(M) == ell and misc.eval_matrix(M, P, Q, D) == (0, 0)
    assert good

print(f'Avg. time: {tot_time/10:.3f}s')
