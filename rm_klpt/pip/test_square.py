from sage.all import *
proof.all(False)

import time

import pip
import misc

import logging
logging.getLogger('pip').setLevel(logging.DEBUG)

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
print(f'Testing fill_square: {rnd = }')
print('-'*40)
print()

# Parameters setup
p = random_prime(2**250)
while p % 4 != 3:
    p = next_prime(p)

print(f'{p = }')

B = QuaternionAlgebra(-1, -p)
i, j, k = B.gens()
O0 = B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1+k)/2))

print('First test')

ell = random_prime(p)
print(f'{ell = }')

I = random_ideal_prime_norm(ell, O0)

M = pip.fill_square(I)

good = misc.nnorm(M) == ell and M[0,0] in I and M[1,0] in I
print(f'Ok: {good}')

logging.getLogger('pip').setLevel(logging.WARNING)

print()
print('-' * 30)
print('Timing')

tot_time = 0
for i in range(100):
    if (i-1) % 50 == 0:
        print(f'Done {i-1} / 100')

    p = random_prime(2**250)
    while p % 4 != 3:
        p = next_prime(p)

    B = QuaternionAlgebra(-1, -p)
    i, j, k = B.gens()
    O0 = B.maximal_order(order_basis=(B(1), i, (i+j)/2, (1+k)/2))

    ell = random_prime(p)

    I = random_ideal_prime_norm(ell, O0)

    t0 = time.time()
    M = pip.fill_square(I)
    t1 = time.time()
    tot_time += t1 - t0

    good = misc.nnorm(M) == ell and M[0,0] in I and M[1,0] in I
    assert good

print(f'Avg. time: {tot_time/100:.3f}s')

print()
print('-' * 30)
print('Testing sizes for l large')

sizes = [0, 0, 0, 0]

for r in range(100):
    if (r-1) % 50 == 0:
        print(f'Done {r-1} / 100')

    ell = random_prime(p)
    I = random_ideal_prime_norm(ell, O0)

    M = pip.fill_square(I)
    assert misc.nnorm(M) == ell and M[0,0] in I and M[1,0] in I

    n = max(sqrt(p).n(), p/ell)

    sizes[0] += M[0, 0].reduced_norm() / (n * ell)
    sizes[1] += M[0, 1].reduced_norm() / (n * sqrt(p).n())
    sizes[2] += M[1, 0].reduced_norm() / (ell * p)
    sizes[3] += M[1, 1].reduced_norm() / (sqrt(p).n() * p)

sizes = [x / 100.0 for x in sizes]

print('-' * 30)
print(f'{p = }')
print(f'a / nl = {int(sizes[0])}')
print(f'b / n*sqrt(p) = {int(sizes[1])}')
print(f'c / l*p = {sizes[2]:.3f}')
print(f'd / p*sqrt(p) = {sizes[3]:.3f}')
print('-' * 30)

print()
print('-' * 30)
print('Testing sizes for l small prime 1 mod 4')

sizes = [0, 0, 0, 0]

for r in range(100):
    if (r-1) % 50 == 0:
        print(f'Done {r-1} / 100')

    ell = random_prime(isqrt(p))
    while ell % 4 != 1:
        ell = random_prime(isqrt(p))

    I = random_ideal_prime_norm(ell, O0)

    M = pip.fill_square(I)
    assert misc.nnorm(M) == ell and M[0,0] in I and M[1,0] in I

    n = ell

    sizes[0] += M[0, 0].reduced_norm() / (n * ell)
    sizes[1] += M[0, 1].reduced_norm() / (n * sqrt(p).n())
    sizes[2] += M[1, 0].reduced_norm() / (ell * p)
    sizes[3] += M[1, 1].reduced_norm() / (sqrt(p).n() * p)

sizes = [x / 100.0 for x in sizes]

print('-' * 30)
print(f'{p = }')
print(f'a / nl = {int(sizes[0])}')
print(f'b / n*sqrt(p) = {int(sizes[1])}')
print(f'c / l*p = {sizes[2]:.3f}')
print(f'd / p*sqrt(p) = {sizes[3]:.3f}')
print('-' * 30)

print()
print('-' * 30)
print('Testing sizes for l small prime 3 mod 4')

sizes = [0, 0, 0, 0]

for r in range(100):
    if (r-1) % 50 == 0:
        print(f'Done {r-1} / 100')

    ell = random_prime(isqrt(p))
    while ell % 4 != 3:
        ell = random_prime(isqrt(p))
    I = random_ideal_prime_norm(ell, O0)

    M = pip.fill_square(I)
    assert misc.nnorm(M) == ell and M[0,0] in I and M[1,0] in I

    n = max(sqrt(p).n(), p/ell)

    sizes[0] += M[0, 0].reduced_norm() / (n * ell)
    sizes[1] += M[0, 1].reduced_norm() / (n * sqrt(p).n())
    sizes[2] += M[1, 0].reduced_norm() / (ell * p)
    sizes[3] += M[1, 1].reduced_norm() / (sqrt(p).n() * p)

sizes = [x / 100.0 for x in sizes]

print('-' * 30)
print(f'{p = }')
print(f'a / nl = {int(sizes[0])}')
print(f'b / n*sqrt(p) = {int(sizes[1])}')
print(f'c / l*p = {sizes[2]:.3f}')
print(f'd / p*sqrt(p) = {sizes[3]:.3f}')
print('-' * 30)



