# Principal Ideal Problem

Code accompaining the paper "The principal ideal problem for endomorphism rings
of superspecial abelian varieties". This code contains some function for
solving the principal ideal problem and the kernel to matrix problem in
dimension 2. We refer to the paper for a more detailed explanation of the
functions.

## Project Structure

- `pip.py`: main algorithms
- `ideals.py`: ideals helpers
- `misc.py`: generic helpers
- `bench.py`: test functions
- `qmat_iso.py` and `qmat_iso_q.py`: isomorphisms between quaternion orders
  modulo n and 2x2 matrices

## Testing the code

The following files can be used to test and benchmark the code, and as an
example for its usage:
- `test_square.py`: tests the function `fill_square` on random O0-ideals,
  checking that the matrix output has the correct norm and that the first
  column generates the input ideal; it also computes average timings and output
  sizes veryfing the claims of Sec. 4.4
- `test_kernel.py`: generates a pair of points `(P, Q)` of `l`-torsion and checks
  that the function `kernel_to_matrix` returns a matrix of norm `l` with `(P,
  Q)` as a kernel

To run the tests simply run `sage test_xxx.py` or `sage --python -O test_xxx.py`
to skip assertions.
