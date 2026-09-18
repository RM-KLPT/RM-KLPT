# RM_DEURING

This is an imlementation of a generalisation of the KLPT algorithm
to quaternion algebras over real quadratic fields with trivial strict
class group and small discriminant.

Also implements a lot of utility functions for the algebra of
superspecial abelian varieties with real multiplication.

## Running

This code was tested using Sage version 10.8 and Python version 3.14.4.
See the notebook `Example.ipynb` for an example of the KLPT computation.

## File organisation

- `rm_deuring/Example.ipynb` presents an example use of the KLPT algorithm.
- `rm_deuring/Heuristics.ipybb` presents computation verifying heuristics upon which our version of KLPT relies.
- `rm_deuring/finite_algebras.py` contains a class computing isomorphisms to matrix algebras over finite fields.
- `rm_deuring/hnf.py` contains a class implementing utility and hermite form computation for pseudo matrices.
- `rm_deuring/nf_utility.py` implements utility functions for computation on number fields.
- `rm_deuring/quaternion_ideal.py` implements a class for fractional ideals in quaternion algebras over number fields. In particular, contains the implementation of the KLPT algorithm.
- `rm_deuring/quaternion_matrix.py` contains a wrapper class for matrices over rational quaternion algebras.
- `rm_deuring/quaternion_order.py` contains a class for orders in quaternion algebras over number fields. In particular, contains the implementation of `strong_approximation` for the KLPT algorithm.
- `rm_deuring/represent_integer.py` contains a norm equation solver for quaternion orders
containing a square root of -1 and of -p, for p a large prime equal to 3 mod 4.
- `rm_deuring/rm_data` contains a class representing real multiplications over
a superspecial abelian variety.
