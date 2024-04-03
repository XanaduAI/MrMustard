from time import time

# this is the rust library
import mean_mrmustard as mmm
from mrmustard import math, settings

from mrmustard.lab_dev import Coherent
from mrmustard.math.lattice.strategies import vanilla as mm_vanilla

state = Coherent([0, 1], x=1)
rep = state.representation

A = rep.A[0]
b = rep.b[0]
c = rep.c[0]
shape = tuple([20] * state.n_modes)

tic = time()
res1 = mmm.vanilla_p(tuple(shape), A, b, c, 128)
toc = time()
print("MeanMrMustard", toc - tic)

tic = time()
res2 = mm_vanilla(tuple(shape), A, b, c)
toc = time()
print("\nMrMustard, Vanilla 64 (Python)", toc - tic)

tic = time()
mm_vanilla(tuple(shape), A, b, c)
toc = time()
print("MrMustard, Vanilla 64 again", toc - tic)

tic = time()
settings.PRECISION_BITS_HERMITE_POLY = 256
toc = time()
print("\nMrMustard initializes Julia", toc - tic)

tic = time()
res3 = math.hermite_renormalized(A, b, c, tuple(shape))
toc = time()
print("MrMustard, Vanilla Julia", toc - tic)

tic = time()
math.hermite_renormalized(A, b, c, tuple(shape))
toc = time()
print("MrMustard, Vanilla Julia again", toc - tic)

tic = time()
math.hermite_renormalized(A, b, c, tuple(shape))
toc = time()
print("MrMustard, Vanilla Julia again", toc - tic)

assert math.allclose(res1, res2)
assert math.allclose(res1, res3)
print("\nDone!!")