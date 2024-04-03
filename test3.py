from time import time

# this is the rust library
import mean_mrmustard as mmm
from mrmustard import math, settings

from mrmustard.lab_dev import Coherent
from mrmustard.math.lattice.strategies import vanilla as mm_vanilla

import numpy as np

shape = tuple([20] * 4)

shape = (5, 5, 5)

A = np.array([[-3.95639080e-18-4.17507872e-20j,  2.27656138e-18-2.16690998e-19j,
        -3.14611309e-19-5.87637652e-18j,  3.33333333e-01-8.70433677e-37j,
        -5.83926984e-17+1.13688612e-16j, -8.12660781e-17+9.62161110e-17j],
       [ 2.27656138e-18-2.16690998e-19j,  2.49482132e-17+2.89520843e-18j,
        -5.76126389e-19-4.70960097e-18j, -5.83926984e-17-1.13688612e-16j,
         3.33333333e-01+1.20118054e-36j,  1.36830210e-17-8.54567337e-18j],
       [-3.14611309e-19-5.87637652e-18j, -5.76126389e-19-4.70960097e-18j,
        -9.94273030e-18+4.92961751e-18j, -8.12660781e-17-9.62161110e-17j,
         1.36830210e-17+8.54567337e-18j,  3.33333333e-01-7.35207860e-35j],
       [ 3.33333333e-01+8.70433677e-37j, -5.83926984e-17-1.13688612e-16j,
        -8.12660781e-17-9.62161110e-17j, -3.95639080e-18+4.17507872e-20j,
         2.27656138e-18+2.16690998e-19j, -3.14611309e-19+5.87637652e-18j],
       [-5.83926984e-17+1.13688612e-16j,  3.33333333e-01-5.88958557e-35j,
         1.36830210e-17+8.54567337e-18j,  2.27656138e-18+2.16690998e-19j,
         2.46716228e-17-2.89520843e-18j, -5.76126389e-19+4.70960097e-18j],
       [-8.12660781e-17+9.62161110e-17j,  1.36830210e-17-8.54567337e-18j,
         3.33333333e-01-1.11686186e-34j, -3.14611309e-19+5.87637652e-18j,
        -5.76126389e-19+4.70960097e-18j,  1.52241392e-33-4.92961751e-18j]])

b = np.array([0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j, 0.-0.j])

c = 0.29629629629629656+3.790712041752127e-35j

tic = time()
res1 = mmm.vanilla_p(tuple(shape), A, b, c, 256).reshape(shape)
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