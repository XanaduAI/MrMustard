from mrmustard.lab_dev import *

coh0123 = Coherent([0, 1, 2, 3])
bs01 = BSgate(modes=[0, 1], theta=0.1, phi=0.2)
bs12 = BSgate(modes=[1, 2], theta=0.3, phi=0.4)
bs23 = BSgate(modes=[2, 3], theta=0.5, phi=0.6)
a0 = Attenuator(modes=[0], transmissivity=0.8)
a1 = Attenuator(modes=[1], transmissivity=0.8)
a2 = Attenuator(modes=[2], transmissivity=0.7)
a0123 = Attenuator(modes=[0, 1, 2, 3], transmissivity=0.7)

def f():
    for _ in range(100):
        coh0123 >> bs01 >> bs12 >> bs23 >> a0123

from scalene import scalene_profiler

# Turn profiling on
scalene_profiler.start()
f()
scalene_profiler.stop()
print("\ney")