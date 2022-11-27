import numpy as np
from mrmustard.lab import *


circuit_1 = (Sgate(r=[0.7, -0.8, -0.4], r_trainable=True, r_bounds=(-1.15,1.15), modes=[0,1,2]) >>
           BSgate(theta=0.5, theta_trainable=True, modes=[0,1]) >> 
           BSgate(np.pi/4, theta_trainable=True, modes=[1,2]) >> Rgate(np.pi/2, modes=[0])
          )

circuit_2 = (Sgate(r=[0.7, -0.8, -0.4], r_trainable=True, r_bounds=(-1.15,1.15), modes=[0,1,2]) >>
           BSgate(theta=0.5, theta_trainable=True, modes=[0,1]) >> Rgate(np.pi/2, modes=[0]) >>
           BSgate(np.pi/4, theta_trainable=True, modes=[1,2]) 
          )

Fock([3,3,3],cutoffs=[30,10,10]) >> circuit_1