from mrmustard.lab_dev.opt_contraction import optimal_path, staircase
from time import time

if __name__ == "__main__":
    
    tic = time()
    c, sol = optimal_path(staircase(3))
    toc = time()
    print()
    print("*" * 50)
    print("Final cost:", " + ".join([str(s[0]) for s in sol]), "=", c)
    print("Final solution:", [s[1] for s in sol])
    print("Time:", toc - tic)