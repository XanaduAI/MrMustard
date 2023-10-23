from time import time

s = time()
from tensorflow import custom_gradient
e = time()

print(e - s)

# 1290