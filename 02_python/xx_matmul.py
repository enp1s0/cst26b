import time
import numpy as np
#import cupy as np

N=8192

rng = np.random.default_rng()
A = rng.random((N, N), dtype=np.float32) 
B = rng.random((N, N), dtype=np.float32) 

start_time = time.perf_counter()
C = np.dot(A, B)
elapsed_time = time.perf_counter() - start_time

print(f"matmul computing time: {elapsed_time:.6f} seconds")
