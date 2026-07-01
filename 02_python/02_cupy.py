import numpy as np
import cupy as cp

n = 1_000_000
a = np.arange(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
print(f"a={a}")
print(f"b={b}")

a_gpu = cp.asarray(a) # CPU to GPU
b_gpu = cp.asarray(b) # CPU to GPU

c_gpu = a_gpu + b_gpu # on GPU
s_gpu = cp.dot(a_gpu, b_gpu) # on GPU

c = cp.asnumpy(c_gpu) # GPU to CPU
print(f"c={c}")

