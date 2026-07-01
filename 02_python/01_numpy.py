import numpy as np

n = 1_000_000
a = np.arange(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
print(f"a={a}")
print(f"b={b}")

c = a + b # vector addition
s = np.dot(a, b) # inner product

print(f"c={c}")
print(f"s={s}")
