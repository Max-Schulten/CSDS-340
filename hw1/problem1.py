import numpy as np

# 1a)
a = np.full((6, 4), 2)
print(a)

# 1b)
b = np.ones((6,4), dtype=int)
np.fill_diagonal(b, 3, )
print(b)

# 1c)
print(a*b)
try:
    print(np.dot(a, b))
except ValueError as e:
    print(e)

# 1d)
print(np.dot(a, b.transpose()))
print(np.dot(a.transpose(), b))