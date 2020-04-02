import dill
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

M = 1000
d1 = []
for i in range(M):
    d1.append({})
    for j in range(M):
        d1[i][j] = 0.0
dill.dump(d1, open('d1.p', 'wb'))

d2 = {}
for i in range(M):
    d2[i] = {}
    for j in range(M):
        d2[i][j] = 0.0
dill.dump(d2, open('d2.p', 'wb'))

