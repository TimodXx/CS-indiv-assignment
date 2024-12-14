import numpy as np
import random
import sympy

# Function to calculate hash values
def hash_func(a, b, x, p):
    return (a + b * x) % p



def min_hash(binary_matrix, k):
    n_rows, n_cols = binary_matrix.shape

    # # k = # number of hash functions
    # k = 1600
    p = sympy.nextprime(n_rows)

    #signmatrix
    M = np.full((k, n_cols), np.inf)

    random.seed(42)  

    a_num = np.array([random.randint(1, p) for _ in range(k)])
    b_num = np.array([random.randint(0, p) for _ in range(k)])

    for r in range(n_rows):
        # for each hash func h_i compute h_i(r)
        h = []
        for i in range(k):
            h.append(hash_func(a_num[i], b_num[i],r, p))
        for c in range(n_cols):
            if binary_matrix[r][c] == 1:
                for j in range(k):
                    if h[j] <= M[j][c]:
                        M[j][c] = h[j]
    
    return M
